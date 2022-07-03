from typing import Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SimpleLstmActor(nn.Module):

    #TODO annotate attrs

    def __init__(self, observation_shape: Any, n_actions: Any):
        super().__init__()
        self.observation_dims = len(observation_shape)
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64), std=0.01),
        )
        self.lstm = nn.LSTM(64, 32)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.logits = layer_init(nn.Linear(32, n_actions), std=0.01)

    # Supports the following shapes for 'x' ('done' must have the same shape):
    # - (I): a 1-long sequence of inputs, single batch
    # - (B, I): a 1-long sequence of inputs, multiple batches
    # - (L, B, I): a L-long sequence of inputs, multiple batches
    # The first LSTM state should also be compatible with the shape of 'x' (TODO doc).
    # outputs are:
    # - action with shape (1,) or (L,) or (L, B), depending on input shape
    # - logprobs with same shape as actions
    # - entropy with same shape as actions
    # where:
    # - I is the shape of the input
    # - L is the BPTT truncation length
    # - B is a batch size
    def forward(self, x: torch.Tensor, done: torch.Tensor, action: Optional[torch.Tensor] = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        while len(x.size()) - self.observation_dims < 2:
            x = x.unsqueeze(0)
            done = done.unsqueeze(0)
        # NOTE: make sure to use networks that accept batches shaped along multiple axes (e.g. linear layers etc.).
        z_network = self.network(x)
        #TODO test if this can directly be allocated as tensor of shape (L, B, ...)
        zs_lstm = []
        # Unroll the LSTM manually since we need to intercept the state at each time step.
        for t in range(x.size(0)):
            # Reset the LSTM states of envs that were just reset to 0s. 
            self.lstm_state = [self.lstm_state[i] * (1. - done[t]).view(1, -1, 1) for i in [0, 1]]
            #print(f'zsize: {z_network.size()}, hsize: {self.lstm_state[0].size()}')
            z_lstm_t, self.lstm_state = self.lstm(z_network[t].unsqueeze(0), self.lstm_state)
            zs_lstm.append(z_lstm_t)
        self.lstm_state = list(self.lstm_state)
        z_lstm = torch.cat(zs_lstm, dim=0)
        #print(z_lstm.size())
        logits = self.logits(z_lstm)
        distr = Categorical(logits=logits)
        if action is None:
            action = distr.mode if deterministic else distr.sample()
        logprob = distr.log_prob(action)
        entropy = distr.entropy()
        return action, logprob, entropy


class SimpleLstmCritic(nn.Module):

    #TODO annotate attrs

    def __init__(self, observation_shape: Any):
        super().__init__()
        self.observation_dims = len(observation_shape)
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64), std=0.01),
        )
        self.lstm = nn.LSTM(64, 32)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.value = layer_init(nn.Linear(32, 1), std=0.01)

    # Supports the following shapes for 'x':
    # - (B, I): a 1-long sequence of inputs, multiple batches
    # - (L, B, I): a L-long sequence of inputs, multiple batches
    # 'done' should always refer only to the first inputs of the sequence, so the corresponding shapes would be:
    # - scalar when shape x is (I) or (L, I)
    # - (B,) when shape is (L, B, I)
    # where:
    # - I is the shape of the input
    # - L is the BPTT truncation length
    # - B is a batch size
    def forward(self, x: torch.Tensor, done: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(x.size()) - self.observation_dims == 1:
            x = x.unsqueeze(0)
            done = done.unsqueeze(0)
        # NOTE: make sure to use networks that accept batches shaped along multiple axes (e.g. linear layers etc.).
        z_network = self.network(x)
        # Unroll the LSTM manually since we need to intercept the state at each time step.
        zs_lstm = []
        for t in range(x.size(0)):
            # Reset the LSTM states of envs that were just reset to 0s. 
            self.lstm_state = [self.lstm_state[i] * (1. - done[t]).view(1, -1, 1) for i in [0, 1]]
            #print(f'zsize: {z_network.size()}, hsize: {self.lstm_state[0].size()}')
            z_lstm_t, self.lstm_state = self.lstm(z_network[t].unsqueeze(0), self.lstm_state)
            zs_lstm.append(z_lstm_t)
        self.lstm_state = list(self.lstm_state)
        z_lstm = torch.cat(zs_lstm, dim=0)
        value = self.value(z_lstm)
        return value.squeeze(-1)

