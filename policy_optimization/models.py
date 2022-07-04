from typing import Any, List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils import repeat, recursive_apply


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

#TODO double check this works as intended for all OPs
def map_double_batched(f, *args):
    batch_sizes = args[0].size()[0:2]
    flat_args = [arg.flatten(end_dim=1) for arg in args]
    flat_outputs = f(*flat_args)
    if isinstance(flat_outputs, tuple):
        return tuple(y.unflatten(0, batch_sizes) for y in flat_outputs)
    else:
        return flat_outputs.unflatten(0, batch_sizes)

def assert_dim(x, dim):
    assert x.dim() == dim, f'Invalid number of dimensions: {x.dim()}, expected {dim}'


class EgocentricHiveMind(nn.Module):

    #TODO initialize layers like in h&s

    def __init__(self, action_space: Tuple[int, ...], x_agent_size: int, x_lidar_size: int, entity_keys: List[str], lidar_conv_features: int = 9, lidar_conv_kernel_size: int = 3, entity_features: int = 64, attention_heads: int = 4, attention_head_size: int = 16, lstm_input_features: int = 128, lstm_size: int = 128):
        super().__init__()
        assert entity_features % 2 == 0
        self.entity_keys = entity_keys
        def _input_sizes(self, entity_sizes: Dict[int, str], seq_length: int = 1, batch_size: int = 1, n_entities: int = 1):
            return (
                # self
                (seq_length, batch_size, x_agent_size),
                # lidar
                (seq_length, batch_size, x_lidar_size),
                # entities
                {k: (seq_length, batch_size, n_entities, v) for k, v in entity_sizes.items()},
                # done
                (seq_length, batch_size)
            )
        self.input_sizes = _input_sizes.__get__(self)
        self.lidar_conv = nn.Conv1d(1, lidar_conv_features, kernel_size=lidar_conv_kernel_size, padding_mode='circular')
        self.self_dense = nn.LazyLinear(entity_features)
        self.entity_denses = {k: nn.LazyLinear(entity_features) for k in entity_keys}
        self.mh_attention = nn.MultiheadAttention(attention_head_size*attention_heads, attention_heads, batch_first=True)
        self.sa_dense = nn.LazyLinear(entity_features)
        self.pool_dense = nn.LazyLinear(lstm_input_features)
        self.lstm = nn.LSTM(lstm_input_features, lstm_size)
#         for name, param in self.lstm.named_parameters():
#             if "bias" in name:
#                 nn.init.constant_(param, 0)
#             elif "weight" in name:
#                 nn.init.orthogonal_(param, 1.0)
        self.heads = [nn.LazyLinear(n_actions) for n_actions in action_space]

    # placeholder for the method dynamically bound in __init__
    def input_sizes(self):
        raise NotImplementedError

    def zero_inputs(self, entity_sizes: Dict[str, int], seq_length: int = 1, batch_size: int = 1, n_entities: int = 1):
        sizes = list(self.input_sizes(entity_sizes, seq_length, batch_size, n_entities))
        return recursive_apply(lambda sz: torch.zeros(sz), sizes)

    def zero_lstm_state(self, batch_size: int = 1):
        return [torch.zeros(1, batch_size, self.lstm.hidden_size) for _ in [0,1]]

    # For input sizes see _input_sizes in __init__. Only accepts batched inputs for now.
    #TODO also accept non-batched input
    #TODO support masking entities (e.g. because of padded entities in batches)
    def forward(self, x_agent: torch.Tensor, x_lidar: torch.Tensor, x_entities: Dict[str, torch.Tensor], done: torch.Tensor, actions: Optional[List[torch.Tensor]] = None, deterministic: bool = False):

        [assert_dim(x, 3) for x in [x_agent, x_lidar] + ([actions] if actions is not None else [])]
        recursive_apply(lambda x: assert_dim(x, 4), x_entities)
        assert_dim(done, 2)

        z_lidar = torch.flatten(
            F.relu(
                map_double_batched(
                    self.lidar_conv, x_lidar.unsqueeze(-2)
                )
            ),
            start_dim=2
        )
        x_self = torch.cat([x_agent, z_lidar], axis=-1)
        #TODO is the agent itself counted as an entity? the openreview comments seem to implicate both options (inclusion is also suggested by the figure in the paper, although not from its appendix B)
        z_entities_list = [F.relu(self.self_dense(torch.cat([x_self, x_self], axis=-1))).unsqueeze(-2)]
        for k in self.entity_keys:
            x = x_entities[k]
            z_entities_list.append(
                F.relu(
                        self.entity_denses[k](
                            torch.cat([
                                x,
                                x_self.unsqueeze(-2).expand(-1, -1, x.size(-2), -1)
                            ], axis=-1)
                    )
                )
            )
        z_entities = torch.cat(z_entities_list, axis=-2)

        residual_sa = z_entities + F.relu(
            self.sa_dense(
                map_double_batched(
                    self.mh_attention, z_entities, z_entities, z_entities
                )[0]
            )
        )

        #TODO is this correct? seems so from the author's comments on openreview
        # "Avg pool" across entities: see https://openreview.net/forum?id=SkxpxJBKwS&noteId=BJxWUARNsr and https://openreview.net/forum?id=SkxpxJBKwS&noteId=ryeCRG40Or for details
        #TODO use the mask
        z_pool = torch.mean(residual_sa, dim=-2)
        z = F.relu(
            self.pool_dense(
                torch.cat([x_self, z_pool], axis=-1)
            )
        )

        zs_lstm = []
        # Unroll the LSTM manually since we need to intercept the state at each time step.
        for t in range(x.size(0)):
            # Reset the LSTM states of envs that were just reset to 0s. 
            self.lstm_state = [self.lstm_state[i] * (1. - done[t]).view(1, -1, 1) for i in [0, 1]]
            #print(f'zsize: {z_network.size()}, hsize: {self.lstm_state[0].size()}')
            z_lstm_t, self.lstm_state = self.lstm(z[t].unsqueeze(0), self.lstm_state)
            zs_lstm.append(z_lstm_t)
        self.lstm_state = list(self.lstm_state)
        z_lstm = torch.cat(zs_lstm, dim=0)
        distrs = [Categorical(logits=head(z_lstm)) for head in self.heads]
        if actions is None:
            if deterministic:
                actions = torch.cat([distr.mode.unsqueeze(-1) for distr in distrs], axis=-1)
            else:
                actions = torch.cat([distr.sample().unsqueeze(-1) for distr in distrs], axis=-1)
        entropies = torch.cat([distr.entropy().unsqueeze(-1) for distr in distrs], axis=-1)
        logprobs = torch.cat([distrs[i].log_prob(actions[:,:,i]).unsqueeze(-1) for i in range(actions.size(0))], axis=-1)
        return actions, logprobs, entropies


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

