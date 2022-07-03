from typing import Any, Union, Tuple, Callable, List, Dict, Optional

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim

from batch_buffer import BatchBuffer
from gae import general_advantage_estimation
from utils import repeat, callmethod, recursive_apply


# Abbreviations used in comments:
# - NA := number of agents
# - NE := number of parallel envs
# - L := BPTT truncation length
# - B := buffer size (in BPTT chunks)
# - H := GAE horizon

def bptt_chunks(x, chunk_size: int):
    if isinstance(x, BatchBuffer):
        x = x.buffer
    dims = x.size()
    assert dims[0] % chunk_size == 0
    n_chunks = dims[1] * (dims[0] // chunk_size)
    # The shape without the horizon and number of envs dimensions.
    shape = list(dims[2:]) if len(dims) > 2 else []
    xT = x.transpose(1, 0)
    chunks = xT.reshape(n_chunks, chunk_size, *shape).transpose(1, 0)
    return chunks

def unbatch_lstm_states(h):
    # h: (1, H/L, NE) -> (1, (H/L)*NE)
    # output is orderder like this:
    # [ s1e1 s2e1 ... sNe1
    if isinstance(h, BatchBuffer):
        h = h.buffer
    dims = h.size()
    assert dims[0] == 1
    shape = list(dims[3:]) if len(dims) > 3 else []
    return h.transpose(2, 1).reshape(1, dims[1]*dims[2], *shape)


class Ppo:

    lstm_keys: List[str] = ['actor', 'critic']

    def __init__(self, envs, actor, critic, ppo_clipping, entropy_coefficient, value_coefficient, max_grad_norm, learning_rate, epochs_per_step, observation_shape, bptt_truncation_length, gae_horizon, buffer_size, minibatch_size, gamma, gae_lambda):
        self.envs = envs
        self.actor = actor
        self.critic = critic
        self.ppo_clipping = ppo_clipping
        self.loss_coefficients = {'entropy': entropy_coefficient, 'value': value_coefficient}
        self.max_grad_norm = max_grad_norm
        self.epochs_per_step = epochs_per_step
        self.bptt_truncation_length = bptt_truncation_length
        self.gae_params = {'gamma': gamma, 'lambda_': gae_lambda}
        assert gae_horizon % bptt_truncation_length == 0
        assert buffer_size % (len(envs)*(gae_horizon // bptt_truncation_length)) == 0
        assert buffer_size % minibatch_size == 0
        self.minibatch_size = minibatch_size
        self.minibatches_per_buffer = buffer_size // minibatch_size
        self.rng = torch.Generator()
        buf_shape = (self.bptt_truncation_length, buffer_size)
        self.buffers = {
            'observations': BatchBuffer(buf_shape + observation_shape, batch_axis=1), #repeat(BatchBuffer, A, buf_shape + observation_shape),
            'dones': BatchBuffer(buf_shape, batch_axis=1), # repeat(BatchBuffer, A, buf_shape),
            'values': BatchBuffer(buf_shape, batch_axis=1), # repeat(BatchBuffer, A, buf_shape),
            'actions': BatchBuffer(buf_shape, batch_axis=1), # repeat(BatchBuffer, A, buf_shape + action_shape),
            'logprobs': BatchBuffer(buf_shape, batch_axis=1), # repeat(BatchBuffer, A, buf_shape + action_shape),
            'entropies': BatchBuffer(buf_shape, batch_axis=1), # repeat(BatchBuffer, A, buf_shape + action_shape),
            'rewards': BatchBuffer(buf_shape, batch_axis=1), # repeat(BatchBuffer, A, buf_shape),
            'advantages': BatchBuffer(buf_shape, batch_axis=1), # repeat(BatchBuffer, A, buf_shape),
            'returns': BatchBuffer(buf_shape, batch_axis=1), # repeat(BatchBuffer, A, buf_shape),
            'lstm_states': {k: repeat(BatchBuffer, 2, (1, buf_shape[1], getattr(self, k).lstm.hidden_size), batch_axis=1) for k in self.lstm_keys}
        }
        gae_buf_shape = (gae_horizon, len(self.envs))
        self.gae_buffers = {
            'observations': BatchBuffer(gae_buf_shape + observation_shape), # repeat(BatchBuffer, A, gae_buf_shape + observation_shape),
            'dones': BatchBuffer(gae_buf_shape), # repeat(BatchBuffer, A, gae_buf_shape),
            'values': BatchBuffer(gae_buf_shape), # repeat(BatchBuffer, A, gae_buf_shape),
            'actions': BatchBuffer(gae_buf_shape), # repeat(BatchBuffer, A, gae_buf_shape + action_shape),
            'logprobs': BatchBuffer(gae_buf_shape), # repeat(BatchBuffer, A, gae_buf_shape + action_shape),
            'entropies': BatchBuffer(gae_buf_shape), # repeat(BatchBuffer, A, gae_buf_shape + action_shape),
            'rewards': BatchBuffer(gae_buf_shape), # repeat(BatchBuffer, A, gae_buf_shape),
            'advantages': BatchBuffer(gae_buf_shape), # repeat(BatchBuffer, A, gae_buf_shape),
            'returns': BatchBuffer(gae_buf_shape), # repeat(BatchBuffer, A, gae_buf_shape),
        }
        chunks_per_gae_horizon = gae_horizon//bptt_truncation_length
        self.lstm_buffers = {k: repeat(BatchBuffer, 2, (1, chunks_per_gae_horizon, gae_buf_shape[1], getattr(self, k).lstm.hidden_size), batch_axis=1) for k in self.lstm_keys}
        self.last_lstm_states = {k: repeat(torch.zeros, 2, (1, gae_buf_shape[1], getattr(self, k).lstm.hidden_size)) for k in self.lstm_keys}
        #TODO is it ok to use 1 optimizer for 2 models?
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate, eps=1e-5)

    def test(self, env: gym.Env, episodes: int = 1) -> List[np.ndarray]:
        rewardss = []
        for episode in range(episodes):
            rewards = np.array(self.test_episode(env))
            print(f'Test rewards: R = {rewards.sum()}, r ~ {rewards.mean()} +- {rewards.std()}')
            rewardss.append(rewards)
        return rewardss

    # assumes action space has only 1 dim
    def test_episode(self, env: gym.Env) -> List[float]:
        for k in self.lstm_keys:
            model = getattr(self, k)
            model.lstm_state = [torch.zeros(1, 1, model.lstm.hidden_size) for i in [0,1]]
        done = True
        x = env.reset()
        rewards = []
        while True:
            env.render()
            a, _1, _2 = self.actor(torch.as_tensor(x).unsqueeze(0), torch.as_tensor(float(done)), deterministic=True)
            x, r, done, _ = env.step(a[0].item())
            rewards.append(r)
            if done:
                break
        return rewards

    def train(self, steps: int):
        for step in range(steps):
            self.step()

    def step(self):
        self.watch()
        self.learn()

    def watch(self):
        self.load_last_lstm_states()
        while not self.buffers['dones'].full:
            gae_batch = {}
            observations = self.envs.observations
            dones = self.envs.dones
            gae_batch['observations'] = observations.unsqueeze(0).clone()
            gae_batch['dones'] = dones.unsqueeze(0).clone()
            if self.gae_buffers['dones'].size % self.bptt_truncation_length == 0:
                recursive_apply(lambda buf, batch: buf.append(batch.unsqueeze(1)), self.lstm_buffers, {k: getattr(self, k).lstm_state for k in self.lstm_keys})
            with torch.no_grad():
                #TODO should also handle multiple agents, i.e. return a list of batches
                gae_batch['values'] = self.critic(observations, dones)
                #TODO should also handle multiple agents, i.e. return a list of batches
                gae_batch['actions'], gae_batch['logprobs'], gae_batch['entropies'] = self.actor(observations, dones)
            self.envs.step(gae_batch['actions'].squeeze(0))
            gae_batch['rewards'] = self.envs.rewards.unsqueeze(0).clone()
            recursive_apply(callmethod('append'), self.gae_buffers, gae_batch)
            if self.gae_buffers['dones'].full:
                dones_next = self.envs.dones
                with torch.no_grad():
                    values_next = self.critic(self.envs.observations, dones_next)
                self.gae_buffers['advantages'], self.gae_buffers['returns'] = general_advantage_estimation(self.gae_buffers['rewards'].buffer, self.gae_buffers['values'].buffer, self.gae_buffers['dones'].buffer, values_next, dones_next, **self.gae_params)
                batch = recursive_apply(lambda b: bptt_chunks(b, self.bptt_truncation_length), self.gae_buffers)
                lstm_batch = {'lstm_states': recursive_apply(unbatch_lstm_states, self.lstm_buffers)}
                recursive_apply(callmethod('append'), self.buffers, batch)
                recursive_apply(callmethod('append'), self.buffers, lstm_batch)
                recursive_apply(lambda b: b.flush() if isinstance(b, BatchBuffer) else None, self.gae_buffers)
                recursive_apply(callmethod('flush'), self.lstm_buffers)
        self.save_last_lstm_states()

    def learn(self):
        #torch.set_printoptions(precision=8)
        #TODO LR annealing
        #TODO normalizations
        for epoch in range(self.epochs_per_step):
            for i, minibatch_id in enumerate(torch.randperm(self.minibatches_per_buffer, generator=self.rng)):
                self.minibatch_step(minibatch_id, debug_first_step=(i == 0 and epoch == 0))
        recursive_apply(lambda b: b.flush() if isinstance(b, BatchBuffer) else None, self.buffers)

    def minibatch_step(self, minibatch_id: int, debug_first_step: bool = False):
        a, b = self.minibatch_size * torch.tensor([minibatch_id, minibatch_id + 1])
        for k in self.lstm_keys:
            getattr(self, k).lstm_state = [self.buffers['lstm_states'][k][i].buffer[:,a:b,...] for i in [0,1]]
        mb_observations = self.buffers['observations'].buffer[:,a:b,...]
        mb_dones = self.buffers['dones'].buffer[:,a:b,...]

        #TODO why sometimes logprobs and values at epoch 0 step 0 or not the same (but very close) w.r.t. the buffer? float precision?
        
        _, newlogprob, entropy = self.actor(mb_observations, mb_dones, action=self.buffers['actions'].buffer[:,a:b,...])
        logratio = newlogprob - self.buffers['logprobs'].buffer[:,a:b,...]
        if debug_first_step:
            ok = torch.allclose(logratio, torch.tensor(0.))
            if not ok and logratio.mean() + logratio.std() > 1e-7:
                print(f'wrong logratio: {logratio.mean()} +- {logratio.std()}')
        ratio = logratio.exp()
        mb_advantages = self.buffers['advantages'].buffer[:,a:b,...]
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.ppo_clipping, 1 + self.ppo_clipping)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        entropy_loss = entropy.mean()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs = ((ratio - 1.0).abs() > self.ppo_clipping).float().mean().item()
            #print(f'[DEBUG] KL ~= {old_approx_kl} ~= {approx_kl}; clipfracs = {clipfracs}')

        newvalue = self.critic(mb_observations, mb_dones)
        if debug_first_step:
            ok = torch.allclose(newvalue, self.buffers['values'].buffer[:,a:b,...])
            if not ok:
                print(f'wrong vals, delta = {newvalue - self.buffers["values"].buffer[:,a:b,...]}, oldvals: {self.buffers["values"].buffer[:,a:b,...]}, newvals: {newvalue}')
        v_loss = 0.5 * ((newvalue - self.buffers['returns'].buffer[:,a:b,...]) ** 2).mean()

        loss = pg_loss - self.loss_coefficients['entropy'] * entropy_loss + v_loss * self.loss_coefficients['value']

        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def save_last_lstm_states(self):
        for k in self.lstm_keys:
            self.last_lstm_states[k] = getattr(self, k).lstm_state

    def load_last_lstm_states(self):
        for k in self.lstm_keys:
            getattr(self, k).lstm_state = self.last_lstm_states[k]
