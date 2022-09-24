import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym.spaces

# modified version of the CombinedExtractor that uses self multihead attention on the specified keys to support multiple entities, also supporting an attention mask in the forward
# NOTE: each observation passed to forwards needs at least 1 entity
class MhSaExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, entity_keys={}, omniscent=True, nonent_features=32, ent_features=32, activation=torch.tanh, sa_heads=4):
        assert isinstance(observation_space, gym.spaces.Dict)
        
        # This relies on further code (in this method) to set self._features_dim to the correct value
        super().__init__(observation_space, features_dim=1)
        
        self.omniscent = omniscent
        
        self.ent_keys = [
            k for k in observation_space.spaces.keys()
            if k in entity_keys
        ]
        self.nonent_keys = [
            k for k in observation_space.spaces.keys()
            if k not in entity_keys and not k.endswith('_mask')
        ]
        self.activation = activation
        print(f'ent_keys = {self.ent_keys}, nonent_keys = {self.nonent_keys}')
        
        for k in self.nonent_keys:
            S = observation_space.spaces[k]
            assert len(S.shape) == 1, f'{S} is not uni-dimensional'
        
        nonent_flat_dim = sum(
            observation_space.spaces[k].shape[0]
            for k in self.nonent_keys
        )
        self.nonent_dense = nn.Linear(nonent_flat_dim, nonent_features)
        
        ent_denses = {}
        for k in self.ent_keys:
            S = observation_space.spaces[k]
            assert len(S.shape) == 2, f'{S} is not bi-dimensional'
            ent_denses[k] = nn.Linear(S.shape[1], ent_features)
        self.ent_denses = nn.ModuleDict(ent_denses)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=ent_features, num_heads=sa_heads
        )

        # Update the features dim manually
        self._features_dim = nonent_features + ent_features

    def forward(self, observations):

        # input: dict with vals of shapes (N, P_i)
        # output: (N, P=sum{P_i})
        x_nonent = torch.cat(
            [observations[k] for k in self.nonent_keys], dim=1
        )
        # output: (N, E_nonent)
        z_nonent = self.activation(self.nonent_dense(x_nonent))

        z_ents_list = []
        for k in self.ent_keys:
            # observations[k]: (N, L_i, -)
            # x: (L_i, N, -)
            x = observations[k].transpose(0, 1)
            # z: (L_i, N, E)
            z = self.activation(self.ent_denses[k](x))
            z_ents_list.append(z)
        # z_ents_seq: (L=sum{L_i}, N, E)
        z_ents_seq = torch.cat(z_ents_list, dim=0)
        
        # Prepare entity mask.
        if not self.omniscent:
            # ent_mask (here): (N, L)
            ent_mask = torch.cat(
                [observations[k + '_mask'].bool() for k in self.ent_keys],
                dim=1
            )
            # Save original ent_mask for later use in avg pooling.
            packed_ent_mask = ent_mask
            # from (N, L) to (N, 1, L)
            ent_mask = ent_mask.unsqueeze(1)
            # from (N, 1, L) to (N, L, L)
            ent_mask = ent_mask.expand(-1, ent_mask.size(2), -1)
            # from (N, L, L) to (N*num_heads, L, L)
            # Using repeat_interleave instead of tile for the correct order (batch, head); also see https://github.com/ttumiel/pytorch/commit/a94df5eb9851fc6c711e9eb5e12e3da743b6b544
            ent_mask = ent_mask.repeat_interleave(
                self.attention.num_heads, dim=0
            )

        mask_args = {} if self.omniscent else dict(attn_mask=ent_mask)
        # z_sa: (L, N, E)
        z_sa, _ = self.attention(
            z_ents_seq, z_ents_seq, z_ents_seq,
            **mask_args
        )

        # Mask invisible entities also in the avg pooling.
        if not self.omniscent:
            # pooling_mask: (L, N, 1)
            pooling_mask = packed_ent_mask.t().unsqueeze(2)
            z_sa *= pooling_mask

        # z_ents: (N, E)
        # avg pooling over all entities
        z_ents = torch.mean(z_sa.transpose(0, 1), dim=1)

        # z_all: (N, E+E_nonent)
        z_all = torch.cat([z_nonent, z_ents], dim=1)

        return z_all
