import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym.spaces

# modified version of the CombinedExtractor that uses self multihead attention on the specified keys to support multiple entities, also supporting an attention mask in the forward
class MhSaExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, entity_keys={}, nonent_features=32, ent_features=32, activation=torch.tanh, sa_heads=4):
        assert isinstance(observation_space, gym.spaces.Dict)
        
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)
        
        self.ent_keys = [
            k for k in observation_space.spaces.keys()
            if k in entity_keys
        ]
        self.nonent_keys = [
            k for k in observation_space.spaces.keys()
            if k not in entity_keys
        ]
        self.activation = activation
        
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
        
        self.attention = nn.MultiheadAttention(embed_dim=ent_features, num_heads=sa_heads)

        # Update the features dim manually
        self._features_dim = nonent_features + ent_features

    def forward(self, observations):

        # input: dict with vals of shapes (N, P_i)
        # output: (N, P=sum{P_i})
        x_nonent = torch.cat([observations[k] for k in self.nonent_keys], dim=1)
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
        # z_sa: (L, N, E)
        z_sa, _ = self.attention(z_ents_seq, z_ents_seq, z_ents_seq)
        # z_ents: (N, E)
        # avg pooling over all entities: is this sensible?
        # NOTE: remember to include masking when mask available
        z_ents = torch.mean(z_sa.transpose(0, 1), dim=1)

        # z_all: (N, E+E_nonent)
        z_all = torch.cat([z_nonent, z_ents], dim=1)

        return z_all