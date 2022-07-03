from typing import Tuple, Optional, Any, List

import torch


def general_advantage_estimation(rewards, values, dones, values_next, dones_next, gamma: float, lambda_: float) -> Tuple[torch.Tensor, torch.Tensor]:
    #TODO implement with cumsum?
    n_steps = rewards.size(0)
    advantages = torch.zeros_like(rewards).to(rewards.device)
    lastgaelam = 0
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            nextnonterminal = 1.0 - dones_next
            nextvalues = values_next
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lambda_ * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns

