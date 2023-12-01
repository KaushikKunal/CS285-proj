from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn
from torch.nn.utils import prune

import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # get the action from the critic using an epsilon-greedy strategy
        if np.random.random() <= epsilon:
            action = torch.randint(self.num_actions, (observation.shape[0],))
        else:
            action = torch.argmax(self.critic(observation))

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        with torch.no_grad():
            # compute target values
            next_qa_values = self.target_critic(next_obs)

            if self.use_double_q:
                next_action = torch.argmax(self.critic(next_obs), dim=-1)
            else:
                next_action = torch.argmax(next_qa_values, dim=-1)
            
            next_q_values = torch.gather(next_qa_values, dim=1, index=next_action.unsqueeze(1)).squeeze(1)
            target_values = reward + (1-1*done) * self.discount * next_q_values

        # train the critic with the target values
        qa_values = self.critic(obs)
        q_values = torch.gather(qa_values, dim=1, index=action.unsqueeze(1)).squeeze(1) # Compute from the data actions; see torch.gather
        loss = self.critic_loss(q_values, target_values)


        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # update the critic, and the target if needed
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)
        if step % self.target_update_period == 0:
            self.update_target_critic()

        return critic_stats
    
    def prune(self, amount=0.2):
        """Prunes the mean net"""
        parameters_to_prune = tuple((layer, 'weight') for layer in self.critic[0:-1:2])
        print(parameters_to_prune)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        parameters_to_prune = tuple((layer, 'weight') for layer in self.target_critic[0:-1:2])
        print(parameters_to_prune)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

    def prune_remove(self):
        tuple(prune.remove(layer, 'weight') for layer in self.critic[0:-1:2])
        tuple(prune.remove(layer, 'weight') for layer in self.target_critic[0:-1:2])

    def lra(self, amount=0.2):
        layers_to_approx = [layer for layer in self.critic[0:-1:2]] + [layer for layer in self.target_critic[0:-1:2]]
        for layer in layers_to_approx:
            print(layer.weight)
            weights = layer.weight
            rank = min(weights.shape[0], weights.shape[1])
            remaining_rank = int(((1-amount) * rank) // 1)
            U, S, V = torch.svd(weights)
            S[remaining_rank:] = torch.zeros((rank-remaining_rank))
            lra_weights = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.mT)
            print(torch.nn.Parameter(lra_weights))
            layer.weight = lra_weights # torch.nn.Parameter(lra_weights)
            print(f"{weights.shape=}\n {U.shape=}\n {S.shape=}\n {V.shape=}\n {remaining_rank=}\n {torch.dist(weights, lra_weights)=}")
    
        print([parameter for parameter in self.critic.parameters()])
