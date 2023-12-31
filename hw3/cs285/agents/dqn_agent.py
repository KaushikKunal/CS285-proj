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
        prune_amount: float = 0,
        lra_amount: float = 0,
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
        self.prune_amount = prune_amount
        self.lra_amount = lra_amount

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
        self.prune(self.prune_amount, False, True)
        self.prune_remove(False, True)
        self.lra(self.lra_amount, self.target_critic)

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
    
    def prune(self, amount=0.2, critic=True, target=False):
        """Prunes the critic nets"""
        if critic:
            parameters_to_prune = tuple((layer, 'weight') for layer in self.critic[0:-1:2])
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
        if target:
            parameters_to_prune = tuple((layer, 'weight') for layer in self.target_critic[0:-1:2])
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )

    def prune_remove(self, critic=True, target=False):
        """gets rid of the metadata that pruning adds to nets"""
        if critic:
            tuple(prune.remove(layer, 'weight') for layer in self.critic[0:-1:2])
        if target:
            tuple(prune.remove(layer, 'weight') for layer in self.target_critic[0:-1:2])

    def lra(self, amount=0.2, network=None):
        """performs low rank approximation on the given network"""
        if network is None:
            network = self.critic
        for parameter in network.parameters():
            weights = parameter.data
            if len(weights.shape) == 1:
                # this parameter corresponds to bias
                continue
            rank = min(weights.shape[0], weights.shape[1])
            # if rank > 16: # this would stop us from quantizing already small networks - worth keeping?
            remaining_rank = int(np.ceil((1-amount)*rank))
            U, S, V = torch.svd(weights)
            S[remaining_rank:] = torch.zeros((rank-remaining_rank))
            lra_weights = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.mT)
            parameter.data = lra_weights
