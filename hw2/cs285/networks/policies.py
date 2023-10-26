import itertools
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import prune
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # implement get_action
        action = None
        if self.discrete:
            action = self(obs).sample()
        else:
            action = self(obs).rsample()
        return ptu.to_numpy(action)

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        obs = ptu.from_numpy(obs)
        dist = None
        if self.discrete:
            # define the forward pass for a policy with a discrete action space.
            dist = distributions.Categorical(logits=self.logits_net(obs))
        else:
            # define the forward pass for a policy with a continuous action space.
            dist = distributions.Normal(self.mean_net(obs), torch.exp(self.logstd))
        return dist

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # implement the policy gradient actor update.
        log_probs = None
        if self.discrete:
            log_probs = self(obs).log_prob(actions)
        else:
            log_probs = torch.sum(self(obs).log_prob(actions), dim=1)
        loss = -torch.mean(log_probs*advantages)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
    
    def prune(self, amount=0.2):
        """Prunes the mean net"""
        parameters_to_prune = tuple((layer, 'weight') for layer in self.mean_net[0:-1:2])
        print(parameters_to_prune)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

    def prune_remove(self):
        parameters_to_prune = tuple(prune.remove(layer, 'weight') for layer in self.mean_net[0:-1:2])