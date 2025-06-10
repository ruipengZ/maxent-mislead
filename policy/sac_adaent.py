from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union, List, Callable

import gymnasium
import numpy as np
import torch
from torch.distributions import Independent, Normal

from tianshou.data import Batch, ReplayBuffer, to_numpy
from tianshou.exploration import BaseNoise
from tianshou.policy import DDPGPolicy
import torch.nn.functional as F

from tianshou.data.utils.converter import to_torch_as
from tianshou.policy import BasePolicy
from tianshou.policy.base import _nstep_return


class SACPolicy(DDPGPolicy):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network. Default to 0.005.
    :param float gamma: discount factor, in [0, 1]. Default to 0.99.
    :param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
        regularization coefficient. Default to 0.2.
        If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
        alpha is automatically tuned.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param BaseNoise exploration_noise: add a noise to action for exploration.
        Default to None. This is useful when solving hard-exploration problem.
    :param bool deterministic_eval: whether to use deterministic action (mean
        of Gaussian policy) instead of stochastic action sampled by the policy.
        Default to True.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action) or empty string for no bounding.
        Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        critic1_s: torch.nn.Module,
        critic1_s_optim: torch.optim.Optimizer,
        critic2_s: torch.nn.Module,
        critic2_s_optim: torch.optim.Optimizer,
        entropy_n_sample: int = 100,
        entropy_dist_threshold: float = 0.9,
        baseline: bool = False,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        exploration_noise: Optional[BaseNoise] = None,
        deterministic_eval: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            None, None, None, None, tau, gamma, exploration_noise,
            reward_normalization, estimation_step, **kwargs
        )
        self.actor, self.actor_optim = actor, actor_optim
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic1_optim = critic1_optim
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        self.critic2_optim = critic2_optim

        self.critic1_s = critic1_s
        self.critic1_s_optim = critic1_s_optim
        self.critic2_s = critic2_s
        self.critic2_s_optim = critic2_s_optim

        self.critic1_s_old = deepcopy(critic1_s)
        self.critic1_s_old.eval()
        self.critic2_s_old = deepcopy(critic2_s)
        self.critic2_s_old.eval()

        self.entropy_n_sample = entropy_n_sample
        self.entropy_dist_threshold = entropy_dist_threshold

        self._is_auto_alpha = False
        self._alpha: Union[float, torch.Tensor]
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            assert alpha[1].shape == torch.Size([1]) and alpha[1].requires_grad
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        self._deterministic_eval = deterministic_eval
        self.__eps = np.finfo(np.float32).eps.item()

        self.baseline = baseline

    def train(self, mode: bool = True) -> "SACPolicy":
        self.training = mode
        self.actor.train(mode)
        self.critic1.train(mode)
        self.critic2.train(mode)
        self.critic1_s.train(mode)
        self.critic2_s.train(mode)
        return self

    def sync_weight(self) -> None:
        self.soft_update(self.critic1_old, self.critic1, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)
        self.soft_update(self.critic1_s_old, self.critic1_s, self.tau)
        self.soft_update(self.critic2_s_old, self.critic2_s, self.tau)

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        obs = batch[input]
        logits, hidden = self.actor(obs, state=state, info=batch.info)
        assert isinstance(logits, tuple)
        dist = Independent(Normal(*logits), 1)
        if self._deterministic_eval and not self.training:
            act = logits[0]
        else:
            act = dist.rsample()
        log_prob = dist.log_prob(act).unsqueeze(-1)
        # apply correction for Tanh squashing when computing logprob from Gaussian
        # You can check out the original SAC paper (arXiv 1801.01290): Eq 21.
        # in appendix C to get some understanding of this equation.
        squashed_action = torch.tanh(act)

        log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) +
                                        self.__eps).sum(-1, keepdim=True)

        return Batch(
            logits=logits,
            act=squashed_action,
            state=hidden,
            dist=dist,
            log_prob=log_prob
        )

    def map_action(self, act: Union[Batch, np.ndarray]) -> Union[Batch, np.ndarray]:
        """Map raw network output to action range in gym's env.action_space.

        This function is called in :meth:`~tianshou.data.Collector.collect` and only
        affects action sending to env. Remapped action will not be stored in buffer
        and thus can be viewed as a part of env (a black box action transformation).

        Action mapping includes 2 standard procedures: bounding and scaling. Bounding
        procedure expects original action range is (-inf, inf) and maps it to [-1, 1],
        while scaling procedure expects original action range is (-1, 1) and maps it
        to [action_space.low, action_space.high]. Bounding procedure is applied first.

        :param act: a data batch or numpy.ndarray which is the action taken by
            policy.forward.

        :return: action in the same form of input "act" but remap to the target action
            space.
        """
        if isinstance(self.action_space, gymnasium.spaces.Box) and \
                isinstance(act, np.ndarray):
            # currently this action mapping only supports np.ndarray action
            if self.action_bound_method == "clip":
                act = np.clip(act, -1.0, 1.0)
            elif self.action_bound_method == "tanh":
                act = np.tanh(act)
            if self.action_scaling:
                assert np.min(act) >= -1.0 and np.max(act) <= 1.0, \
                    "action scaling only accepts raw action range = [-1, 1]"
                low, high = self.action_space.low, self.action_space.high
                act = low + (high - low) * (act + 1.0) / 2.0  # type: ignore
        return act

    def map_action_inverse(
        self, act: Union[Batch, List, np.ndarray]
    ) -> Union[Batch, List, np.ndarray]:
        """Inverse operation to :meth:`~tianshou.policy.BasePolicy.map_action`.

        This function is called in :meth:`~tianshou.data.Collector.collect` for
        random initial steps. It scales [action_space.low, action_space.high] to
        the value ranges of policy.forward.

        :param act: a data batch, list or numpy.ndarray which is the action taken
            by gym.spaces.Box.sample().

        :return: action remapped.
        """
        if isinstance(self.action_space, gymnasium.spaces.Box):
            act = to_numpy(act)
            if isinstance(act, np.ndarray):
                if self.action_scaling:
                    low, high = self.action_space.low, self.action_space.high
                    scale = high - low
                    eps = np.finfo(np.float32).eps.item()
                    scale[scale < eps] += eps
                    act = (act - low) * 2.0 / scale - 1.0
                if self.action_bound_method == "tanh":
                    act = (np.log(1.0 + act) - np.log(1.0 - act)) / 2.0  # type: ignore
        return act


    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs: s_{t+n}
        obs_next_result = self(batch, input="obs_next")
        act_ = obs_next_result.act
        target_q = torch.min(
            self.critic1_old(batch.obs_next, act_),
            self.critic2_old(batch.obs_next, act_),
        ) - self._alpha * obs_next_result.log_prob

        target_q_s = torch.min(
            self.critic1_s_old(batch.obs_next, act_),
            self.critic2_s_old(batch.obs_next, act_),
        )
        return target_q, target_q_s

    @staticmethod
    def compute_nstep_return(
            batch: Batch,
            buffer: ReplayBuffer,
            indice: np.ndarray,
            target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
            gamma: float = 0.99,
            n_step: int = 1,
            rew_norm: bool = False,
    ) -> Batch:
        r"""Compute n-step return for Q-learning targets.

        .. math::
            G_t = \sum_{i = t}^{t + n - 1} \gamma^{i - t}(1 - d_i)r_i +
            \gamma^n (1 - d_{t + n}) Q_{\mathrm{target}}(s_{t + n})

        where :math:`\gamma` is the discount factor, :math:`\gamma \in [0, 1]`,
        :math:`d_t` is the done flag of step :math:`t`.

        :param Batch batch: a data batch, which is equal to buffer[indice].
        :param ReplayBuffer buffer: the data buffer.
        :param function target_q_fn: a function which compute target Q value
            of "obs_next" given data buffer and wanted indices.
        :param float gamma: the discount factor, should be in [0, 1]. Default to 0.99.
        :param int n_step: the number of estimation step, should be an int greater
            than 0. Default to 1.
        :param bool rew_norm: normalize the reward to Normal(0, 1), Default to False.

        :return: a Batch. The result will be stored in batch.returns as a
            torch.Tensor with the same shape as target_q_fn's return tensor.
        """
        assert not rew_norm, \
            "Reward normalization in computing n-step returns is unsupported now."
        rew = buffer.rew
        bsz = len(indice)
        indices = [indice]
        for _ in range(n_step - 1):
            indices.append(buffer.next(indices[-1]))
        indices = np.stack(indices)
        # terminal indicates buffer indexes nstep after 'indice',
        # and are truncated at the end of each episode
        terminal = indices[-1]
        with torch.no_grad():
            target_q_torch, target_q_wo_ent_torch = target_q_fn(buffer, terminal)  # (bsz, ?)
        target_q = to_numpy(target_q_torch.reshape(bsz, -1))
        target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step)

        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)

        target_q_wo_ent = to_numpy(target_q_wo_ent_torch.reshape(bsz, -1))
        target_q_wo_ent = target_q_wo_ent * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        target_q_wo_ent = _nstep_return(rew, end_flag, target_q_wo_ent, indices, gamma, n_step)
        batch.returns_wo_ent = to_torch_as(target_q_wo_ent, target_q_wo_ent_torch)
        if hasattr(batch, "weight_s"):
            batch.weight_s = to_torch_as(batch.weight_s, target_q_wo_ent_torch)


        return batch

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        batch = self.compute_nstep_return(
            batch, buffer, indices, self._target_q, self._gamma, self._n_step,
            self._rew_norm
        )
        return batch

    @staticmethod
    def _mse_optimizer(
            batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer,
            critic_s: torch.nn.Module, optimizer_s: torch.optim.Optimizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        critic_loss = (td.pow(2) * weight).mean()
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()

        weight_s = getattr(batch, "weight_s", 1.0)
        current_q_s = critic_s(batch.obs, batch.act).flatten()
        target_q_s = batch.returns_wo_ent.flatten()
        td_s = current_q_s - target_q_s
        critic_s_loss = (td_s.pow(2) * weight_s).mean()
        optimizer_s.zero_grad()
        critic_s_loss.backward()
        optimizer_s.step()
        return td, critic_loss, td_s, critic_s_loss

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        # critic 1&2
        td1, critic1_loss, td1_s, critic1_s_loss = self._mse_optimizer(
            batch, self.critic1, self.critic1_optim, self.critic1_s, self.critic1_s_optim
        )
        td2, critic2_loss, td2_s, critic2_s_loss = self._mse_optimizer(
            batch, self.critic2, self.critic2_optim, self.critic2_s, self.critic2_s_optim
        )
        batch.weight = (td1 + td2) / 2.0  # prio-buffer
        batch.weight_s = (td1_s + td2_s) / 2.0  # prio-buffer

        # actor
        obs_result = self(batch)
        act = obs_result.act

        current_q1a = self.critic1(batch.obs, act).flatten()
        current_q2a = self.critic2(batch.obs, act).flatten()
        current_q = torch.min(current_q1a, current_q2a)

        current_q1a_s = self.critic1_s(batch.obs, act).flatten()
        current_q2a_s = self.critic2_s(batch.obs, act).flatten()
        current_q_s = torch.min(current_q1a_s, current_q2a_s)

        # sample multiple actions wrt batch.obs and calculate the q withou entropy using critic1_s and critic2_s
        # create obs_sample by repeating batch.obs(np.array) for self.entropy_n_sample times
        obs_sample = np.repeat(batch.obs, self.entropy_n_sample, axis=0)
        act_sample = self(Batch(obs=obs_sample, info={})).act
        current_q1a_sample = self.critic1(obs_sample, act_sample).flatten()
        current_q2a_sample = self.critic2(obs_sample, act_sample).flatten()
        current_q_sample = torch.min(current_q1a_sample, current_q2a_sample).reshape(-1, self.entropy_n_sample)
        current_q_sample = current_q_sample - current_q_sample.min(dim=1, keepdim=True)[0]

        current_q1a_s_sample = self.critic1_s(obs_sample, act_sample).flatten()
        current_q2a_s_sample = self.critic2_s(obs_sample, act_sample).flatten()
        current_q_s_sample = torch.min(current_q1a_s_sample, current_q2a_s_sample).reshape(-1, self.entropy_n_sample)
        current_q_s_sample = current_q_s_sample - current_q_s_sample.min(dim=1, keepdim=True)[0]

        cosine_similarity = F.cosine_similarity(current_q_sample, current_q_s_sample)
        condition = cosine_similarity < self.entropy_dist_threshold

        target_q = torch.where(condition, current_q_s, current_q)
        if self.baseline:
            target_q = current_q


        surrogate_rate = condition.float().mean()

        actor_loss = (
            self._alpha * obs_result.log_prob.flatten() -
            target_q
        ).mean()



        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_prob = obs_result.log_prob.detach() + self._target_entropy
            # please take a look at issue #258 if you'd like to change this line
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self.sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "loss/critic1_s": critic1_s_loss.item(),
            "loss/critic2_s": critic2_s_loss.item(),
            "loss/surrogate_rate": surrogate_rate.item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore

        return result
