import argparse
import datetime
import pprint
import os, sys

import random

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
from make_envs import make_custom_env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--task", type=str)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--algo-name", type=str, default="sac")
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--rew-norm", type=int, default=False)
    parser.add_argument("--action-unbounded", type=bool, default=True)
    parser.add_argument("--action-scaling", type=bool, default=True)
    parser.add_argument("--bound-action-method", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="MisleadingEntropy")
    parser.add_argument("--start-timesteps", type=int, default=25000)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=False, action="store_true") #False
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--step-per-collect", type=int, default=1)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
    )

    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser.parse_args()


def test_sac(args=get_args()):
    env, train_envs, test_envs = make_custom_env(
        args.task, args.seed, args.training_num, args.test_num, obs_norm=False
    )
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Env:", args.task)
    print("Algorithm", args.algo_name)
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # model
    net_a = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    )

    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        unbounded=args.action_unbounded,
        conditioned_sigma=True,
        device=args.device,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)


    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env.action_space,
        action_scaling=args.action_scaling,
        reward_normalization=args.rew_norm,
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.actor.load_state_dict(ckpt["actor"])
        policy.critic1.load_state_dict(ckpt["critic1"])
        policy.critic2.load_state_dict(ckpt["critic2"])

        actor_optim.load_state_dict(ckpt['actor_optim'])
        critic1_optim.load_state_dict(ckpt['critic1_optim'])
        critic2_optim.load_state_dict(ckpt['critic2_optim'])

        if "obs_rms" in ckpt:
            train_envs.set_obs_rms(ckpt["obs_rms"])
            test_envs.set_obs_rms(ckpt["obs_rms"])
        if "ret_rms" in ckpt:
            policy.ret_rms = ckpt["ret_rms"]

        print("Loaded agent from: ", args.resume_path)


    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    if not args.watch:
        train_collector.collect(n_step=args.start_timesteps, random=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(args.task, str(args.seed), args.algo_name, now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=args.save_interval,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):

        state = {
            "model": policy.state_dict(),
            "actor": policy.actor.state_dict(),
            "critic1": policy.critic1.state_dict(),
            "critic2": policy.critic2.state_dict(),
            "actor_optim": actor_optim.state_dict(),
            "critic1_optim": critic1_optim.state_dict(),
            "critic2_optim": critic2_optim.state_dict(),
            "alpha": policy._alpha
        }

        torch.save(state, os.path.join(log_path, "policy.pth"))


    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = os.path.join(log_path, f"checkpoint_env{env_step}_g{gradient_step}.pth")

        save_dict = \
            {
                "model": policy.state_dict(),
                "actor": policy.actor.state_dict(),
                "critic1": policy.critic1.state_dict(),
                "critic2": policy.critic2.state_dict(),
                "actor_optim": actor_optim.state_dict(),
                "critic1_optim": critic1_optim.state_dict(),
                "critic2_optim": critic2_optim.state_dict(),
                "alpha": policy._alpha
            }
        torch.save(save_dict, ckpt_path)

        return ckpt_path

    if not args.watch:
        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            args.epoch,
            args.step_per_epoch,
            args.step_per_collect,
            args.test_num,
            args.batch_size,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
        )
        pprint.pprint(result)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == "__main__":
    test_sac()
