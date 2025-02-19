import os, sys
from datetime import datetime
from tqdm import tqdm
import argparse

import numpy as np
import torch
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
import pathlib, yaml, logging
from omegaconf import OmegaConf
import wandb

from envs.utils import make_env
from agents.ppo_multiagent import PPOagent

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import socket

def setup_ddp(rank, worlds_size, port):
    """Initialize DistributedDataParallel (DDP) for multi-GPU training."""
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{port}',
        rank=rank,
        world_size=worlds_size
    )

def cleanup_ddp():
    """Cleanup DistributedDataParallel (DDP) resources."""
    dist.destroy_process_group()

def ddp_train(rank, devices, world_size, cfg, results_dir, suf_add, dname, port):

    sys.stderr = open(results_dir+'/err.e', 'w')
    log_file = f"{results_dir}/output_{rank}.log"
    logging.basicConfig(
        filename=log_file,
        format="[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filemode='w'
    )
    logging.info(f"Rank {rank} started training.")
    """Train the agent using DDP."""
    setup_ddp(rank, world_size, port)
    # Set the CUDA device for the current process
    device = torch.device(f'cuda:{devices[rank]}')
    torch.cuda.set_device(device)

    num_envs = cfg.agent.num_envs
    if rank % 2 == 0:
        envs = gym.vector.SyncVectorEnv([make_env(cfg.agent.task1, cfg.agent.seed + i, idx) for idx, i in enumerate(range(num_envs))])
    else:
        envs = gym.vector.SyncVectorEnv([make_env(cfg.agent.task2, cfg.agent.seed + i, idx) for idx, i in enumerate(range(num_envs))])

    # Wrap the agent model with DDP
    agent = PPOagent(envs, cfg, device)
    agent.policy_model = DDP(agent.policy_model, device_ids=[devices[rank]])

    # Create SummaryWriter for all ranks
    if cfg.hyperparameters.wandb_init:
        wandb.init(
            project=f"{cfg.agent.server_name}_{suf_add}",         # Change project name 
            entity=None,
            sync_tensorboard=True,
            config=dict(cfg.agent),
            name=dname,
            group="multi-rank-experiment",
        )

    log_dir = f"{results_dir}/tensorboard/rank_{rank}"
    writer = SummaryWriter(log_dir)
    if rank == 0:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in cfg.agent.items()])),
        )

    num_steps = cfg.agent.num_steps
    batch_size = int(num_steps * num_envs)
    num_updates  = cfg.agent.total_timesteps // batch_size
    
    global_step = 0
    initial_update = 0

    obs, _ = envs.reset()
    obs, frame = agent.process_obs(obs)
    next_done = torch.zeros(num_envs)

    for update in range(initial_update, initial_update + num_updates):
        if rank == 0:
            pbar = tqdm(range(num_steps), desc=f'Update {update+1}/{initial_update + num_updates} ', unit='step', file=sys.stdout)
        else:
            pbar = range(num_steps)
        
        for step in pbar:
            global_step += 1 * num_envs

            action, logprob, _, val = agent.select_action(obs)
            next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
            agent.store_experience(obs, action, logprob, torch.tensor(reward), next_done, val.squeeze(), frame)

            obs, frame = agent.process_obs(next_obs)
            next_done = torch.Tensor(done).to(rank)

            if "final_info" in info:
                for ind, agent_info in enumerate(info["final_info"]):
                    if agent_info is not None:
                        ep_rew = agent_info["episode"]["r"]
                        ep_len = agent_info["episode"]["l"]

                        logging.info(f"rank: {rank} global step: {global_step}, agent_id={ind}, reward={ep_rew[-1]}, length={ep_len[-1]}")
                        
                        # Log to TensorBoard to all ranks
                        writer.add_scalar("charts/episodic_return", ep_rew, global_step)
                        writer.add_scalar("charts/episodic_length", ep_len, global_step)

        agent.learn(last_obs=obs, last_done=next_done, writer=writer, global_step=global_step)
        if rank == 0:
            if num_updates < 40:
                agent.save_model(update+1)
            elif (update + 1) % (num_updates // 40) == 0:
                agent.save_model(update+1)
            pbar.close()
    if rank == 0:
        writer.close()
    envs.close()
    # Cleanup DDP resources
    cleanup_ddp()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    dir_path = pathlib.Path(__file__).parent.resolve()
    with open(dir_path.joinpath(args.config), "r") as f:    # Change config file, conf_local.yaml
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)

    dname = f"{cfg.agent.task1.replace(' ', '_')}_{cfg.agent.task2.replace(' ', '_')}_{datetime.now().strftime('%m_%d-%H:%M')}"
    if cfg.agent.clip_vloss:
        dname = dname + "_vclip"
    if cfg.agent.return_norm:
        dname = dname + "_rnorm"
    if cfg.agent.autocast_flag:
        dname = dname + "_autocast"
    if cfg.agent.multigpu:
        dname = dname + "_multigpu"

    cfg.agent.image_model = cfg.feature_net_kwargs.rgb_feat.image_model

    suf_add = f'only-ppo_{cfg.feature_net_kwargs.rgb_feat.image_model}-multiagent'
    if cfg.agent.train_image_model: suf_add = f'train-imgppo_{cfg.feature_net_kwargs.rgb_feat.image_model}-multiagent'
    
    devices = cfg.agent.devices
    if not isinstance(devices, list):
        devices = list(range(torch.cuda.device_count()))
    print(f'Devices for training: {devices}')
    cfg.agent.devices = devices

    results_dir = f"results/{suf_add}/{dname}"
    cfg.agent.results_dir = results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    OmegaConf.save(cfg, results_dir + '/config.yaml')

    world_size = len(devices)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
        print(f"port: {port}")
    mp.spawn(ddp_train, args=(devices, world_size, cfg, results_dir, suf_add, dname, port), nprocs=world_size, join=True)