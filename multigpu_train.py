import os, sys
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
import pathlib, yaml, logging
from omegaconf import OmegaConf
import wandb

from envs.utils import make_env
from agents.ppo_model_multigpu import PPOagent

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_ddp(rank, worlds_size):
    """Initialize DistributedDataParallel (DDP) for multi-GPU training."""
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:23456',
        rank=rank,
        world_size=worlds_size
    )

def cleanup_ddp():
    """Cleanup DistributedDataParallel (DDP) resources."""
    dist.destroy_process_group()

def ddp_train(rank, devices, world_size, cfg, results_dir):
    sys.stderr = open(results_dir+'/err.e', 'w')
    log_file = f"{cfg.results_dir}/output_{rank}.log"
    logging.basicConfig(
        filename=log_file,
        format="[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filemode='w'
    )
    logging.info(f"Rank {rank} started training.")
    """Train the agent using DDP."""
    setup_ddp(rank, world_size)
    # Set the CUDA device for the current process
    device = torch.device(f'cuda:{devices[rank]}')
    torch.cuda.set_device(device)

    num_envs = cfg.env.num_envs
    envs = gym.vector.AsyncVectorEnv([make_env(cfg.env.task, cfg.agent.seed + i, idx, results_dir) for idx, i in enumerate(range(num_envs))])
    agent = PPOagent(envs, cfg, device)

    initial_update = 0
    if cfg.agent.load_ppo_model:
        initial_update = agent.load_model(cfg.agent.ppo_checkpoint_path, cfg.agent.image_checkpoint_path)
    
    # Wrap the agent model with DDP
    agent.policy_model = DDP(agent.policy_model, device_ids=[devices[rank]])

    # Create SummaryWriter only for rank 0
    writer = None
    if rank == 0:
        writer = SummaryWriter(results_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in cfg.agent.items()])),
        )

    num_steps = cfg.agent.num_steps
    batch_size = int(num_steps * num_envs)
    num_updates  = cfg.agent.total_timesteps // batch_size
    
    global_step = 0

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
                        # Log to TensorBoard only rank 0
                        if rank == 0:
                            writer.add_scalar("charts/episodic_return", ep_rew, global_step)
                            writer.add_scalar("charts/episodic_length", ep_len, global_step)

        agent.learn(last_obs=obs, last_done=next_done, writer=writer, global_step=global_step, rank=rank)
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

    dir_path = pathlib.Path(__file__).parent.resolve()
    with open(dir_path.joinpath("multigpu_config.yaml"), "r") as f:    # Change config file, conf_local.yaml
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)

    dname = f"{cfg.env.task.replace(' ', '_')}_{datetime.now().strftime('%m_%d-%H:%M')}"
    if cfg.agent.clip_vloss:
        dname = dname + "_vclip"
    if cfg.agent.return_norm:
        dname = dname + "_rnorm"
    if cfg.agent.autocast_flag:
        dname = dname + "_autocast"
    if cfg.agent.multigpu:
        dname = dname + "_multigpu"

    cfg.agent.n_envs = cfg.env.num_envs
    cfg.agent.tsk = cfg.env.task
    cfg.agent.image_model = cfg.feature_net_kwargs.rgb_feat.image_model

    suf_add = f'only-ppo_{cfg.feature_net_kwargs.rgb_feat.image_model}'
    if cfg.agent.train_image_model: suf_add = f'ppo-imgenc_{cfg.feature_net_kwargs.rgb_feat.image_model}'

    # wandb.init(
    #     project=f"{cfg.agent.server_name}_{suf_add}",         # Change project name 
    #     entity=None,
    #     sync_tensorboard=True,
    #     config=dict(cfg.agent),
    #     name=dname,
    # )

    results_dir = f"debug_results/{suf_add}/{dname}"
    cfg.results_dir = results_dir
    if not os.path.exists(cfg.results_dir):
        os.makedirs(cfg.results_dir)
    
    devices = cfg.agent.devices
    world_size = len(devices)
    mp.spawn(ddp_train, args=(devices, world_size, cfg, results_dir), nprocs=world_size, join=True)
