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
from agents.ppo_model_mineclip import PPOagent

def main(cfg):
    
    torch.cuda.set_device(cfg.agent.cuda_number)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_envs = cfg.agent.num_envs
    envs = gym.vector.AsyncVectorEnv([make_env(cfg.agent.task, cfg.agent.seed + i, idx) for idx, i in enumerate(range(num_envs))])
    agent = PPOagent(envs, cfg, device)

    num_steps = cfg.agent.num_steps
    batch_size = int(num_steps * num_envs)
    num_updates  = cfg.agent.total_timesteps // batch_size
    
    global_step = 0
    initial_update = 0

    obs, _ = envs.reset()
    obs, frame = agent.process_obs(obs)
    next_done = torch.zeros(num_envs)

    for update in range(initial_update, initial_update + num_updates):
        for step in tqdm(range(num_steps), desc=f'Update {update+1}/{initial_update + num_updates} ', unit='step', file=sys.stdout):
            global_step += 1 * num_envs

            action, logprob, _, val = agent.select_action(obs)
            next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
            agent.store_experience(obs, action, logprob, torch.tensor(reward), next_done, val.squeeze(), frame)

            obs, frame = agent.process_obs(next_obs)
            next_done = torch.Tensor(done).to(device)

            if "final_info" in info:
                for ind, agent_info in enumerate(info["final_info"]):
                    if agent_info is not None:
                        ep_rew = agent_info["episode"]["r"]
                        ep_len = agent_info["episode"]["l"]

                        logging.info(f"global step: {global_step}, agent_id={ind}, reward={ep_rew[-1]}, length={ep_len[-1]}")
                        writer.add_scalar("charts/episodic_return", ep_rew, global_step)
                        writer.add_scalar("charts/episodic_length", ep_len, global_step)

        agent.learn(last_obs=obs, last_done=next_done, writer=writer, global_step=global_step)

        if num_updates < 40:
            agent.save_model(update+1)
        elif (update + 1) % (num_updates // 40) == 0:
            agent.save_model(update+1)

    envs.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    dir_path = pathlib.Path(__file__).parent.resolve()
    with open(dir_path.joinpath(args.config), "r") as f:    # Change config file, conf_local.yaml
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)

    dname = f"{cfg.agent.task.replace(' ', '_')}_{datetime.now().strftime('%m_%d-%H:%M')}"
    if cfg.agent.clip_vloss:
        dname = dname + "_vclip"
    if cfg.agent.return_norm:
        dname = dname + "_rnorm"
    if cfg.agent.autocast_flag:
        dname = dname + "_autocast"
    if cfg.agent.multigpu:
        dname = dname + "_multigpu"

    cfg.agent.image_model = cfg.feature_net_kwargs.rgb_feat.image_model

    suf_add = f'only-ppo_{cfg.feature_net_kwargs.rgb_feat.image_model}'
    if cfg.agent.train_image_model: suf_add = f'train-imgppo_{cfg.feature_net_kwargs.rgb_feat.image_model}'
    
    if cfg.hyperparameters.wandb_init:
        wandb.init(
            project=f"{cfg.agent.server_name}_{suf_add}",         # Change project name 
            entity=None,
            sync_tensorboard=True,
            config=dict(cfg.agent),
            name=dname,
        )

    results_dir = f"results/{suf_add}/{dname}"
    cfg.agent.results_dir = results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    OmegaConf.save(cfg, results_dir + '/config.yaml')

    writer = SummaryWriter(results_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in cfg.agent.items()])),
    )
    sys.stderr = open(results_dir+'/err.e', 'w')

    log_file = f"{results_dir}/output.log"
    logging.basicConfig(
        filename=log_file,
        format="[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filemode='w'
    )

    main(cfg)
    writer.close()
