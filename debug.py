import numpy as np
from datetime import datetime
import torch
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
import pathlib, yaml, logging
import wandb
from omegaconf import OmegaConf

from mineclip.mineagent.batch import Batch
from envs.utils import make_env
from agents import PPOagent

def main(cfg):
    dname = f"{cfg.env.task.replace(' ', '_')}_{datetime.now().strftime('%d_%m-%H:%M')}"
    if cfg.agent.clip_vloss:
        dname = dname + "_vclip"
    if cfg.agent.return_norm:
        dname = dname + "_rnorm"

    results_dir = f"debug_results/{dname}"

    cfg.results_dir = results_dir
    cfg.agent.n_envs = cfg.env.num_envs
    cfg.agent.tsk = cfg.env.task

    writer = SummaryWriter(results_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in cfg.agent.items()])),
    )

    log_file = f"{cfg.results_dir}/output.log"
    logging.basicConfig(
        filename=log_file,
        format="[%(asctime)s] [%(levelname)8s] --- %(message)s (%(filename)s:%(lineno)s)", datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filemode='w'
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_envs = cfg.env.num_envs
    envs = gym.vector.AsyncVectorEnv([make_env(cfg.env.task, cfg.agent.seed + i, idx, results_dir) for idx, i in enumerate(range(num_envs))])
    agent = PPOagent(envs, cfg, device)
    # agent.load_network('/home/kenjic/documents/update_mineppo/results/milk_a_cow_21_04-14:31/checkpoints/update_20.pth')

    num_steps = cfg.agent.num_steps
    batch_size = int(num_steps * num_envs)
    num_updates  = cfg.agent.total_timesteps // batch_size
    
    global_step = 0
    initial_update = 0

    obs, _ = envs.reset()
    obs, frame = agent.process_obs(obs)
    next_done = torch.zeros(num_envs)

    for update in range(initial_update, initial_update + num_updates):
        for step in range(num_steps):
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
                        # print(f"global step: {global_step}, agent_id={ind}, reward={ep_rew[-1]}, length={ep_len[-1]}")
                        logging.info(f"global step: {global_step}, agent_id={ind}, reward={ep_rew[-1]}, length={ep_len[-1]}")
                        writer.add_scalar("charts/episodic_return", ep_rew, global_step)
                        writer.add_scalar("charts/episodic_length", ep_len, global_step)

        agent.learn(last_obs=obs, last_done=next_done, writer=writer, global_step=global_step)

    envs.close()
    writer.close()

if __name__ == "__main__":

    dir_path = pathlib.Path(__file__).parent.resolve()
    with open(dir_path.joinpath("deb_config.yaml"), "r") as f:    # Change config file, conf_local.yaml
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)

    main(cfg)
