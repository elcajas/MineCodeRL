import numpy as np
import time
import os
import logging
from gymnasium import error

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mineclip import SimpleFeatureFusion, MineCLIP
from mineclip.mineagent.batch import Batch
from mineclip.mineagent.actor.distribution import MultiCategorical
from mineclip.utils import build_mlp

from groundingdino.models import GroundingDINO
import groundingdino.datasets.transforms as T
from PIL import Image

from .utils import set_MineCLIP, set_gDINO, layer_init
from .inference import predict
from agents import features_mlp as F

class PPOBuffer:
    def __init__(self, env, cfg, device) -> None:
        
        self.cfg = cfg
        self.env = env
        capacity = cfg.agent.num_steps
        num_envs = cfg.env.num_envs

        obss = {
            "rgb_feat": torch.zeros((capacity, num_envs, 512)).to(device),
            "compass": torch.zeros((capacity, num_envs, 4)).to(device),
            "gps": torch.zeros((capacity, num_envs, 3)).to(device),
            # "biome_id": torch.zeros((num_steps, num_envs, 1)),
        }
        self.obss = Batch(**obss)
        self.actions = torch.zeros((capacity, num_envs) + env.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((capacity, num_envs)).to(device)
        self.rewards = torch.zeros((capacity, num_envs)).to(device)
        self.dones = torch.zeros((capacity, num_envs)).to(device)
        self.values = torch.zeros((capacity, num_envs)).to(device)

        self.advantages = torch.zeros_like(self.rewards).to(device)
        self.returns = torch.zeros_like(self.rewards).to(device)

        self.ep_counter = torch.zeros((num_envs,)).to(device)
        self.frames = np.zeros((capacity, num_envs, 160, 256, 3))
        self.remain_frames = None
        self.pointer = 0
    
    def store(self, obs, action, logprob, reward, done, value, frame) -> None:
        self.obss[self.pointer] = obs
        self.actions[self.pointer] = action
        self.logprobs[self.pointer] = logprob
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.values[self.pointer] = value
        self.frames[self.pointer] = frame
        self.pointer += 1

    def calc_adv_and_return(self, last_value, last_done):
        gamma = self.cfg.agent.gamma
        lambbda = self.cfg.agent.gae_lambda
        lastgaelam = 0

        for t in reversed(range(self.pointer)):
            if t == self.pointer - 1:
                nextnonterminal = 1.0 - last_done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = lastgaelam = delta + gamma * lambbda * nextnonterminal * lastgaelam
        self.returns = self.advantages + self.values

        agent_idx = 0
        inds = self.dones[:,agent_idx].argwhere().squeeze(-1)
        for ep, ind in enumerate(inds):
            final_reward = self.rewards[ind - 3,agent_idx]
            ep_num = self.ep_counter[agent_idx] + ep + 1
            if final_reward >= 200:
                if ep > 0:
                    self.capture_video(inds[ep-1], inds[ep], ep_num.item(), final_reward.item())
                else:
                    self.capture_video(0, inds[ep], ep_num.item(), final_reward.item())

        self.remain_frames = self.frames[inds[-1].item():, agent_idx].squeeze()
        self.ep_counter += self.dones.sum(dim=0)
        self.pointer = 0

    def capture_video(self, first_frame_idx, last_frame_idx, ep, rew):
        frames = self.frames[first_frame_idx:last_frame_idx,0].squeeze()
        if first_frame_idx == 0 and self.remain_frames is not None:
            frames = np.concatenate((self.remain_frames, frames), axis=0)
            self.remain_frames = None
        if len(frames) > 0:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise error.DependencyNotInstalled("moviepy is not installed, run `pip install moviepy`") from e

            clip = ImageSequenceClip(list(frames), fps=10)

            dirpath = f"{self.cfg.results_dir}/videos/agent0"
            os.makedirs(dirpath, exist_ok=True)
            filepath = os.path.join(dirpath, f"ep{int(ep)}_rew{rew:.1f}_len{len(frames)}.mp4")
            clip.write_videofile(filepath, logger=None)

    def get_batch(self):
        b_obss = {}
        for key, value in self.obss.items():
            b_obss[key] = value.reshape(-1, value.shape[-1])

        b_obss = Batch(**b_obss)
        b_actions = self.actions.reshape((-1,) + self.env.single_action_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)

        return b_obss, b_actions, b_logprobs, b_advantages, b_returns, b_values
    
class Actor(nn.Module):
    def __init__(
            self,
            *,
            input_dim: int,
            action_dim: list[int],
            hidden_dim: int,
            hidden_depth: int,
            activation: str = "relu",
            deterministic_eval: bool = True,
            device,
    ) -> None:
        super().__init__()
        self.mlps = nn.ModuleList()
        for action in action_dim:
            # net = build_mlp(
            #     input_dim=input_dim,
            #     output_dim=action,
            #     hidden_dim=hidden_dim,
            #     hidden_depth=hidden_depth,
            #     activation=activation,
            #     norm_type=None,
            # )
            net = nn.Sequential(
                layer_init(nn.Linear(input_dim, hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, action), std=0.01),
            )
            self.mlps.append(net)
        
        self._action_dim = action_dim
        self._device = device
        self._deterministic_eval = deterministic_eval
    
    def forward(self, x) -> torch.Tensor:
        hidden = None
        return torch.cat([mlp(x) for mlp in self.mlps], dim=1), hidden

    @property
    def dist_fn(self):
        return lambda x: MultiCategorical(logits=x, action_dims=self._action_dim)

class Critic(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "relu",
        device
    ):
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=None
        )
        self._device = device
    
    def forward(self, x):
        hidden = None
        return self.net(x), hidden

class PolicyNetwork(nn.Module):
    def __init__(self, env, cfg, device) -> None:
        self.cfg = cfg
        self.device = device
        super().__init__()

        feature_net_kwargs = cfg.feature_net_kwargs
        feature_net = {}

        for k, v in feature_net_kwargs.items():
            v = dict(v)
            cls = v.pop("cls")
            cls = getattr(F, cls)
            feature_net[k] = cls(**v, device=self.device)

        feature_fusion_kwargs = cfg.feature_fusion
        self.network_model = SimpleFeatureFusion(feature_net, **feature_fusion_kwargs, device=self.device)
        self.actor = Actor(
            input_dim = self.network_model.output_dim,
            action_dim = list(env.single_action_space.nvec),
            device = self.device,
            **cfg.actor,             
        )
        self.critic = Critic(
            input_dim = self.network_model.output_dim,
            device = self.device,
            **cfg.critic,
        )

    def get_action_and_value(self, batch, action=None):
        hidden, _ = self.network_model(batch)
        logits, _ = self.actor(hidden)
        value, _ = self.critic(hidden)

        dist = self.actor.dist_fn(*logits) if isinstance(logits, tuple) else self.actor.dist_fn(logits)
        if action is None:
            action = dist.mode() if self.actor._deterministic_eval and not self.training else dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, logprob, entropy, value
    
    def get_value(self, batch):
        hidden, _ = self.network_model(batch)
        value, _ = self.critic(hidden)
        return value

class PPOagent:
    def __init__(self, env, cfg, device) -> None:

        num_steps = cfg.agent.num_steps
        batch_size = int(num_steps * cfg.env.num_envs)
        num_updates  = cfg.agent.total_timesteps // batch_size
        img_model = cfg.feature_net_kwargs.rgb_feat.image_model

        self.bf = PPOBuffer(env, cfg, device)
        if img_model == 'mineclip':
            self.image_model: MineCLIP = set_MineCLIP(cfg.mineclip).to(device)
        elif img_model == 'gdino':
            self.image_model: GroundingDINO = set_gDINO(cfg)
        else:
            raise ValueError("Invalid value for img_model. Supported options are 'mineclip' and 'gdino'.")
        
        self.policy_model = PolicyNetwork(env, cfg, device).to(device)
        self.optimizer = Adam(self.policy_model.parameters(), lr = cfg.agent.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_updates, eta_min=cfg.agent.min_lr)
        self.env = env
        self.cfg = cfg
        self.device = device
        self.start_time = time.time()
    
    def select_action(self, obs) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_model.get_action_and_value(obs)
    
    def store_experience(self, *args) -> None:
        self.bf.store(*args)

    def get_features(self, images: np.ndarray):
        # calculated from 21K video clips, which contains 2.8M frames
        MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
        MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)
        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25
        
        if isinstance(self.image_model, MineCLIP):
            img_tensor = torch.from_numpy(images).to(self.device)
            return self.image_model.forward_image_features(img_tensor)
        else:
            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize(MC_IMAGE_MEAN, MC_IMAGE_STD),
                ]
            )
            img_array = images.transpose((0,2,3,1))
            # img = Image.fromarray(img_array)
            img_transformed, _ = transform(img_array, None)
            
            TEXT_PROMPT = "spider . cow . sky . animal . tree ."
            logits = predict(
                model=self.image_model,
                image=img_transformed,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )

            return logits
    def process_obs(self, obs):
        raw_rgb = obs['rgb'].copy()
        with torch.no_grad():
            rgb_feat = self.get_features(raw_rgb)

        pitch = torch.deg2rad(torch.from_numpy(obs['pitch']))
        yaw = torch.deg2rad(torch.from_numpy(obs['yaw']))
        new_obs = {
            "rgb_feat": rgb_feat,
            "compass": torch.cat((torch.sin(pitch), torch.cos(pitch), torch.sin(yaw), torch.cos(yaw)), dim=1),
            "gps": torch.tensor(obs['pos']),
            # "biome_id": torch.tensor(obs['biome_id']).unsqueeze(dim=1),
        }
        return Batch(**new_obs), obs['rgb'].transpose((0,2,3,1))
    
    def minecip_reward(self):

        prompts = [
            "combat spider",
            "Milk a cow with empty bucket",
            "Combat zombie with sword",
            "Hunt a cow",
            "Hunt a sheep"
        ]

        rgb_feats = self.bf.obss.rgb_feat

        #Grouping into sets of 16
        seqs = []
        for i in range(len(rgb_feats) - 15):
            seq = torch.cat((rgb_feats[i:i + 16]), dim=0)
            seqs.append(seq)
        image_feats_batch = torch.stack((seqs), dim=0)

        with torch.no_grad():
            video_batch = self.image_model.forward_video_features(image_feats_batch)
            prompt_batch = self.image_model.encode_text(prompts)
            _, rew = self.image_model.forward_reward_head(video_batch, prompt_batch)
            rew = torch.nn.functional.softmax(rew, dim=0)
        
        rew_clamp = torch.clamp(rew[0] - 1/len(prompts), min=0).cpu()
        rewards = torch.cat([torch.full((15,),rew_clamp[0]), rew_clamp])
        rewards = rewards.cpu().numpy()

        self.shift_rewards()
        rewards = rewards + (self.bf.rewards >= 200)
        self.bf.rewards = rewards
    
    def shift_rewards(self):
        rews = torch.zeros_like(self.bf.rewards)
        rews[:-2] = self.bf.rewards[2:]
        rews[-2:] = torch.zeros(2, self.bf.rewards.size(1))
        self.bf.rewards = rews

    def learn(self, last_obs, last_done, writer: SummaryWriter, global_step):

        self.shift_rewards()
        with torch.no_grad():
            last_value = self.policy_model.get_value(last_obs).reshape(1, -1)
            self.bf.calc_adv_and_return(last_value, last_done)
        
        b_obss, b_actions, b_logprobs, b_advantages, b_returns, b_values =  self.bf.get_batch()

        batch_size = int(self.cfg.agent.num_steps * self.cfg.env.num_envs)

        mb_size = self.cfg.agent.num_minibatches 
        assert batch_size % mb_size == 0, f"Number of samples: {batch_size} is not divisible by num_minibatches: {mb_size}"
        minibatch_size = int(batch_size // mb_size)

        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(self.cfg.agent.learning_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.policy_model.get_action_and_value(b_obss[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.cfg.agent.clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg.agent.clip_coef, 1 + self.cfg.agent.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                mb_returns = b_returns[mb_inds]
                if self.cfg.agent.return_norm:
                    mb_returns = (mb_returns - mb_returns.mean()) / (mb_returns.std() + 1e-8)
                if self.cfg.agent.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -self.cfg.agent.clip_coef, self.cfg.agent.clip_coef)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - self.cfg.agent.ent_coef * entropy_loss +  self.cfg.agent.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.cfg.agent.max_grad_norm)
                self.optimizer.step()

            if approx_kl > self.cfg.agent.target_kl:
                break

        self.scheduler.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - self.start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - self.start_time)), global_step)

    def save_network(self, update):
        try:
            # Create the directory if it doesn't exist
            dirpath = f"{self.cfg.results_dir}/checkpoints"
            os.makedirs(dirpath, exist_ok=True)
            filepath = os.path.join(dirpath, f"update_{update}.pth")
            torch.save(self.policy_model.state_dict(), filepath)
            logging.info(f"Saving model weights for update {update} in {filepath}.")
        except Exception as e:
            print("Error occurred while saving model weights:", e)

    def load_network(self, path):
        try:
            self.policy_model.load_state_dict(torch.load(path))
            logging.info(f"Loading model weights from {path}.")
        except Exception as e:
            print("Error occurred while loading model weights:", e)


