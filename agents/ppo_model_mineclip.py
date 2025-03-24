import numpy as np
import time
import os
import logging
from gymnasium import error

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from mineclip import MineCLIP
from mineclip import SimpleFeatureFusion
from mineclip.mineagent.batch import Batch
from mineclip.mineagent.actor.distribution import MultiCategorical
from mineclip.utils import build_mlp

from transformers import AutoProcessor

from .utils import set_MineCLIP, set_gDINO, layer_init, set_hf_gDINO
from .inference import predict
from agents import features_mlp as F
from .encoders import ImageEncoder

class PPOBuffer:
    def __init__(self, env, cfg, device) -> None:
        
        self.cfg = cfg
        self.env = env
        capacity = cfg.agent.num_steps
        num_envs = cfg.agent.num_envs
        
        # set feature dimension depending on image model (mineclip: 512, gdino: (256, 900))
        # If train grounding dino, rgb pixel dimension is used (3,160,256)

        feat_dim = [512]
        if cfg.feature_net_kwargs.rgb_feat.image_model == 'gdino': feat_dim = [256, 900]
        if cfg.agent.train_image_model: feat_dim = [3, 160, 256]
        
        obss = {
            "rgb_feat": torch.zeros((capacity, num_envs, *(feat_dim))),      
            "compass": torch.zeros((capacity, num_envs, 4)),
            "gps": torch.zeros((capacity, num_envs, 3)),
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
        self.img_rew = torch.zeros((capacity, num_envs, 512))
        self.tmp = None
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
        self.img_rew[self.pointer] = self.tmp
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

        if len(inds) > 0:
            self.remain_frames = self.frames[inds[-1].item():, agent_idx]
        elif self.remain_frames is None:
            self.remain_frames = self.frames[:, agent_idx]
        else:
            self.remain_frames = np.concatenate((self.remain_frames, self.frames[:, agent_idx]), axis=0)
        self.ep_counter += self.dones.sum(dim=0)
        self.pointer = 0

    def capture_video(self, first_frame_idx, last_frame_idx, ep, rew):
        # To do separate by ranks
        frames = self.frames[first_frame_idx:last_frame_idx,0]
        if first_frame_idx == 0 and self.remain_frames is not None:
            frames = np.concatenate((self.remain_frames, frames), axis=0)
            self.remain_frames = None
        if len(frames) > 0:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise error.DependencyNotInstalled("moviepy is not installed, run `pip install moviepy`") from e

            clip = ImageSequenceClip(list(frames), fps=10)

            dirpath = f"{self.cfg.agent.results_dir}/videos/agent0"
            os.makedirs(dirpath, exist_ok=True)
            filepath = os.path.join(dirpath, f"ep{int(ep)}_rew{rew:.1f}_len{len(frames)}.mp4")
            clip.write_videofile(filepath, logger=None)

    def get_batch(self):
        b_obss = {}
        for key, value in self.obss.items():
            b_obss[key] = value.reshape((-1,) + value.shape[2:])

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
            net = build_mlp(
                input_dim=input_dim,
                output_dim=action,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
                activation=activation,
                norm_type=None,
            )
            # net = nn.Sequential(
            #     layer_init(nn.Linear(input_dim, hidden_dim)),
            #     nn.ReLU(),
            #     layer_init(nn.Linear(hidden_dim, hidden_dim)),
            #     nn.ReLU(),
            #     layer_init(nn.Linear(hidden_dim, hidden_dim)),
            #     nn.ReLU(),
            #     layer_init(nn.Linear(hidden_dim, hidden_dim)),
            #     nn.ReLU(),
            #     layer_init(nn.Linear(hidden_dim, action), std=0.01),
            # )
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

        img_model = cfg.feature_net_kwargs.rgb_feat.image_model
        if img_model == 'mineclip':
            self.image_model: MineCLIP = set_MineCLIP(cfg.mineclip)

        elif img_model == 'gdino':
            self.image_model = set_gDINO(cfg, device)

        elif img_model == 'hf_gdino':
            self.image_model = set_hf_gDINO(cfg, device)
            self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")

        elif img_model == 'image_encoder':
            self.image_model = ImageEncoder()

        else:
            raise ValueError("Invalid value for img_model. Supported options are 'mineclip' and 'gdino'.")

    def get_features(self, images: torch.Tensor):
        # calculated from 21K video clips, which contains 2.8M frames
        MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
        MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)
        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25
        
        if isinstance(self.image_model, MineCLIP):
            return self.image_model.forward_image_features(images.to(self.device))
        
        if isinstance(self.image_model, ImageEncoder):
            return self.image_model(images.to(self.device))
        
        if self.cfg.feature_net_kwargs.rgb_feat.image_model == "gdino":
            TEXT_PROMPT = "spider . cow . sky . animal . tree ."

            logits = predict(
                model=self.image_model,
                images=images.cpu().numpy(),
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD,
                device=self.device,
            )
            return logits
        
        else:
            TEXT_PROMPT = "spider . cow . sky . animal . tree ."
            inputs = self.processor(images=images, text=[TEXT_PROMPT]*len(images), return_tensors="pt").to(self.device)
            # with torch.no_grad():
            outputs = self.image_model(**inputs, output_hidden_states=True)
            logits = outputs.decoder_hidden_states[1].transpose(-1,-2)
            return logits

    
    def get_action_and_value(self, batch, action=None):
        img_feat = batch.rgb_feat
        if self.cfg.agent.train_image_model: img_feat = self.get_features(batch.rgb_feat)
        
        hidden, _ = self.network_model(Batch(rgb_feat=img_feat, compass=batch.compass, gps=batch.gps))
        logits, _ = self.actor(hidden)
        value, _ = self.critic(hidden)

        dist = self.actor.dist_fn(*logits) if isinstance(logits, tuple) else self.actor.dist_fn(logits)
        if action is None:
            action = dist.mode() if self.actor._deterministic_eval and not self.training else dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, logprob, entropy, value
    
    def get_value(self, batch):
        img_feat = batch.rgb_feat
        if self.cfg.agent.train_image_model: img_feat = self.get_features(batch.rgb_feat)

        hidden, _ = self.network_model(Batch(rgb_feat=img_feat, compass=batch.compass, gps=batch.gps))
        value, _ = self.critic(hidden)
        return value
    
class PPOagent:
    def __init__(self, env, cfg, device) -> None:

        num_steps = cfg.agent.num_steps
        batch_size = int(num_steps * cfg.agent.num_envs)
        num_minibatch = cfg.agent.num_minibatches
        assert batch_size % num_minibatch == 0, f"Number of samples: {batch_size} is not divisible by num_minibatches: {num_minibatch}"
        minibatch_size = int(batch_size // num_minibatch)
        num_updates  = cfg.agent.total_timesteps // batch_size

        self.batch_size = batch_size
        self.minibatch_size = minibatch_size

        self.epochs = cfg.agent.learning_epochs
        self.clip_coef = cfg.agent.clip_coef
        self.vf_coef = cfg.agent.vf_coef
        self.ent_coef = cfg.agent.ent_coef
        self.max_grad_norm = cfg.agent.max_grad_norm
        self.target_kl = cfg.agent.target_kl

        self.results_dir = cfg.agent.results_dir
        self.normalized_return = cfg.agent.return_norm
        self.clip_vloss = cfg.agent.clip_vloss
        self.train_vision = cfg.agent.train_image_model

        self.bf = PPOBuffer(env, cfg, device)
        self.policy_model = PolicyNetwork(env, cfg, device).to(device)
        self.reward_model: MineCLIP = set_MineCLIP(cfg.mineclip).to(device)

        # self.optimizer = Adam(self.policy_model.parameters(), lr = cfg.agent.learning_rate)
        self.optimizer = Adam([
            {'params': self.policy_model.network_model.parameters(), 'lr': cfg.agent.policy_learning_rate},
            {'params': self.policy_model.actor.parameters(), 'lr': cfg.agent.policy_learning_rate},
            {'params': self.policy_model.critic.parameters(), 'lr': cfg.agent.policy_learning_rate},
            {'params': self.policy_model.image_model.parameters(), 'lr': cfg.agent.vision_learning_rate},
        ])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_updates, eta_min=cfg.agent.min_lr)
        self.env = env
        self.cfg = cfg
        self.device = device
        self.start_time = time.time()

        if cfg.agent.load_ppo_model:
            self.load_model(cfg.agent.ppo_checkpoint_path, cfg.agent.load_image_model, cfg.agent.image_checkpoint_path)
    
    def select_action(self, obs) -> torch.Tensor:
        with torch.no_grad():
            return self.policy_model.get_action_and_value(obs)
    
    def store_experience(self, *args) -> None:
        self.bf.store(*args)
        
    def process_obs(self, obs):
        if not self.train_vision:
            raw_rgb = obs['rgb'].copy()
            with torch.no_grad():
                rgb_feat = self.policy_model.get_features(torch.tensor(raw_rgb))            # later check if with torch.no_grad() is required
                img_rew = self.reward_model.forward_image_features(torch.tensor(raw_rgb).to(self.device))

            pitch = torch.deg2rad(torch.from_numpy(obs['pitch']))
            yaw = torch.deg2rad(torch.from_numpy(obs['yaw']))
            new_obs = {
                "rgb_feat": rgb_feat,
                "compass": torch.cat((torch.sin(pitch), torch.cos(pitch), torch.sin(yaw), torch.cos(yaw)), dim=1),
                "gps": torch.tensor(obs['pos']),
                # "biome_id": torch.tensor(obs['biome_id']).unsqueeze(dim=1),
            }
            self.bf.tmp = img_rew
            return Batch(**new_obs), obs['rgb'].transpose((0,2,3,1))
        
        else:
            pitch = torch.deg2rad(torch.from_numpy(obs['pitch']))
            yaw = torch.deg2rad(torch.from_numpy(obs['yaw']))
            new_obs = {
                "rgb_feat": torch.tensor(obs['rgb']),
                "compass": torch.cat((torch.sin(pitch), torch.cos(pitch), torch.sin(yaw), torch.cos(yaw)), dim=1),
                "gps": torch.tensor(obs['pos']),
            }
            return Batch(**new_obs), obs['rgb'].transpose((0,2,3,1))
    
    def minecip_reward(self):

        # prompts = [
        #     "combat spider with a sword",
        #     "Milk a cow with empty bucket",
        #     "Combat zombie",
        #     "Hunt a cow",
        #     "Hunt a sheep"
        # ]

        # rgb_feats = self.bf.img_rew

        # #Grouping into sets of 16
        # seqs = []
        # for i in range(len(rgb_feats) - 15):
        #     # seq = torch.cat((rgb_feats[i:i + 16]), dim=0)
        #     seqs.append(rgb_feats[i:i + 16])
        # image_feats_batch = torch.stack((seqs), dim=0).to(self.device)
        # B, L, N, D = image_feats_batch.shape
        # with torch.no_grad():
             
        #     video_batch = self.reward_model.temporal_encoder(image_feats_batch.reshape(B*N, L, D))
        #     prompt_batch = self.reward_model.encode_text(prompts)
        #     _, rew = self.reward_model.forward_reward_head(video_batch, prompt_batch)
        #     rew = torch.nn.functional.softmax(rew, dim=0)
        
        # rew_clamp = torch.clamp(rew[0] - 1/len(prompts), min=0)
        # rewards = torch.cat([torch.full((15,),rew_clamp[0]).to(self.device), rew_clamp])
        # rewards = rewards.reshape(B+15,N)

        # self.shift_rewards()
        # rewards = rewards + (self.bf.rewards >= 200)
        # self.bf.rewards = rewards
        prompts = [
            "combat spider with a sword",
            "Milk a cow with empty bucket",
            "Combat zombie",
            "Hunt a cow",
            "Hunt a sheep"
        ]

        SEQ_LEN = 16  # Define as a constant for clarity

        rgb_feats = self.bf.img_rew

        # Efficiently extract sequences of length SEQ_LEN
        image_feats_batch = torch.stack([rgb_feats[i:i + SEQ_LEN] for i in range(len(rgb_feats) - (SEQ_LEN - 1))], dim=0).to(self.device)
        
        B, L, N, D = image_feats_batch.shape

        with torch.no_grad():
            video_batch = self.reward_model.temporal_encoder(image_feats_batch.reshape(B * N, L, D))
            prompt_batch = self.reward_model.encode_text(prompts)
            _, rew = self.reward_model.forward_reward_head(video_batch, prompt_batch)
            rew = torch.nn.functional.softmax(rew, dim=0)

        rew_clamp = torch.clamp(rew[0] - 1 / len(prompts), min=0).reshape(-1, N)
        first_rew = torch.tile(rew_clamp[0], (SEQ_LEN - 1, 1))
        rewards = torch.cat([first_rew, rew_clamp])
        
        self.shift_rewards()
        self.bf.rewards = rewards + (self.bf.rewards >= 200)
    
    def shift_rewards(self):
        rews = torch.zeros_like(self.bf.rewards)
        rews[:-2] = self.bf.rewards[2:]
        rews[-2:] = torch.zeros(2, self.bf.rewards.size(1))
        self.bf.rewards = rews

    def learn(self, last_obs, last_done, writer: SummaryWriter, global_step):

        self.minecip_reward()
        with torch.no_grad():
            last_value = self.policy_model.get_value(last_obs).reshape(1, -1)
            self.bf.calc_adv_and_return(last_value, last_done)
        
        b_obss, b_actions, b_logprobs, b_advantages, b_returns, b_values =  self.bf.get_batch()

        b_inds = np.arange(self.batch_size)
        clipfracs = []

        for epoch in range(self.epochs):
            np.random.shuffle(b_inds)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                if end == self.batch_size:
                    logging.info(f"Update [{epoch+1}/{self.epochs}] for minibatch: [{end}/{self.batch_size}]")

                _, newlogprob, entropy, newvalue = self.policy_model.get_action_and_value(b_obss[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                mb_returns = b_returns[mb_inds]
                if self.normalized_return:
                    mb_returns = (mb_returns - mb_returns.mean()) / (mb_returns.std() + 1e-8)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -self.clip_coef, self.clip_coef)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()

                # Final Loss
                loss = pg_loss - self.ent_coef * entropy_loss +  self.vf_coef * v_loss
                
                # Bacward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # if approx_kl > self.target_kl:
            #     break

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
        # print("SPS:", int(global_step / (time.time() - self.start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - self.start_time)), global_step)

    def save_model(self, update):
        try:
            import loralib as lora
            # Create the directory if it doesn't exist
            dirpath = f"{self.results_dir}/checkpoints"
            os.makedirs(dirpath, exist_ok=True)
            ppo_filepath = os.path.join(dirpath, f"ppo_update_{update}.pth")
            torch.save({
                'network_model': self.policy_model.network_model.state_dict(),
                'actor': self.policy_model.actor.state_dict(),
                'critic': self.policy_model.critic.state_dict(),
            }, ppo_filepath)
            logging.info(f"Saving ppo model weights for update {update} in {ppo_filepath}.")

            if self.train_vision:
                im_filepath = os.path.join(dirpath, f"im_update_{update}.pth")
                torch.save(lora.lora_state_dict(self.policy_model.image_model), im_filepath)
                logging.info(f"Saving image model weights for update {update} in {im_filepath}.")

        except Exception as e:
            print("Error occurred while saving model weights:", e)

    def load_model(self, policy_path, load_vision, vision_path):
        try:
            print(f"Loading ppo model weights from {policy_path}.")
            checkpoint = torch.load(policy_path, map_location='cpu')
            self.policy_model.network_model.load_state_dict(checkpoint['network_model'])
            self.policy_model.actor.load_state_dict(checkpoint['actor'])
            self.policy_model.critic.load_state_dict(checkpoint['critic'])
            # start_update = checkpoint['update']
            logging.info(f"Loading ppo model weights from {policy_path}.")

            if load_vision:
                print(f"Loading image model weights from {vision_path}.")
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.policy_model.image_model.load_state_dict(torch.load(vision_path), strict=False)
                logging.info(f"Loading image model weights from {vision_path}.")

        except Exception as e:
            print("Error occurred while loading model weights:", e)
            raise
