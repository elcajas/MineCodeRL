from __future__ import annotations
from typing import Any, Dict, Optional, SupportsFloat, Tuple

import numpy as np

import gymnasium as gym
from gymnasium.core import RenderFrame

# import minedojo
# from minedojo.sim import ALL_CRAFT_SMELT_ITEMS, ALL_ITEMS
# from minedojo.sim.wrappers.ar_nn import ARNNWrapper

from envs.dummy_env import MineEnv
from envs.animal_zoo import HuntCowDenseRewardEnv, HuntSheepDenseRewardEnv, MilkCowDenseRewardEnv, ShearSheepDenseRewardEnv
from envs.mob_combat import CombatSpiderDenseRewardEnv, CombatZombieDenseRewardEnv

# N_ALL_ITEMS = len(ALL_ITEMS)
# ITEM_ID_TO_NAME = dict(enumerate(ALL_ITEMS))
# ITEM_NAME_TO_ID = dict(zip(ALL_ITEMS, range(N_ALL_ITEMS)))

ACTION_MAP_KEY = {
    0: np.array([0, 0, 0, 0, 0, 0, 0, 0]),  # no-op
    1: np.array([1, 0, 0, 0, 0, 0, 0, 0]),  # forward
    2: np.array([2, 0, 0, 0, 0, 0, 0, 0]),  # back
    3: np.array([0, 1, 0, 0, 0, 0, 0, 0]),  # left
    4: np.array([0, 2, 0, 0, 0, 0, 0, 0]),  # right
    5: np.array([1, 0, 1, 0, 0, 0, 0, 0]),  # jump + forward
    6: np.array([0, 0, 0, 0, 0, 1, 0, 0]),  # use
    7: np.array([0, 0, 0, 0, 0, 3, 0, 0]),  # attack
    8: np.array([0, 0, 0, 0, 0, 0, 0, 0]),  # no-op
}

ACTION_MAP_CAM = {
    0: np.array([0, 0, 0, 12, 12, 0, 0, 0]),  # no-op
    1: np.array([0, 0, 0, 11, 12, 0, 0, 0]),  # pitch down (-15)
    2: np.array([0, 0, 0, 13, 12, 0, 0, 0]),  # pitch up (+15)
    3: np.array([0, 0, 0, 12, 12, 0, 0, 0]),  # no-op
    4: np.array([0, 0, 0, 12, 11, 0, 0, 0]),  # yaw down (-15)
    5: np.array([0, 0, 0, 12, 13, 0, 0, 0]),  # yaw up (+15)
    6: np.array([0, 0, 0, 12, 12, 0, 0, 0]),  # no-op
}

class MineDojoWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        seed: int,
        pitch_limits: Tuple[int, int] = (-60, 60),
        sticky_attack: Optional[int] = 30,
        sticky_jump: Optional[int] = 10,

    ):
        self._pitch_limits = pitch_limits
        self._sticky_attack = sticky_attack
        self._sticky_jump = sticky_jump
        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._pos = None

        super().__init__(env)
        self._render_mode = "rgb_array"

        self.action_space = gym.spaces.MultiDiscrete(np.array([len(ACTION_MAP_KEY.keys()), len(ACTION_MAP_CAM.keys())]))
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(0, 255, self.env.observation_space["rgb"].shape, np.uint8),
                "biome_id": gym.spaces.Box(0, 167, self.env.observation_space["location_stats"]["biome_id"].shape, np.uint8),
                "pos": gym.spaces.Box(-640000.0, 640000.0, self.env.observation_space["location_stats"]["pos"].shape, np.float32),
                "yaw": gym.spaces.Box(-180.0, 180.0, self.env.observation_space["location_stats"]["yaw"].shape, np.float32),
                "pitch": gym.spaces.Box(-180.0, 180.0, self.env.observation_space["location_stats"]["pitch"].shape, np.float32),
            }
        )
        self.seed(seed)
    
    @property
    def render_mode(self) -> str | None:
        return self._render_mode
    
    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        obs = self.env.reset()
        self._update_pos(obs)

        obs = self._convert_obs(obs)
        info = {}

        self._sticky_jump_counter = 0
        self._sticky_attack_counter = 0
        info = {}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        a = action
        action = self._convert_action(action)

        next_pitch = self._pos["pitch"] + (action[3] - 12) * 15
        if not (self._pitch_limits[0] <= next_pitch <= self._pitch_limits[1]):
            action[3] = 12

        obs, reward, done, info = self.env.step(action)
        self._update_pos(obs)
        obs = self._convert_obs(obs)
        info = {}
        
        return obs, reward, done, False, info
    
    def seed(self, seed: int) -> None:
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
    
    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        # assert action.ndim == 2
        action = action.squeeze()
        converted_action = ACTION_MAP_KEY[action[0]].copy() + ACTION_MAP_CAM[action[1]].copy()
        return converted_action

    def _convert_obs(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        converted_obs = {
            "rgb": obs["rgb"].copy(),
            "biome_id": obs["location_stats"]["biome_id"],
            "pos": obs["location_stats"]["pos"],
            "yaw": obs["location_stats"]["yaw"],
            "pitch": obs["location_stats"]["pitch"],
        }
        return converted_obs
    
    def _update_pos(self, obs: Dict[str, Any]) -> None:
        self._pos = {
            "x": float(obs["location_stats"]["pos"][0]),
            "y": float(obs["location_stats"]["pos"][1]),
            "z": float(obs["location_stats"]["pos"][2]),
            "pitch": float(obs["location_stats"]["pitch"].item()),
            "yaw": float(obs["location_stats"]["yaw"].item()),
        }

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "human":
            super().render()
        elif self.render_mode == "rgb_array":
            if self.env.unwrapped._prev_obs is None:
                return None
            else:
                return self.env.unwrapped._prev_obs["rgb"].transpose((1, 2, 0))
        return None
    
def create_env(task, seed):
    step_penalty = 0
    nav_reward_scale = 0
    attack_reward = 5
    success_reward = 200
    env_dic = {
        'dummy env': MineEnv,
        'combat spider': CombatSpiderDenseRewardEnv,
        'combat zombie': CombatZombieDenseRewardEnv,
        'hunt a cow': HuntCowDenseRewardEnv,
        'hunt a sheep': HuntSheepDenseRewardEnv,
        'milk a cow': MilkCowDenseRewardEnv,
        'shear a sheep': ShearSheepDenseRewardEnv,
    }

    env_cls = env_dic[task]

    if 'combat' in task:
        env = env_cls(
            step_penalty=step_penalty,
            attack_reward=attack_reward,
            success_reward=success_reward
        )
        return MineDojoWrapper(env, seed)

    if 'milk' in task or 'shear' in task:    
        nav_reward_scale = 0
        attack_reward = 0

    env = env_cls(
        step_penalty=step_penalty,
        nav_reward_scale=nav_reward_scale,
        attack_reward=attack_reward,
        success_reward=success_reward,
    )
    return MineDojoWrapper(env, seed)

def make_env(task, seed, idx, run_name):
    def thunk():
        env = create_env(task, seed)
        # if idx == 0:
        #     env = gym.wrappers.RecordVideo(env, f"{run_name}/videos/agent{idx}", episode_trigger=lambda x: (x % 5==0) and x > 0)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk