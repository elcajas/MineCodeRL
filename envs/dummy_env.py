import pickle
import pathlib
import numpy as np
import imageio
import gymnasium as gym

dir_path = pathlib.Path(__file__).parent.resolve()

## DUMMY ENV
class MineEnv:
    def __init__(self, step_penalty, nav_reward_scale, attack_reward, success_reward):
        self.val = 0
        self._elapsed_steps = 0
        with open(dir_path.joinpath("episode_dict.pkl"), 'rb') as f:
            self.episode = pickle.load(f)
        self._episode_len = len(self.episode['rewards'])
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(0, 255, self.episode['states'][0][0]["rgb"].shape, np.uint8),
                "location_stats": gym.spaces.Dict(
                    {
                        "biome_id": gym.spaces.Box(0, 167, self.episode['states'][0][0]["location_stats"]["biome_id"].shape, np.uint8),
                        "pos": gym.spaces.Box(-640000.0, 640000.0, self.episode['states'][0][0]["location_stats"]["pos"].shape, np.float32),
                        "yaw": gym.spaces.Box(-180.0, 180.0, self.episode['states'][0][0]["location_stats"]["yaw"].shape, np.float32),
                        "pitch": gym.spaces.Box(-180.0, 180.0, self.episode['states'][0][0]["location_stats"]["pitch"].shape, np.float32),
                    }
                )
            }
        )
        self.metadata = None

    def reset(self):
        self._elapsed_steps = 0
        return self.episode['states'][0][0]

    def step(self, action):
        # mu, sigma = 0, 0.01

        # obs = self.obs
        # reward = 1 + np.random.normal(mu, sigma)
        # if action.item() == 1:
        #     reward = 1
        # else:
        #     reward = 0
        # done = False
        # info = None

        self._elapsed_steps += 1
        obs, reward, done, info = self.episode['states'][self._elapsed_steps]

        return obs, reward, done, info

    def close(self):
        return None
    
    def show_episode(self):
        frames = []
        for state in self.episode['states']:
            frame = state[0]['rgb'].transpose(1, 2, 0) # Change to (H, W, C)
            frames.append(frame)
        
        writer = imageio.get_writer('dummy_episode.mp4', fps=10)
        for frame in frames:
            writer.append_data(frame)
        writer.close()

if __name__ == "__main__":
    env = MineEnv(
        step_penalty=0,
        nav_reward_scale=1,
        attack_reward=1,
        success_reward=1
    )
    # import ipdb; ipdb.set_trace()
    env.show_episode()