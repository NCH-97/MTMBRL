import os
os.environ["LD_LIBRARY_PATH"] = "/home/nch97/.mujoco/mujoco210/bin:/usr/lib/nvidia:" + os.environ.get("LD_LIBRARY_PATH", "")
import gymnasium as gym
import numpy as np

from PIL import Image
os.environ["MUJOCO_GL"] = "egl"

class HC:
    def __init__(self, name, action_repeat=1,size=(64,64), camera=None, seed=0):
        domain, task = name.split("_",1)
        render_mode = 'rgb_array'
        self._env = gym.make('HalfCheetah-v5',render_mode=render_mode)
        self._size = size
        self._action_repeat = action_repeat
        self.reward_range = [-np.inf, np.inf]
        if task == "G1":
            self._env.unwrapped.model.opt.gravity[:] = np.array([0.0, 0.0, -9.81])
        elif task == "G.5":
            self._env.unwrapped.model.opt.gravity[:] = np.array([0.0, 0.0, -4.91])
        elif task == "G.75":
            self._env.unwrapped.model.opt.gravity[:] = np.array([0.0, 0.0, -7.36])
        elif task == "G1.25":
            self._env.unwrapped.model.opt.gravity[:] = np.array([0.0, 0.0, -12.26])
        elif task == "G1.5":
            self._env.unwrapped.model.opt.gravity[:] = np.array([0.0, 0.0, -14.72])
        else:
            print('TASK ERROR')
    @property
    def observation_space(self):
        space = {}
        space["obs"] = self._env.observation_space
        space["image"] = gym.spaces.Box(0,255,self._size+(3,),dtype=np.uint8)
        return gym.spaces.Dict(space)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        reward = 0
        obs = []
        for _ in range(self._action_repeat):
            obs, reward1, terminate, truncate, info = self._env.step(action)
            reward += reward1 or 0
            if terminate or truncate:
                break
        obs_dict = {"obs":obs}
        obs_r = {key: [val] if len(val.shape) == 0 else val for key, val in obs_dict.items()}
        obs_r["image"] = self.render()
        obs_r["is_terminal"] = terminate or truncate
        obs_r["is_first"] = False
        info_r = info
        return obs_r, reward, terminate or truncate, info_r

    def reset(self):
        time_step, _  = self._env.reset()
        obs_dict = {'obs':time_step}
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs_dict.items()}
        obs["image"] = self.render()
        obs["is_terminal"] = False
        obs["is_first"] = True
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        image = self._env.render()
        image = Image.fromarray(image)
        resized_image = image.resize((64, 64), resample=Image.LANCZOS)
        resized_image = np.array(resized_image)
        return resized_image

def main():
    env = gym.make('HalfCheetah-v5',render_mode='rgb_array')
    env.reset()
    image = env.render()
    Image.fromarray(image).save("1.png")
    hc = HC("HC_HC_HC")
    print(hc._env.unwrapped.model.opt.gravity)


if __name__ == "__main__":
    main()