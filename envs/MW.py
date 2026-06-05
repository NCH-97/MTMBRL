import os
os.environ["LD_LIBRARY_PATH"] = ("/home/nch97/.mujoco/mujoco210/bin:/usr/lib/nvidia:" +
                                 os.environ.get("LD_LIBRARY_PATH", ""))
import random
import gymnasium
import numpy as np
import metaworld
from PIL import Image
import metaworld.env_dict as _env_dict
import envs.wrappers as wrappers
# import wrappers
os.environ["MUJOCO_GL"] = "egl"

class MT_ML_MW:
    def __init__(self, task_name, time_limit=200, para=1):
        """
        Args:
            task_name: MT10 / MT50 / ML10 / ML45
            time_limit: episode length
            para: number of parallel environments per task
        """
        self.para = para
        self.train_env = []
        self.test_env = []
        self.eval_env = []
        self.name_dict = {}
        self.test_name_dict = {}

        def build_env(key, idx, seed):
            env = MW(key, seed=seed, idx=idx)
            env = wrappers.NormalizeActions(env)
            env = wrappers.TimeLimit(env, time_limit)
            env = wrappers.SelectAction(env, key="action")
            env = wrappers.UUID(env)
            return env

        if task_name.startswith("ML"):
            print("Loading ML task", task_name)

            if task_name == "ML10":
                # train
                self.train_task_num = 10
                for idx, key in enumerate(_env_dict.ML10_V3["train"].keys()):
                    for seed in range(1, para + 1):
                        env = build_env(key, idx, seed)
                        self.train_env.append(env)
                        self.eval_env.append(env)
                    self.name_dict[idx] = key
                    

                # test
                self.test_task_num = 5
                for idx, key in enumerate(_env_dict.ML10_V3["test"].keys()):
                    for seed in range(1, para + 1):
                        env = build_env(key, idx + 10, seed)
                        self.test_env.append(env)
                    
                    self.test_name_dict[idx] = key

            elif task_name == "ML45":
                # train
                self.train_task_num = 45
                for idx, key in enumerate(_env_dict.ML45_V3["train"].keys()):
                    for seed in range(1, para + 1):
                        env = build_env(key, idx, seed)
                        self.train_env.append(env)
                        self.eval_env.append(env)
                    self.name_dict[idx] = key
                    
                # test
                self.test_task_num = 5
                for idx, key in enumerate(_env_dict.ML45_V3["test"].keys()):
                    for seed in range(1, para + 1):
                        env = build_env(key, idx + 45, seed)
                        self.test_env.append(env)
                    self.test_task_num = idx
                    self.test_name_dict[idx] = key
            else:
                raise NotImplementedError(task_name)

        else:
            print("Loading MT task", task_name)
            self.test_task_num = 0
            
            if task_name == "MT10":
                self.train_task_num = 10
                for idx, key in enumerate(_env_dict.MT10_V3.keys()):
                    for seed in range(1, para + 1):
                        env = build_env(key, idx, seed)
                        self.train_env.append(env)
                        self.eval_env.append(env)
                    self.name_dict[idx] = key
                    
            elif task_name == "MT50":
                self.train_task_num = 50
                for idx, key in enumerate(_env_dict.MT50_V3.keys()):
                    for seed in range(1, para + 1):
                        env = build_env(key, idx, seed)
                        self.train_env.append(env)
                        self.eval_env.append(env)
                    self.name_dict[idx] = key
            else:
                raise NotImplementedError(task_name)

        print(
            f"Loaded train envs {len(self.train_env)}, "
            f"eval envs {len(self.eval_env)}, "
            f"test envs {len(self.test_env)} "
            f"(para={para}) for task {task_name}"
        )

class MW(gymnasium.Env):
    metadata = {}
    def __init__(self, name, action_repeat=1,size=(64,64), camera=None, seed=1, idx=None):
        render_mode = 'rgb_array'
        camera_name = 'corner'
        self.env_idx = idx
        self.seed = seed
        if isinstance(name, str):
            # ml1 = metaworld.ML1(name)
            self._env = gymnasium.make('Meta-World/MT1', env_name=name, seed=seed, render_mode=render_mode, camera_name=camera_name)
            obs = self._env.reset()

        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            self._camera = None
        else:
            self._camera = camera_name
        self.reward_range = [-np.inf, np.inf]
        self._step = 0

    @property
    def observation_space(self):
        spaces = {}
        low=np.array(self._env.observation_space.low)
        high=np.array(self._env.observation_space.high)
        shape=np.array(self._env.observation_space.shape)
        spaces["obs"] = gymnasium.spaces.Box(low=low, high=high, shape=shape, dtype=np.float64)
        return gymnasium.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        # print(f"MW: step={self._step}, action={action}")
        reward = 0
        obs = []
        terminate = False
        truncate = False

        for _ in range(self._action_repeat):
            obs, reward1, terminate, truncate, info = self._env.step(action)
            reward += reward1 or 0
            if terminate or truncate:
                break
        obs_dict = {'obs': obs}
        obs_r = {key: [val] if len(val.shape) == 0 else val for key, val in obs_dict.items()}
        self._step += 1
        obs_r["is_terminal"] = terminate or truncate
        obs_r["is_first"] = False
        info_r = info
        # print(f"MW: step={self._step}, reward={reward}, done={terminate or truncate}")
        return obs_r, reward, terminate or truncate, info_r

    def reset(self):
        time_step, _  = self._env.reset(seed=self.seed)
        obs_dict = {'obs':time_step}
        obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs_dict.items()}
        obs["is_terminal"] = False
        obs["is_first"] = True
        self._step = 0
        return obs

    def render(self, *args, **kwargs):

        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        image = self._env.render()
        image = image[::-1, :, :]
        image = Image.fromarray(image)
        resized_image = image.resize((64, 64), resample=Image.LANCZOS)
        resized_image = np.array(resized_image)

        return resized_image

def main():
    # 加载 MetaWorld 环境
    env = metaworld.ML10()  # 示例：MT10 任务环境
    # 列出 `env` 中的所有属性和方法
    # print(dir(env._train_classes.keys))
    # print(env._train_classes)
    # print(env._train_classes.keys())
    # print(_env_dict.ML10_V3["train"].keys())
    # print(_env_dict.ML10_V3["test"].keys())
    # print(_env_dict.MT10_V3.keys())

    config = {}
    config['time_limit'] = 1000
    MT_ML_MW("MT10", para=10)

    # for key in env._train_classes.keys():
    #     print(key)

    # print(metaworld.ML1.ENV_NAMES)  # Check out the available environments
    # print(f"MT10{list(_env_dict.MT10_V3.keys())}")
    # print(f"MT50{list(_env_dict.MT50_V3.keys())}")
    # mw = MW('window-open-v3')
    # print(mw.observation_space)
    # print(mw._env.observation_space)



if __name__ == "__main__":

    main()
