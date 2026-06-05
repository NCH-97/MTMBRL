import argparse
import functools
import os
import pathlib
import sys
import dill
#os.environ["MUJOCO_GL"] = "osmesa"
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
from ruamel.yaml import YAML

sys.path.append(str(pathlib.Path(__file__).parent))
import networks
import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd
import torch.nn.functional as F


to_np = lambda x: x.detach().cpu().numpy()

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self.Agent_type = 1
        
        if self.Agent_type == 0:
            self._task_behavior = models.ImagBehavior(config, self._wm)
        elif self.Agent_type == 1:
            self._task_behavior = models.ImagBehavior_PPO(config, self._wm)
        else:
            self._task_behavior = models.ImagBehavior(config, self._wm)

        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)
        
        # self._task_one_hot = tools.genrate_onehot_env_idx(self._config.train_envs_size, self._config.batch_size, self._config.envs_para).to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
            self._wm._context_buffer.clear()
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = obs['obs']
        # embed = self._wm.encoder(obs)
        obs_stack, action_stack, embed_stack = self._wm._context_buffer.get()
        task_one_hot = self._wm.ctx_encoder(obs_stack, action_stack, embed_stack, True).detach()
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"], task_one_hot=task_one_hot)

        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]

        feat = self._wm.dynamics.get_feat(latent)
        # print(f"feat {feat.shape}")
        x = feat
        x = torch.cat([feat, task_one_hot], dim=-1)
        if not training:
            actor = self._task_behavior.actor(x)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(x)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(x)
            action = actor.sample()

        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        self._wm._context_buffer.update(obs["obs"], action, embed)
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        # before_W = self._wm.dynamics._cell.layers[0].weight.data.clone()
        post, context, mets, _task_one_hot = self._wm._train(data)
        # after_W = self._wm.dynamics._cell.layers[0].weight.data.clone()
        # changed1 = not torch.allclose(before_W, after_W)
        metrics.update(mets)
        start = post
        old_log_prob = data['logprob']
        old_actions = data['action']
        reward = lambda f, s, a, one_hot: self._wm.heads["reward"](
            torch.cat([self._wm.dynamics.get_feat(s), one_hot], dim=-1)
        ).mode()
        # reward = lambda f, s, a, one_hot: self._wm.heads["reward"](
        #     self._wm.dynamics.get_feat(s)
        # ).mode()
        # before_A = self._task_behavior.actor.layers[0].weight.data.clone()
        # before_C = self._task_behavior.Q0.layers[0].weight.data.clone()
        if self.Agent_type == 0:
            metrics.update(self._task_behavior._train(start, reward, _task_one_hot)[-1])
        elif self.Agent_type == 1:
            metrics.update(self._task_behavior._train(start, reward, old_log_prob, old_actions, _task_one_hot)[-1])
        else:
            metrics.update(self._task_behavior._train(start, reward, _task_one_hot)[-1])

        # after_A = self._task_behavior.actor.layers[0].weight.data.clone()
        # after_C = self._task_behavior.Q0.layers[0].weight.data.clone()
        # changed2 = not torch.allclose(before_A, after_A)
        # changed3 = not torch.allclose(before_C, after_C)
        # print(f"WM changed {changed1} A changed {changed2} C changed {changed3}")
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_dataset_with_env_idx(episodes, config, task_num):

    assert config.batch_size % task_num == 0, \
        f"The batch_size {config.batch_size} must be divisible by task_num {task_num}"

    generators = [
        tools.sample_episodes_with_env_idx(
            episodes, config.batch_length, seed=config.seed + i, idx= i
        )
        for i in range(task_num)
    ]

    dataset = tools.from_generators_with_env_idx(
        generators, config.batch_size, task_num
    )
    return dataset

def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "MW":
       import envs.MW as mw
       env = mw.MW(task, config.action_repeat, config.size, seed=config.seed + id)
       env = wrappers.NormalizeActions(env)

    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env

def make_TL_ML_envs(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "MW":
        import envs.MW as mw
        time_limit = config.time_limit
        env_set = mw.MT_ML_MW(task, time_limit, para=config.envs_para) #MT10 5 MT50 1
    else:
        env_set = None
        raise NotImplementedError(suite)
    return env_set

def main(config):
    Need_Save = 0
    load_agent = 0

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    # make = lambda mode, id: make_env(config, mode, id)
    # train_envs = [make("train", i) for i in range(config.envs)]
    # eval_envs = [make("eval", i) for i in range(config.envs)]
    env_set = make_TL_ML_envs(config, "train", 0)
    config.train_task_num = env_set.train_task_num
    config.test_task_num = env_set.test_task_num
    ENV_type, taskset = config.task.split("_", 1)
    
    # need polish ----
    if taskset.startswith("MT"):
        train_envs = env_set.train_env
        eval_envs = env_set.eval_env
        train_task_size = len(train_envs)
        print(f"train_task_size: {train_task_size}")
        config.train_envs_size = train_task_size
    elif taskset.startswith("ML"):
        train_envs = env_set.train_env
        eval_envs = env_set.eval_env
        test_envs = env_set.test_env
        train_task_size = len(train_envs)
        test_task_size = len(test_envs)
        config.test_envs_size = test_task_size
        print(f"train_task_size: {train_task_size} test_task_size: {test_task_size}")
        config.train_envs_size = train_task_size
    else:
        raise NotImplementedError(taskset)
    # ----

    if ENV_type == "MW":
        need_success = True
    else:
        need_success = False

    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]

    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.train_envs_size, 1),
                    torch.tensor(acts.high).repeat(config.train_envs_size, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        print(f"train_envs: {len(train_envs)}")

        state, _ = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            task_name_dict=env_set.name_dict,
            para=env_set.para,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset_with_env_idx(train_eps, config, env_set.train_task_num)
    # eval_dataset = make_dataset_with_env_idx(eval_eps, config, env_set.train_task_num)
    
    # train_dataset, train_buckets, train_index_map = make_dataset_with_env_idx_fast(train_eps, config, env_set.train_task_num)
    # eval_dataset = make_dataset_with_env_idx_fast(eval_eps, config, env_set.train_task_num)

    # for k,v in train_eps.items():
    #     print(f"env_len: {len(v['env_idx'])} env_idx 0: {v['env_idx'][0]}")

    if Need_Save:
        print("Save config.")
        print(config)
        with open('Policy_AGG/Dream_config.dill', 'wb') as f:
            dill.dump(config, f)

    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    if load_agent:
        policy_dir = "./Policy/ML45"
        Load_RSSM_name_pkl = 'basketball'
        Load_RSSM_name_pkl += "_agent.pkl"
        Load_actor_name_pkl = 'basketball'
        Load_actor_name_pkl += "_agent.pkl"
        Load_critic_name_pkl = 'basketball'
        Load_critic_name_pkl += "_agent.pkl"

        with open(policy_dir + '/' + Load_RSSM_name_pkl, "rb") as f:
            loaded_policy = dill.load(f)
        agent._wm.encoder = loaded_policy[0]
        agent._wm.dynamics = loaded_policy[1]
        agent._wm.heads["decoder"] = loaded_policy[2]
        agent._wm.heads["reward"] = loaded_policy[3]
        agent._wm.heads["cont"] = loaded_policy[4]

        with open(policy_dir + '/' + Load_actor_name_pkl, "rb") as f:
            loaded_policy = dill.load(f)
        agent._task_behavior.actor = loaded_policy[5]

        with open(policy_dir + '/' + Load_critic_name_pkl, "rb") as f:
            loaded_policy = dill.load(f)
        agent._task_behavior.value = loaded_policy[6]

    print("Start training.")
    
    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")

            eval_policy = functools.partial(agent, training=False)

            eval_policy.func._task_behavior.actor.eval()
            eval_policy.func._task_behavior.value.eval()
            eval_policy.func._wm.dynamics.eval()
            # eval_policy.func._wm.heads["decoder"].eval()
            eval_policy.func._wm.heads["reward"].eval()
            # eval_policy.func._wm.heads["cont"].eval()
            # eval_policy.func._wm.encoder.eval()

            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
                need_success=need_success,
                task_name_dict=env_set.name_dict,
                para=env_set.para,
            )

            print("Agent Saving")
            objs = (agent._wm.dynamics, agent._wm.heads['reward'], agent._task_behavior.actor, agent._task_behavior.value)

            with open(logdir / "save_agent.pkl", "wb") as f:
                dill.dump(objs, f)
            
            items_to_save = {
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            }
            torch.save(items_to_save, logdir / "latest.pt")
            
            # if config.video_pred_log:
            #     video_pred = agent._wm.video_pred(next(eval_dataset))
            #     logger.video("eval_openl", to_np(video_pred))
        print(f"Start training.")
        print(f"Steps {agent._step} / {config.steps + config.eval_every}")
        state, _ = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")

    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass
    objs = (agent._wm.dynamics, agent._wm.heads['reward'], agent._task_behavior.actor, agent._task_behavior.value)
    with open(logdir / "save_agent_f.pkl", "wb") as f:
        dill.dump(objs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    yaml_parser = YAML(typ='safe', pure=True)
    configs = yaml_parser.load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
