import copy
import torch
from torch import nn
import numpy as np
import torch.optim as optim
import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()

class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        config.task_embed_size = config.train_envs_size // config.envs_para
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        # print(f"shapes: {shapes}")
        self.outdim = obs_space['obs'].shape[0]
        # print(f"outdim: {self.outdim}")
        # self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        # self.outdim = self.encoder.outdim
        self.embed_size = self.outdim
        self.ctx_encoder = networks.ContextEncoder(
            obs_space['obs'].shape[0], 
            config.num_actions,
            config.task_embed_size,
            self.embed_size,
            rssm_state_dim = config.dyn_stoch * config.dyn_discrete,
            use_crossmodal = True,
            device=config.device
        )

        self._context_buffer = tools.OnlineContextBuffer(
            self._config.train_envs_size, 
            config.batch_length, 
            obs_space['obs'].shape[0], 
            config.num_actions,
            self.embed_size,
            config.device,
        )

        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
            config,
        )
        self.heads = nn.ModuleDict()
        
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter + config.task_embed_size #32*32+300
        else:
            feat_size = config.dyn_stoch + config.dyn_deter + config.task_embed_size

        # self.heads["decoder"] = networks.MultiDecoder(
        #     feat_size, shapes, **config.decoder
        # )

        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )

        # self.heads["cont"] = networks.MLP(
        #     feat_size,
        #     (),
        #     config.cont_head["layers"],
        #     config.units,
        #     config.act,
        #     config.norm,
        #     dist="binary",
        #     outscale=config.cont_head["outscale"],
        #     device=config.device,
        #     name="Cont",
        # )
        
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )
        self._task_one_hot = None
        # self._task_one_hot = tools.genrate_onehot_env_idx(self._config.train_envs_size, self._config.batch_size, self._config.envs_para).to(self._config.device)
        # self.harmony_s1 = networks.Harmonizer(name='har_s1')
        # self.harmony_s2 = networks.Harmonizer(name='har_s2')
        # self.harmony_s3 = networks.Harmonizer(name='har_s3')

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)

        data = self.preprocess(data)
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                # embed = self.encoder(data)
                embed = data['obs']
                B, T = data["obs"].shape[:2]
                ctx_losses = []
                cm_losses = []
                states = (None, None)
                for t in range(1, data["obs"].shape[1]):
                    pad_width = T - t
                    padded_obs = self.edge_pad_left(data["obs"][:, :t], pad_width)
                    padded_action = self.edge_pad_left(data["action"][:, :t], pad_width)
                    if "embed" in self._config.ctx_encoder["inputs"]:
                        padded_embed = self.edge_pad_left(embed[:, :t], pad_width)
                    else:
                        padded_embed = None
                    ctx_enc_data = {
                        "action": padded_action,
                        "obs": padded_obs,
                        "embed": padded_embed,
                    }
                    rolling_ctx = self.ctx_encoder(**ctx_enc_data)
                    loss_fd , _ = self.ctx_encoder.compute_representation_loss(**ctx_enc_data, context=rolling_ctx)
                    if loss_fd.ndim == 0:
                        loss_fd = loss_fd.expand(B)
                    ctx_losses.append(loss_fd)
                    if self._config.with_cross_model_loss:
                        # print(f"rolling_ctx shape: {rolling_ctx.shape}")  # debug
                        states = self.dynamics.obs_step(
                            states[0], 
                            data["action"][:, t - 1],
                            embed[:, t], 
                            data["is_first"][:, t], 
                            task_one_hot=rolling_ctx[:, -1])
                        stoch_state = self.dynamics.get_stoch_feat(states[0])
                        loss_cm = self.ctx_encoder.compute_crossmodal_loss(rolling_ctx[:, -1], stoch_state)
                        if loss_cm.ndim == 0:
                            loss_cm = loss_cm.expand(B)
                        cm_losses.append(loss_cm)

                ctx_losses = torch.stack(ctx_losses, dim=1)
                # print(f"ctx_losses[0] shape: {ctx_losses[0].shape}")  # debug
                if self._config.with_cross_model_loss:
                    cm_losses = torch.stack(cm_losses, dim=1)
                zero_tail = torch.zeros(
                    B,
                    1,
                    device=ctx_losses.device,
                    dtype=ctx_losses.dtype,
                )

                self.dynamics._task_one_hot = self.ctx_encoder(data["obs"],data["action"], embed=None, eval_mode=True).detach()
                self._task_one_hot = self.dynamics._task_one_hot[:, None, :].expand(-1, T, -1).detach()
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    x = feat
                    x = torch.cat([feat, self._task_one_hot], dim=-1)
                    pred = head(x)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}

                losses["dali_cr"] = torch.cat([ctx_losses, zero_tail], dim=-1)  # [B, T]
                if self._config.with_cross_model_loss:
                    losses["cross_model"] = torch.cat([cm_losses, zero_tail], dim=-1)  # [B, T]

                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
                
            metrics = self._model_opt(torch.mean(model_loss))

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics, self._task_one_hot

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs_is_first = obs["is_first"]

        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)
    
    def edge_pad_left(self, x, pad_width):
        """
        Equivalent to:
            jnp.pad(x, ((0, 0), (pad_width, 0), (0, 0)), mode='edge')

        Args:
            x: [B, t, D]
            pad_width: int

        Returns:
            padded_x: [B, t + pad_width, D]
        """
        if pad_width == 0:
            return x

        # repeat the first time step pad_width times
        left_pad = x[:, :1].expand(-1, pad_width, -1)
        return torch.cat([left_pad, x], dim=1)


    def mask_state(self, rssm, state, mask):
        """
        Apply RSSM mask to a nested state dict.

        Args:
            state: dict of tensors, each [B, ...]
            mask: [B] or [B, 1]

        Returns:
            masked state dict
        """
        return {
            k: rssm._mask(v, mask)
            for k, v in state.items()
        }


    def add_state(self, a, b):
        """
        Add two RSSM state dicts.
        """
        return {
            k: a[k] + b[k]
            for k in a.keys()
        }


    def detach_state(self, state):
        """
        Equivalent to sg(state) in JAX.
        """
        return {
            k: v.detach()
            for k, v in state.items()
        }

class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter #+ config.train_envs_size // config.envs_para
        else:
            feat_size = config.dyn_stoch + config.dyn_deter #+ config.train_envs_size // config.envs_para
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)
            self._task_one_hot = tools.genrate_onehot_env_idx(self._config.train_envs_size, self._config.batch_size, self._config.envs_para).to(self._config.device)

    def _train(
        self,
        start,
        objective,
        task_one_hot,
    ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon, task_one_hot=task_one_hot
                )
                flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
                # task_one_hot = self._task_one_hot.unsqueeze(1).expand(-1, self._config.batch_length, -1)
                env_flat = flatten(task_one_hot)
                h_env_flat = env_flat.unsqueeze(0).expand(imag_feat.shape[0], -1, -1)
                #imag_feat = torch.cat([imag_feat, h_env_flat], dim=-1)
                reward = objective(imag_feat, imag_state, imag_action, h_env_flat)
                actor_ent = self.actor(imag_feat).entropy()
                # state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward, h_env_flat
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss))
            metrics.update(self._value_opt(value_loss))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon, task_one_hot=None):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        # task_one_hot = self._task_one_hot.unsqueeze(1).expand(-1, self._config.batch_length, -1)
        env_flat = flatten(task_one_hot)
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            x = inp
            #x = torch.cat([inp, env_flat], dim=-1)
            action = policy(x).sample()
            succ = dynamics.img_step(state, action, task_one_hot=env_flat)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward, h_env_flat):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            # inp_e = torch.cat([inp, h_env_flat], dim=-1)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

class ImagBehavior_PPO(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior_PPO, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter + config.train_envs_size // config.envs_para
        else:
            feat_size = config.dyn_stoch + config.dyn_deter + config.train_envs_size // config.envs_para
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)
        self.old_actor = copy.deepcopy(self.actor)
        self.old_actor.eval()
        self._task_one_hot = tools.genrate_onehot_env_idx(self._config.train_envs_size, self._config.batch_size, self._config.envs_para).to(self._config.device)

    def _train(
        self,
        start,
        objective,
        old_logprob=None,
        old_actions=None,
        task_one_hot=None,
    ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon, task_one_hot=task_one_hot
                )

                flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
                # task_one_hot = self._task_one_hot.unsqueeze(1).expand(-1, self._config.batch_length, -1)
                env_flat = flatten(task_one_hot)
                h_env_flat = env_flat.unsqueeze(0).expand(imag_feat.shape[0], -1, -1)
                imag_feat = torch.cat([imag_feat, h_env_flat], dim=-1)
                reward = objective(imag_feat, imag_state, imag_action, h_env_flat)
                actor_ent = self.actor(imag_feat).entropy()
                # state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward, h_env_flat
                )
                feat4ratio = self._world_model.dynamics.get_feat(start)
                # task_one_hot = self._task_one_hot.unsqueeze(1).expand(-1, self._config.batch_length, -1)
                # print(f"feat4ratio {feat4ratio.shape}, env_flat {task_one_hot.shape}")  # debug
                feat4ratio = torch.cat([feat4ratio, task_one_hot], dim=-1)
                actor4ratio = self.actor(feat4ratio)
                old_actions = torch.tensor(old_actions, device=self._config.device, dtype=torch.float32)
                new_logprob = actor4ratio.log_prob(old_actions)
                old_logprob = torch.tensor(old_logprob, device=self._config.device, dtype=torch.float32)
                ratio = torch.exp(new_logprob - old_logprob)
                ratio = ratio.to(self._config.device)
                metrics["ratio"] = to_np(torch.mean(ratio))
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                    ratio
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = actor_loss.mean()
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss))
            metrics.update(self._value_opt(value_loss))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon, task_one_hot=None):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        # task_one_hot = self._task_one_hot.unsqueeze(1).expand(-1, self._config.batch_length, -1)
        env_flat = flatten(task_one_hot)  # (B*T, K)
        start = {k: flatten(v) for k, v in start.items()}
        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            x = inp
            x = torch.cat([inp, env_flat], dim=-1)
            action = policy(x).sample()
            succ = dynamics.img_step(state, action, task_one_hot=env_flat)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward, h_env_flat):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            inp = torch.cat([inp, h_env_flat], dim=-1)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
        ratio,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics": # <-
            actor_target = adv
            clip_eps = 0.2
            ratio = ratio.reshape(-1)
            loss = torch.min(ratio * adv,
                  torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv)
            actor_target = loss
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics
    
    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1