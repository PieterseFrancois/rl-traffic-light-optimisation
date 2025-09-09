# ... imports ...
import gymnasium as gym
import numpy as np
import traci

class IntersectionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, agent, ticks_per_decision: int = 5, max_seconds: float | None = None):
        super().__init__()
        self.agent = agent
        self.ticks_per_decision = int(ticks_per_decision)
        self.max_seconds = None if max_seconds is None else float(max_seconds)

        # --- observation & action spaces ---
        # pad K to >=1 using agent's shape helper
        K_eff, D_emb = self.agent.nbr_embed_shape
        # infer F_raw once from a probe obs
        probe = self.agent._build_obs()
        F_raw = int(np.asarray(probe["self_raw"]).size)

        self.observation_space = gym.spaces.Dict({
            "self_raw":  gym.spaces.Box(low=-np.inf, high=np.inf, shape=(F_raw,), dtype=np.float32),
            "nbr_embed": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(K_eff, D_emb), dtype=np.float32),
        })
        # discrete actions over the mapping defined in the action module
        n_actions = len(self.agent.action.map or self.agent.action.map_state)
        self.action_space = gym.spaces.Discrete(n_actions)

        self._t0 = float(traci.simulation.getTime())  # type: ignore
        self._last_obs = probe

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # no SUMO reset here (caller already started sim)
        self._t0 = float(traci.simulation.getTime())  # type: ignore
        self._last_obs = self.agent._build_obs()
        info = {}
        return self._last_obs, info

    def step(self, action):
        # 1) SB3 supplies the discrete action index; apply it via the action module
        a = int(np.asarray(action).squeeze())
        self.agent.action.apply(a)

        # 2) Advance SUMO for this decision window, servicing amber/all-red state machine
        for _ in range(max(1, self.ticks_per_decision)):
            traci.simulationStep()
            self.agent.action.tick()

        # 3) Build next obs at the decision boundary
        obs_next = self.agent._build_obs()

        # 4) Compute reward (post-transition state), if a reward module is set
        reward = 0.0
        if getattr(self.agent, "reward", None) is not None:
            try:
                info = dict()
                info["tls_id"] = self.agent.cfg.tls_id
                # chosen action’s target green string (state-string mode)
                if self.agent.action.map_state is not None and 0 <= action < len(self.agent.action.map_state):
                    info["action_state"] = self.agent.action.map_state[action]
                # actual current TLS state (may be amber/red if called immediately after apply)
                info["cur_state"] = traci.trafficlight.getRedYellowGreenState(self.agent.cfg.tls_id)
                reward = float(self.agent.reward.compute(obs_next, action, info))
            except Exception as e:
                print(f"Error computing reward: {e}")
                reward = 0.0  # keep training running even if reward errors

        # 5) Book-keep (optional: keep agent history up to date)
        try:
            phase = traci.trafficlight.getPhase(self.agent.cfg.tls_id)
            self.agent._log(t=traci.simulation.getTime(), action=a, phase=int(phase), reward=reward)  # type: ignore
        except Exception:
            pass

        # 6) Termination/truncation
        terminated = bool(traci.simulation.getMinExpectedNumber() <= 0)  # type: ignore
        truncated = False
        if self.max_seconds is not None:
            truncated = (float(traci.simulation.getTime()) - self._t0) >= self.max_seconds  # type: ignore

        return obs_next, reward, terminated, truncated, {}