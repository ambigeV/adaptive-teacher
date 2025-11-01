from typing import List, Tuple, Optional, cast
import numpy as np
import torch
import torch.nn as nn

# -------- QD archive (uniform cell sampling, best entry per cell = k=1) --------
def _bd_from_state(x: np.ndarray, horizon: int, bins_per_dim: int = 4) -> tuple:
    z = (x / (horizon - 1)).astype(np.float32)           # [0,1]
    b = np.clip((z * bins_per_dim).astype(int), 0, bins_per_dim - 1)
    return tuple(b.tolist())

class QDArchive:
    def __init__(self, bins_per_dim: int = 4):
        self.bins = bins_per_dim
        self.store = {}  # cell -> {"state": np.ndarray, "reward": float}

    def add(self, state: np.ndarray, reward: float, horizon: int):
        cell = _bd_from_state(state, horizon, self.bins)
        cur = self.store.get(cell)
        if cur is None or reward > cur["reward"]:
            self.store[cell] = {"state": state.copy(), "reward": float(reward)}

    def coverage(self) -> int:
        return len(self.store)

    # --- uniform cell sampling, then pick best entry (k=1) ---
    def sample_best_uniform(self, k: int) -> List[dict]:
        if not self.store: return []
        cells = list(self.store.keys())
        k = min(k, len(cells))
        idx = np.random.choice(len(cells), size=k, replace=False)
        return [ self.store[cells[i]] for i in idx ]

# -------- QD: back-and-forth with crossover/mutation guided by QD elites --------
def qd_back_and_forth(
    agent,                                              # BaseAgent (TB) with .model, .envs, .ndim, .eps
    train_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    archive: QDArchive,
    ls_back_ratio: float = 0.5,                         # reuse same “cut ratio” interface
    iterations: int = 4,                                # same outer loops as LS
    qd_cells_per_iter: int = 0,                         # how many elites to draw each iter (0 = auto = batch_size)
    mutation_prob: float = 0.15,                        # per-step mutation prob
    crossover_prob: float = 0.7,                        # probability to bias toward elite’s direction
    eps_noisy: bool = False,                            # same as LS
) -> Tuple[List[Tuple], List[Tuple], int]:
    """
    Returns:
      new_train_batch: list[(traj_obs, traj_s, traj_a, reward)]
      visited_x:       list[terminal_state_tuple]
      n_new_rewards:   int (how many new terminal rewards consumed)
    Contract matches 'back_and_forth' style but introduces QD guidance:
      - Uniformly sample BD cells from global archive (best-per-cell entries)
      - For each parent traj in train_batch, take 'back' steps, then 'forth' with GA-style guidance
      - Selection: keep better (by env reward), exactly like LS
      - Archive gets updated inside this routine (parents and offspring)
    """
    if archive is None or archive.coverage() == 0:
        # nothing to guide with; behave as no-op like LS early-exit
        visited_x = [tuple(tup[1][-1].tolist()) for tup in train_batch]
        return train_batch, visited_x, 0

    policy = cast(nn.Module, agent.model)

    n_new_rewards = 0  # we’ll count how many new trajectories reached terminal reward

    for _ in range(iterations):
        # draw elites uniformly from occupied cells (k defaults to batch size)
        k = len(train_batch) if qd_cells_per_iter <= 0 else qd_cells_per_iter
        elites = archive.sample_best_uniform(k)

        new_train_batch: List[Tuple] = []
        for i, (traj_obs, traj_s, traj_a, reward_old) in enumerate(train_batch):
            # 1) split parent at a 'mid' point (same style as LS)
            n_back_steps = int(traj_s.shape[0] * ls_back_ratio)
            if n_back_steps == 0:
                new_train_batch.append((traj_obs, traj_s, traj_a, reward_old))
                continue

            # we reconstruct "source_to_mid" deterministically from terminal
            back_obs, back_s, back_a = agent.envs[0].generate_backward(traj_s[-1].cpu().numpy())
            source_to_mid = [torch.from_numpy(arr[:-n_back_steps]) for arr in (back_obs, back_s, back_a)]
            obs_last, s_last = source_to_mid[0][-1], source_to_mid[1][-1]

            # pick an elite (uniform cell, best entry)
            elite = elites[i % len(elites)]
            elite_state = elite["state"]  # np.ndarray terminal
            # precompute per-dim “direction” from current mid to elite (signs in {0, +1})
            target_dir = torch.from_numpy(np.sign(elite_state - s_last.numpy())).to(s_last.dtype)
            target_dir = torch.clamp(target_dir, 0, 1)  # only forward increments allowed

            # 2) GA-guided forward rollout from mid to terminal
            mid_to_target_obs, mid_to_target_s, mid_to_target_a = [], [], []
            done = False
            while not done:
                with torch.no_grad():
                    pred = cast(torch.Tensor, policy(obs_last.unsqueeze(0)))
                    f_probs = (pred[..., : agent.ndim + 1]).softmax(1)

                    # crossover: bias toward elite direction with prob crossover_prob
                    if torch.rand(1).item() < crossover_prob:
                        # upweight dims where target_dir == 1 (move that coordinate)
                        bias = torch.ones_like(f_probs)
                        # index dims 0..ndim-1 are moves; last index is STOP
                        for d in range(agent.ndim):
                            if target_dir[d] == 1:
                                bias[0, d] = 1.0 + 1.0   # simple 2x boost toward elite moves
                        # (keep STOP unchanged)
                        f_probs = f_probs * bias
                        f_probs = f_probs / f_probs.sum(dim=1, keepdim=True)

                    # epsilon-noisy like LS
                    eps = 0.0 if not eps_noisy else agent.eps
                    f_probs = (1 - eps) * f_probs + eps / (agent.ndim + 1)

                    action = f_probs.multinomial(1)  # [1,1]

                    # mutation: with prob, flip action to a random move (or STOP less likely)
                    if torch.rand(1).item() < mutation_prob:
                        a_mut = np.random.randint(0, agent.ndim + 1)
                        action[0, 0] = int(a_mut)

                new_obs, new_r, done, new_s = agent.envs[i].step(int(action[0]), s_last.numpy().copy())
                # (env.step yields terminal reward at done=True)
                obs_last = agent.to_ft(new_obs)
                s_last  = agent.to_lt(new_s)

                mid_to_target_obs.append(obs_last)
                mid_to_target_s.append(s_last)
                mid_to_target_a.append(action[0])

            # 3) stitch trajectory (same tensors as LS)
            traj_obs_new = torch.cat([source_to_mid[0], torch.stack(mid_to_target_obs)], dim=0)
            traj_s_new   = torch.cat([source_to_mid[1], torch.stack(mid_to_target_s)], dim=0)
            traj_a_new   = torch.cat([source_to_mid[2], torch.stack(mid_to_target_a)], dim=0)
            reward_new   = new_r  # scalar/0D tensor/float depending on env

            # count one terminal reward consumption
            n_new_rewards += 1

            # 4) selection (elitist by env reward, like LS)
            better = reward_new > reward_old
            kept_obs = traj_obs_new if better else traj_obs
            kept_s   = traj_s_new   if better else traj_s
            kept_a   = traj_a_new   if better else traj_a
            kept_r   = reward_new   if better else reward_old
            new_train_batch.append((kept_obs, kept_s, kept_a, kept_r))

            # 5) QD archive update (parents + offspring both eligible)
            # parent
            parent_terminal = traj_s[-1].cpu().numpy()
            parent_reward = float(reward_old if not isinstance(reward_old, torch.Tensor) else reward_old.item())
            archive.add(parent_terminal, parent_reward, agent.horizon)
            # offspring
            child_terminal = traj_s_new[-1].cpu().numpy()
            child_reward = float(reward_new if not isinstance(reward_new, torch.Tensor) else reward_new.item())
            archive.add(child_terminal, child_reward, agent.horizon)

        train_batch = new_train_batch

    visited_x = [tuple(tup[1][-1].tolist()) for tup in train_batch]
    return train_batch, visited_x, n_new_rewards


def qd_back_and_forth_refined_from_start(
    agent,                                              # BaseAgent (TB) with .model, .envs, .ndim, .eps, .horizon
    train_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    archive,                                            # QDArchive with coverage(), sample_best_uniform(), add()
    iterations: int = 4,                                # outer refinement loops
    qd_cells_per_iter: int = 0,                         # elites to sample per iter (0 => auto = batch size)
    crossover_prob: float = 0.7,                        # prob to take elite action (vs parent/policy)
    mutation_prob: float = 0.15,                        # prob to use random action
    eps_noisy: bool = False,                            # epsilon-noise on policy sampling
) -> Tuple[List[Tuple], List[Tuple], int]:
    # Early exit if archive has no coverage
    if archive is None or archive.coverage() == 0:
        visited_x = [tuple(tup[1][-1].tolist()) for tup in train_batch]
        return train_batch, visited_x, 0

    policy = cast(nn.Module, agent.model)
    # safe device resolution (works even if model has no params yet)
    try:
        device = next(policy.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    n_new_rewards = 0

    # --- helpers to normalize action shapes/devices ---
    def _as_actions_col(x) -> torch.Tensor:
        """(T,1) Long on model device."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.detach()
        if x.ndim == 1:
            x = x.view(-1, 1)
        elif x.ndim == 2 and x.size(-1) == 1:
            pass
        else:
            x = x.view(-1, 1)
        return x.to(device=device, dtype=torch.long)

    def _as_actions_1d(x) -> torch.Tensor:
        """(T,) Long on model device."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.detach().cpu()
        if x.ndim == 2 and x.size(-1) == 1:
            x = x.squeeze(-1)
        elif x.ndim != 1:
            x = x.view(-1)
        return x.to(device=device, dtype=torch.long)

    def _action0d(a) -> torch.Tensor:
        """0-D Long on model device."""
        if isinstance(a, torch.Tensor):
            a = int(a.item())
        return torch.tensor(a, dtype=torch.long, device=device)

    for _ in range(iterations):
        # Draw elites uniformly from occupied cells (k defaults to batch size)
        k = len(train_batch) if qd_cells_per_iter <= 0 else qd_cells_per_iter
        elites = archive.sample_best_uniform(k)

        # Precompute elite trajectories (actions) from START → elite
        elite_trajectories = []
        for elite in elites:
            elite_state = elite["state"]  # np.ndarray terminal
            e_obs, e_s, e_a = agent.envs[0].generate_backward(elite_state)
            elite_trajectories.append({
                "obs": torch.from_numpy(e_obs),
                "states": torch.from_numpy(e_s),
                "actions": torch.from_numpy(e_a),  # may be (T,) or (T,1)
                "reward": elite["reward"],
            })

        new_train_batch: List[Tuple] = []

        for i, (traj_obs, traj_s, traj_a, reward_old) in enumerate(train_batch):
            # Pick elite (cyclic)
            elite_traj = elite_trajectories[i % len(elite_trajectories)]
            elite_actions_1d = _as_actions_1d(elite_traj["actions"])

            # Parent actions for crossover source
            parent_actions_1d = _as_actions_1d(traj_a)
            parent_len = parent_actions_1d.numel()

            # Deterministic START via parent's terminal
            parent_terminal = traj_s[-1].cpu().numpy()
            back_obs, back_s, _ = agent.envs[0].generate_backward(parent_terminal)

            # Initial (pre-action) state at t=0
            obs_last = agent.to_ft(back_obs[0]).to(device)
            s_last   = agent.to_lt(back_s[0]).to(device)

            # --- Child rollout FROM START ---
            # IMPORTANT: push initial obs/state BEFORE stepping so that len(obs)=len(actions)+1
            child_obs: List[torch.Tensor] = [obs_last]
            child_s:   List[torch.Tensor] = [s_last]
            child_a:   List[torch.Tensor] = []

            done = False
            step_count = 0
            new_r = torch.tensor(0.0, device=device)  # default if no terminal is reached

            while not done and step_count < agent.horizon:
                has_parent_action = step_count < parent_len
                elite_has = elite_actions_1d.numel() > 0

                rv = torch.rand(1, device=device).item()
                if rv < mutation_prob:
                    action = _action0d(np.random.randint(0, agent.ndim + 1))
                elif rv < mutation_prob + crossover_prob * (1 - mutation_prob) and elite_has:
                    action = _action0d(elite_actions_1d[step_count % elite_actions_1d.numel()])
                else:
                    if has_parent_action:
                        action = _action0d(parent_actions_1d[step_count])
                    else:
                        with torch.no_grad():
                            pred = cast(torch.Tensor, policy(obs_last.unsqueeze(0)))
                            f_probs = (pred[..., : agent.ndim + 1]).softmax(1)
                            eps = 0.0 if not eps_noisy else agent.eps
                            f_probs = (1 - eps) * f_probs + eps / (agent.ndim + 1)
                            action = _action0d(f_probs.multinomial(1).item())

                # env step (NB: step() expects numpy state copy)
                new_obs, new_r, done, new_s = agent.envs[i].step(
                    int(action.item()),
                    s_last.detach().cpu().numpy().copy(),
                )

                # Convert next obs/state and append
                obs_last = agent.to_ft(new_obs).to(device)
                s_last   = agent.to_lt(new_s).to(device)

                child_obs.append(obs_last)         # now len(obs) = steps + 1
                child_s.append(s_last)
                child_a.append(action.view(1))     # keep (1,) so stack→(T-1,1)

                step_count += 1

            # Assemble child's trajectory tensors
            if len(child_obs) > 1:  # took at least one step
                traj_obs_new = torch.stack(child_obs, dim=0)           # (T,   obs_dim)
                traj_s_new   = torch.stack(child_s,   dim=0)           # (T,   state_dim)
                traj_a_new   = _as_actions_col(torch.stack(child_a))   # (T-1, 1)
                reward_new   = new_r
            else:
                # Fallback (no step happened)
                traj_obs_new = traj_obs.clone()
                traj_s_new   = traj_s.clone()
                traj_a_new   = _as_actions_col(traj_a)                 # ensure (T-1,1)
                reward_new   = reward_old

            n_new_rewards += 1

            # Elitist selection by reward
            r_old_f = float(reward_old if not isinstance(reward_old, torch.Tensor) else reward_old.item())
            r_new_f = float(reward_new if not isinstance(reward_new, torch.Tensor) else reward_new.item())
            better = r_new_f > r_old_f

            kept_obs = traj_obs_new if better else traj_obs
            kept_s   = traj_s_new   if better else traj_s
            kept_a   = traj_a_new   if better else _as_actions_col(traj_a)  # normalize parent to (T-1,1)
            kept_r   = reward_new   if better else reward_old
            new_train_batch.append((kept_obs, kept_s, kept_a, kept_r))

            # Archive updates
            archive.add(parent_terminal, r_old_f, agent.horizon)

            child_terminal = kept_s[-1].detach().cpu().numpy()
            archive.add(child_terminal, float(kept_r if not isinstance(kept_r, torch.Tensor) else kept_r.item()), agent.horizon)

        # Next iteration uses the newly selected batch
        train_batch = new_train_batch

    visited_x = [tuple(tup[1][-1].tolist()) for tup in train_batch]
    return train_batch, visited_x, n_new_rewards
