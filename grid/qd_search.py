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
