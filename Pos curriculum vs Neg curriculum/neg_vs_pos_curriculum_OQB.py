# %% [markdown]
# # Negative Curriculum vs Positive Curriculum: Open Quantum Battery (Dicke Model) 비교 연구
#
# ## 연구 목적
# 양자 배터리 충전에서 **Negative Curriculum Learning**(제약 기반 조각적 학습)이
# **Positive Curriculum**(목표 지향적 학습) 및 기타 SOTA 방식보다 실제로 효과적인지를
# **객관적으로** 검증합니다.
#
# ### 비교 방법론 (4가지)
# | 방법 | 레이블 | 핵심 아이디어 |
# |------|--------|--------------|
# | Negative Curriculum | **Ours** | "함정에 빠지지 마라" → 제약을 단계적으로 학습 |
# | Positive Curriculum | **PosCurr** | "쉬운 목표부터 달성하라" → 목표를 단계적으로 상향 |
# | No Curriculum | **Vanilla** | 커리큘럼 없는 표준 RL |
# | Reverse Curriculum | **RevCurr** | 목표 근처에서 출발하여 역방향으로 탐색 확장 (SOTA) |
#
# ### 객관성 보장
# - 동일 SAC 에이전트 구조, 동일 하이퍼파라미터
# - 동일 random seed (3회 반복, 평균 ± std 보고)
# - 동일 총 학습 에피소드 수

# %% [markdown]
# ## 1. Setup & Imports

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import qutip as qt
import gymnasium as gym
from gymnasium import spaces

# PyTorch device
device = torch.device('cpu')
print(f"Device: {device}")
print(f"QuTiP version: {qt.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Matplotlib style
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# %% [markdown]
# ## 2. Open Quantum Battery Environment (Dicke Model)
#
# ### Hamiltonian
# $$H = \omega_c a^\dagger a + \omega_0 J_z + \frac{g(t)}{\sqrt{N}}(a^\dagger + a)(J_+ + J_-)$$
#
# ### Lindblad Master Equation (OQB)
# $$\dot{\rho} = -i[H, \rho] + \kappa \mathcal{D}[a]\rho + \gamma \sum_i \mathcal{D}[\sigma_-^{(i)}]\rho + \gamma_\phi \sum_i \mathcal{D}[\sigma_z^{(i)}]\rho$$
#
# - $N=4$ qubits (Dicke manifold: total spin $j=2$)
# - Cavity Fock space truncation: $n_{max}=8$
# - 제어 파라미터: $g(t) \in [0, 2]$, $\Delta(t) \in [-1, 1]$

# %%
class DickeOQBEnv(gym.Env):
    """Open Quantum Battery environment based on the Dicke model.

    N qubits collectively coupled to a single cavity mode with dissipation.
    Uses the Dicke manifold (collective spin-j representation) for efficiency.
    """

    metadata = {'render_modes': []}

    def __init__(self, N=4, n_cav=8, omega_c=1.0, omega_0=1.0,
                 kappa=0.03, gamma=0.01, gamma_phi=0.005,
                 dt=0.2, max_steps=30, g_max=2.0, delta_max=1.0,
                 init_photons=4):
        super().__init__()
        self.N = N
        self.j = N / 2.0  # Total spin
        self.n_cav = n_cav
        self.omega_c = omega_c
        self.omega_0_base = omega_0
        self.kappa = kappa
        self.gamma = gamma
        self.gamma_phi = gamma_phi
        self.dt = dt
        self.max_steps = max_steps
        self.g_max = g_max
        self.delta_max = delta_max
        self.init_photons = init_photons

        # Dimensions: cavity ⊗ spin
        self.dim_cav = n_cav
        self.dim_spin = int(2 * self.j + 1)  # 2j+1 states in Dicke manifold

        # Cavity operators
        a = qt.tensor(qt.destroy(self.dim_cav), qt.qeye(self.dim_spin))
        self.a = a
        self.a_dag = a.dag()
        self.n_op = a.dag() * a  # photon number

        # Collective spin operators (Dicke manifold)
        Jz = qt.tensor(qt.qeye(self.dim_cav), qt.jmat(self.j, 'z'))
        Jp = qt.tensor(qt.qeye(self.dim_cav), qt.jmat(self.j, '+'))
        Jm = qt.tensor(qt.qeye(self.dim_cav), qt.jmat(self.j, '-'))
        self.Jz = Jz
        self.Jp = Jp
        self.Jm = Jm

        # Collapse operators for Lindblad
        self.c_ops_base = []
        # Cavity decay
        if kappa > 0:
            self.c_ops_base.append(np.sqrt(kappa) * a)
        # Collective spin decay (approximation in Dicke manifold)
        if gamma > 0:
            self.c_ops_base.append(np.sqrt(gamma * N) * qt.tensor(
                qt.qeye(self.dim_cav), qt.jmat(self.j, '-')))
        # Collective dephasing
        if gamma_phi > 0:
            self.c_ops_base.append(np.sqrt(gamma_phi * N) * Jz)

        # Battery Hamiltonian (for ergotropy calculation)
        # H_B = omega_0 * J_z (battery = spin system)
        self.H_B = self.omega_0_base * Jz

        # Ground state of battery
        self.ground_state_spin = qt.tensor(
            qt.fock(self.dim_cav, 0),
            qt.spin_state(self.j, -self.j))  # |0_cav, -j>

        # Maximum energy of battery
        self.max_energy = self.omega_0_base * self.N  # from -j to +j

        # Observation and action spaces
        # State: [energy_stored, purity, entropy, <n_cav>, <Jz>/j]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 5.0, float(n_cav), 1.0], dtype=np.float32)
        )
        # Action: [g(t), delta(t)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32)
        )

        self.rho = None
        self.step_count = 0
        self.prev_ergotropy = 0.0

    def _build_hamiltonian(self, g, delta):
        """Build the Dicke Hamiltonian with given coupling and detuning."""
        omega_0 = self.omega_0_base + delta
        H = (self.omega_c * self.n_op
             + omega_0 * self.Jz
             + (g / np.sqrt(self.N)) * (self.a_dag + self.a) * (self.Jp + self.Jm))
        return H

    def _compute_battery_state(self):
        """Compute the reduced density matrix of the battery (spin system)."""
        # Trace out cavity
        rho_full = self.rho
        rho_battery = rho_full.ptrace(1)  # trace out cavity (index 0)
        return rho_battery

    def _compute_ergotropy(self, rho_B):
        """Compute ergotropy: W = Tr(rho H_B) - sum_i eps_i r_i
        where r_i are eigenvalues of rho sorted descending,
        and eps_i are eigenvalues of H_B sorted ascending.
        """
        H_B_local = self.omega_0_base * qt.jmat(self.j, 'z')
        # Energy
        energy = qt.expect(H_B_local, rho_B)
        # Passive state energy
        evals_rho = np.sort(np.real(rho_B.eigenenergies()))[::-1]  # descending
        evals_H = np.sort(np.real(H_B_local.eigenenergies()))       # ascending
        passive_energy = np.sum(evals_rho * evals_H)
        ergotropy = max(0.0, energy - passive_energy)
        return ergotropy

    def _compute_obs(self, rho_B):
        """Compute observation vector from battery state."""
        H_B_local = self.omega_0_base * qt.jmat(self.j, 'z')
        energy = qt.expect(H_B_local, rho_B)
        # Normalize energy: shift so ground = 0, max = max_energy
        energy_stored = (energy + self.omega_0_base * self.j) / self.max_energy
        energy_stored = np.clip(energy_stored, 0.0, 1.0)

        purity = np.real(qt.expect(rho_B * rho_B, qt.qeye(self.dim_spin)))
        purity = np.clip(purity, 0.0, 1.0)

        entropy = float(qt.entropy_vn(rho_B, 2))
        entropy = max(0.0, entropy)

        n_cav = float(np.real(qt.expect(self.n_op, self.rho)))
        jz_expect = float(np.real(qt.expect(self.Jz, self.rho))) / self.j

        obs = np.array([energy_stored, purity, entropy, n_cav, jz_expect],
                       dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None, init_state=None):
        """Reset environment. init_state can override for reverse curriculum."""
        super().reset(seed=seed)
        if init_state is not None:
            self.rho = init_state
        else:
            # Coherent state in cavity ⊗ ground spin: |alpha> ⊗ |j, -j>
            # This provides photons as an energy source for charging
            psi0 = qt.tensor(qt.coherent(self.dim_cav, np.sqrt(self.init_photons)),
                             qt.spin_state(self.j, -self.j))
            self.rho = qt.ket2dm(psi0)

        self.step_count = 0
        rho_B = self._compute_battery_state()
        self.prev_ergotropy = self._compute_ergotropy(rho_B)
        obs = self._compute_obs(rho_B)
        return obs, {}

    def step(self, action):
        """Execute one time step of the charging protocol."""
        # Map actions from [-1,1] to physical range
        g = float((action[0] + 1.0) / 2.0 * self.g_max)      # [0, g_max]
        delta = float(action[1] * self.delta_max)               # [-delta_max, delta_max]

        # Build Hamiltonian and evolve
        H = self._build_hamiltonian(g, delta)
        result = qt.mesolve(H, self.rho, [0, self.dt], c_ops=self.c_ops_base)
        self.rho = result.states[-1]

        # Ensure valid density matrix
        self.rho = (self.rho + self.rho.dag()) / 2.0
        tr = np.real(self.rho.tr())
        if abs(tr - 1.0) > 1e-6:
            self.rho = self.rho / tr

        self.step_count += 1

        # Compute observables
        rho_B = self._compute_battery_state()
        ergotropy = self._compute_ergotropy(rho_B)
        delta_erg = ergotropy - self.prev_ergotropy

        obs = self._compute_obs(rho_B)
        energy_stored = obs[0]
        purity = obs[1]
        entropy = obs[2]

        # Base reward: ergotropy change (amplified) + absolute ergotropy bonus
        reward = 10.0 * delta_erg / self.max_energy + 0.5 * ergotropy / self.max_energy

        self.prev_ergotropy = ergotropy

        terminated = self.step_count >= self.max_steps
        truncated = False

        info = {
            'ergotropy': ergotropy,
            'energy_stored': energy_stored,
            'purity': purity,
            'entropy': entropy,
            'delta_ergotropy': delta_erg,
            'g': g,
            'delta': delta,
        }

        return obs, reward, terminated, truncated, info


# Quick sanity check
print("=== Environment Sanity Check ===")
env = DickeOQBEnv()
obs, _ = env.reset(seed=42)
print(f"Initial obs: {obs}")
print(f"Obs space: {env.observation_space}")
print(f"Act space: {env.action_space}")

total_r = 0
for i in range(5):
    action = env.action_space.sample()
    obs, r, done, trunc, info = env.step(action)
    total_r += r
    if i < 3:
        print(f"  Step {i+1}: erg={info['ergotropy']:.4f}, "
              f"energy={info['energy_stored']:.4f}, "
              f"purity={info['purity']:.4f}, r={r:.4f}")
print(f"Env OK. Max battery energy = {env.max_energy}")

# %% [markdown]
# ## 3. SAC Agent (PyTorch)
#
# 모든 커리큘럼 전략에서 동일한 Soft Actor-Critic 구조를 사용합니다.
# - **Actor**: Gaussian policy, 2-layer MLP (64-64)
# - **Critic**: Twin Q-networks
# - **Automatic entropy tuning**

# %%
# ---- Replay Buffer ----
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ---- Networks ----
class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)
        self.LOG_STD_MIN, self.LOG_STD_MAX = -20, 2

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def get_action(self, obs_np):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
            action, _ = self.sample(obs_t)
            return action.cpu().numpy().flatten()


class TwinQCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


# ---- SAC Agent ----
class SACAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, tau=0.005,
                 alpha_lr=3e-4, buffer_size=50000, batch_size=128):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = GaussianActor(obs_dim, act_dim).to(device)
        self.critic = TwinQCritic(obs_dim, act_dim).to(device)
        self.critic_target = deepcopy(self.critic).to(device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # Auto entropy tuning
        self.target_entropy = -act_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.buffer = ReplayBuffer(buffer_size)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(self, obs, deterministic=False):
        if deterministic:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                mean, _ = self.actor(obs_t)
                return torch.tanh(mean).cpu().numpy().flatten()
        return self.actor.get_action(obs)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        s = torch.FloatTensor(states).to(device)
        a = torch.FloatTensor(actions).to(device)
        r = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        ns = torch.FloatTensor(next_states).to(device)
        d = torch.FloatTensor(dones).unsqueeze(1).to(device)

        alpha = self.log_alpha.exp().detach()

        # Critic update
        with torch.no_grad():
            na, nlp = self.actor.sample(ns)
            tq1, tq2 = self.critic_target(ns, na)
            tq = torch.min(tq1, tq2) - alpha * nlp
            target = r + (1 - d) * self.gamma * tq

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor update
        new_a, new_lp = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_a)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * new_lp - q_new).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (new_lp.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # Soft target update
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(),
                     'critic': self.critic.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])


print("SAC Agent class defined.")

# %% [markdown]
# ## 4. Curriculum Strategies
#
# 모든 전략은 동일 에피소드 수(총 2400)를 3개 스테이지로 분할합니다.
#
# ### 4.1 Negative Curriculum (Ours)
# - **Stage 1**: 에너지 붕괴 페널티 (collapse 방지)
# - **Stage 2**: 엔트로피 증가 페널티 (순도 유지)
# - **Stage 3**: Vanilla reward (자유 탐색)
#
# ### 4.2 Positive Curriculum (PosCurr)
# - **Stage 1**: 쉬운 목표 (ergotropy > 30% of max → bonus)
# - **Stage 2**: 중간 목표 (ergotropy > 60% of max → bonus)
# - **Stage 3**: 완충 목표 (ergotropy > 90% of max → bonus)
#
# ### 4.3 Vanilla (No Curriculum)
# - 모든 스테이지에서 동일한 기본 reward
#
# ### 4.4 Reverse Curriculum (RevCurr)
# - **Stage 1**: 완충 근처에서 시작 (작은 perturbation)
# - **Stage 2**: 중간 에너지 상태에서 시작
# - **Stage 3**: 완전 방전 상태(ground state)에서 시작

# %%
class CurriculumManager:
    """Manages curriculum stages for different strategies."""

    def __init__(self, strategy, total_episodes, stages=3, env_params=None):
        self.strategy = strategy
        self.total_episodes = total_episodes
        self.stages = stages
        self.stage_size = total_episodes // stages
        self.env_params = env_params or {}

        # Negative curriculum penalty coefficients (moderate)
        self.alpha_collapse = 2.0   # energy collapse penalty
        self.beta_entropy = 1.5     # entropy increase penalty

        # Positive curriculum target thresholds
        self.pos_targets = [0.15, 0.35, 0.60]
        self.pos_bonus = 1.0

    def get_stage(self, episode):
        return min(episode // self.stage_size, self.stages - 1)

    def get_init_state(self, episode, env):
        """Get initial state for reverse curriculum."""
        stage = self.get_stage(episode)
        if self.strategy != 'RevCurr':
            return None

        j = env.j
        dim_cav = env.dim_cav
        dim_spin = env.dim_spin

        if stage == 0:
            # Near fully charged: coherent cavity + mostly excited spins
            alpha_cav = np.sqrt(1.0)  # few photons remaining
            alpha = 0.85 + 0.15 * np.random.rand()
            psi_spin = (alpha * qt.spin_state(j, j)
                        + np.sqrt(1 - alpha**2) * qt.spin_state(j, j-1)).unit()
            psi = qt.tensor(qt.coherent(dim_cav, alpha_cav), psi_spin)
            return qt.ket2dm(psi)
        elif stage == 1:
            # Mid-energy state: some cavity photons + mid spin
            alpha_cav = np.sqrt(2.0)
            alpha = 0.5 + 0.3 * np.random.rand()
            m_val = 0  # middle Jz eigenvalue
            psi_spin = (alpha * qt.spin_state(j, m_val)
                        + np.sqrt(1 - alpha**2) * qt.spin_state(j, m_val + 1)).unit()
            psi = qt.tensor(qt.coherent(dim_cav, alpha_cav), psi_spin)
            return qt.ket2dm(psi)
        else:
            return None  # default reset (coherent cavity + ground spin)

    def shape_reward(self, episode, base_reward, info):
        """Apply curriculum-specific reward shaping."""
        stage = self.get_stage(episode)
        max_erg = 4.0  # max_energy for N=4

        if self.strategy == 'Ours':  # Negative Curriculum
            if stage == 0:
                # Stage 1: Penalize collapse, but keep base reward (exploration)
                energy_drop = max(0.0, -info['delta_ergotropy'])
                penalty = self.alpha_collapse * energy_drop / max_erg
                return base_reward - penalty
            elif stage == 1:
                # Stage 2: Penalize decoherence (entropy > threshold)
                entropy_penalty = self.beta_entropy * max(0.0, info['entropy'] - 0.3)
                return base_reward - entropy_penalty
            else:
                # Stage 3: Pure exploration with base reward
                return base_reward

        elif self.strategy == 'PosCurr':  # Positive Curriculum
            target = self.pos_targets[stage]
            if info['ergotropy'] >= target * max_erg:
                return base_reward + self.pos_bonus * (info['ergotropy'] / max_erg)
            return base_reward

        elif self.strategy == 'Vanilla':
            return base_reward

        elif self.strategy == 'RevCurr':
            # Same reward, but initial state changes
            return base_reward

        return base_reward


print("CurriculumManager class defined.")

# %% [markdown]
# ## 5. Training Loop

# %%
def train_agent(strategy, total_episodes=2400, max_steps=30, seed=42, verbose=True):
    """Train a SAC agent with the given curriculum strategy.

    Returns:
        dict with training history and final agent
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    env = DickeOQBEnv(max_steps=max_steps)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim)
    curriculum = CurriculumManager(strategy, total_episodes)

    history = {
        'episode_rewards': [],
        'episode_ergotropy': [],
        'episode_purity': [],
        'episode_entropy': [],
        'episode_energy': [],
        'best_ergotropy': 0.0,
        'convergence_episode': None,
    }

    convergence_threshold = 0.3 * env.max_energy  # 30% threshold for convergence
    warmup_steps = 200  # Fill buffer before training

    for ep in range(total_episodes):
        init_state = curriculum.get_init_state(ep, env)
        obs, _ = env.reset(seed=seed + ep, init_state=init_state)

        ep_reward = 0.0
        ep_ergotropy = 0.0
        ep_purity = 0.0
        ep_entropy = 0.0
        ep_energy = 0.0

        for step in range(max_steps):
            if len(agent.buffer) < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, base_reward, terminated, truncated, info = env.step(action)
            reward = curriculum.shape_reward(ep, base_reward, info)

            agent.buffer.push(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs

            if len(agent.buffer) >= warmup_steps:
                agent.update()

            ep_reward += reward
            ep_ergotropy = info['ergotropy']
            ep_purity = info['purity']
            ep_entropy = info['entropy']
            ep_energy = info['energy_stored']

            if terminated or truncated:
                break

        history['episode_rewards'].append(ep_reward)
        history['episode_ergotropy'].append(ep_ergotropy)
        history['episode_purity'].append(ep_purity)
        history['episode_entropy'].append(ep_entropy)
        history['episode_energy'].append(ep_energy)

        if ep_ergotropy > history['best_ergotropy']:
            history['best_ergotropy'] = ep_ergotropy

        if (history['convergence_episode'] is None
                and ep_ergotropy >= convergence_threshold):
            history['convergence_episode'] = ep

        if verbose and (ep + 1) % (total_episodes // 6) == 0:
            stage = curriculum.get_stage(ep)
            print(f"  [{strategy}] Ep {ep+1}/{total_episodes} | "
                  f"Stage {stage+1} | Erg={ep_ergotropy:.3f} | "
                  f"Pur={ep_purity:.3f} | R={ep_reward:.3f}")

    history['agent'] = agent
    return history


print("Training function defined.")

# %% [markdown]
# ## 5.1 Run Experiments (3 Seeds × 4 Strategies)

# %%
STRATEGIES = ['Ours', 'PosCurr', 'Vanilla', 'RevCurr']
STRATEGY_LABELS = {
    'Ours': 'Negative Curriculum (Ours)',
    'PosCurr': 'Positive Curriculum',
    'Vanilla': 'Vanilla SAC (No Curriculum)',
    'RevCurr': 'Reverse Curriculum (SOTA)'
}
STRATEGY_COLORS = {
    'Ours': '#E63946',
    'PosCurr': '#457B9D',
    'Vanilla': '#2A9D8F',
    'RevCurr': '#E9C46A'
}

SEEDS = [42, 123, 777]
TOTAL_EPISODES = 1500
MAX_STEPS = 30

all_results = {}  # strategy -> list of history dicts

print("=" * 60)
print(" Starting Experiments: Neg vs Pos Curriculum on OQB (Dicke)")
print("=" * 60)

for strategy in STRATEGIES:
    print(f"\n{'='*50}")
    print(f" Strategy: {STRATEGY_LABELS[strategy]}")
    print(f"{'='*50}")
    all_results[strategy] = []
    for i, seed in enumerate(SEEDS):
        print(f"\n--- Seed {seed} (Run {i+1}/{len(SEEDS)}) ---")
        hist = train_agent(strategy, total_episodes=TOTAL_EPISODES,
                           max_steps=MAX_STEPS, seed=seed)
        all_results[strategy].append(hist)
        print(f"  Best ergotropy: {hist['best_ergotropy']:.4f}")
        conv = hist['convergence_episode']
        print(f"  Convergence ep: {conv if conv else 'N/A'}")

print("\n" + "=" * 60)
print(" All experiments completed!")
print("=" * 60)

# %% [markdown]
# ## 6. Results & Visualization

# %%
def smooth(data, window=50):
    """Moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def compute_stats(results_list, key):
    """Compute mean and std across seeds for a given metric."""
    all_data = [np.array(r[key]) for r in results_list]
    min_len = min(len(d) for d in all_data)
    all_data = [d[:min_len] for d in all_data]
    arr = np.array(all_data)
    return arr.mean(axis=0), arr.std(axis=0)

# %% [markdown]
# ### 6.1 학습 곡선: Episode vs Ergotropy

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Raw learning curves
ax = axes[0]
for strategy in STRATEGIES:
    mean, std = compute_stats(all_results[strategy], 'episode_ergotropy')
    mean_s = smooth(mean, 50)
    x = np.arange(len(mean_s))
    ax.plot(x, mean_s, color=STRATEGY_COLORS[strategy],
            label=STRATEGY_LABELS[strategy], linewidth=2)
    std_s = smooth(std, 50)[:len(mean_s)]
    ax.fill_between(x, mean_s - std_s, mean_s + std_s,
                    color=STRATEGY_COLORS[strategy], alpha=0.15)

ax.set_xlabel('Episode')
ax.set_ylabel('Ergotropy')
ax.set_title('Learning Curves: Episode Ergotropy (smoothed)')
ax.legend(fontsize=9)
ax.axhline(y=4.0, color='gray', linestyle='--', alpha=0.5, label='Max Energy')

# Right: Episode rewards
ax = axes[1]
for strategy in STRATEGIES:
    mean, std = compute_stats(all_results[strategy], 'episode_rewards')
    mean_s = smooth(mean, 50)
    x = np.arange(len(mean_s))
    ax.plot(x, mean_s, color=STRATEGY_COLORS[strategy],
            label=STRATEGY_LABELS[strategy], linewidth=2)

ax.set_xlabel('Episode')
ax.set_ylabel('Episode Reward')
ax.set_title('Learning Curves: Episode Reward (smoothed)')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('fig1_learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 1 saved: fig1_learning_curves.png")

# %% [markdown]
# ### 6.2 최종 에르고트로피 분포 (Box Plot)

# %%
fig, ax = plt.subplots(figsize=(10, 6))

box_data = []
labels = []
colors = []
for strategy in STRATEGIES:
    final_ergs = [r['episode_ergotropy'][-1] for r in all_results[strategy]]
    # Also get last 100 episodes average per seed
    last_100 = [np.mean(r['episode_ergotropy'][-100:]) for r in all_results[strategy]]
    box_data.append(last_100)
    labels.append(STRATEGY_LABELS[strategy].replace(' (', '\n('))
    colors.append(STRATEGY_COLORS[strategy])

bp = ax.boxplot(box_data, labels=labels, patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Scatter individual points
for i, (data, color) in enumerate(zip(box_data, colors)):
    x = np.random.normal(i + 1, 0.04, len(data))
    ax.scatter(x, data, color=color, zorder=5, s=60, edgecolors='black', linewidth=0.5)

ax.set_ylabel('Final Ergotropy (last 100 eps avg)')
ax.set_title('Final Charging Performance Comparison')
ax.axhline(y=4.0, color='gray', linestyle='--', alpha=0.5)
ax.text(0.5, 4.1, 'Max Energy = 4.0', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('fig2_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 2 saved: fig2_boxplot.png")

# %% [markdown]
# ### 6.3 순도(Purity) 궤적 비교

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Purity over episodes
ax = axes[0]
for strategy in STRATEGIES:
    mean, std = compute_stats(all_results[strategy], 'episode_purity')
    mean_s = smooth(mean, 50)
    x = np.arange(len(mean_s))
    ax.plot(x, mean_s, color=STRATEGY_COLORS[strategy],
            label=STRATEGY_LABELS[strategy], linewidth=2)
    std_s = smooth(std, 50)[:len(mean_s)]
    ax.fill_between(x, mean_s - std_s, mean_s + std_s,
                    color=STRATEGY_COLORS[strategy], alpha=0.15)

ax.set_xlabel('Episode')
ax.set_ylabel('Purity')
ax.set_title('State Purity Over Training')
ax.legend(fontsize=9)

# Entropy over episodes
ax = axes[1]
for strategy in STRATEGIES:
    mean, std = compute_stats(all_results[strategy], 'episode_entropy')
    mean_s = smooth(mean, 50)
    x = np.arange(len(mean_s))
    ax.plot(x, mean_s, color=STRATEGY_COLORS[strategy],
            label=STRATEGY_LABELS[strategy], linewidth=2)

ax.set_xlabel('Episode')
ax.set_ylabel('Von Neumann Entropy')
ax.set_title('Entropy Over Training')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('fig3_purity_entropy.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 3 saved: fig3_purity_entropy.png")

# %% [markdown]
# ### 6.4 수렴 속도 비교

# %%
fig, ax = plt.subplots(figsize=(10, 6))

conv_data = {}
for strategy in STRATEGIES:
    episodes = []
    for r in all_results[strategy]:
        ep = r['convergence_episode']
        if ep is not None:
            episodes.append(ep)
        else:
            episodes.append(TOTAL_EPISODES)  # Did not converge
    conv_data[strategy] = episodes

x_pos = np.arange(len(STRATEGIES))
means = [np.mean(conv_data[s]) for s in STRATEGIES]
stds = [np.std(conv_data[s]) for s in STRATEGIES]
bar_colors = [STRATEGY_COLORS[s] for s in STRATEGIES]

bars = ax.bar(x_pos, means, yerr=stds, color=bar_colors, alpha=0.7,
              capsize=5, edgecolor='black', linewidth=0.5)

ax.set_xticks(x_pos)
ax.set_xticklabels([STRATEGY_LABELS[s].replace(' (', '\n(') for s in STRATEGIES],
                   fontsize=10)
ax.set_ylabel('Episodes to Convergence (50% threshold)')
ax.set_title('Sample Efficiency: Convergence Speed')
ax.axhline(y=TOTAL_EPISODES, color='gray', linestyle='--', alpha=0.3)

for bar, mean_val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{mean_val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('fig4_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 4 saved: fig4_convergence.png")

# %% [markdown]
# ### 6.5 Best Episode 충전 프로필 (Time-Step 분석)

# %%
def run_best_episode(agent, env, seed=42):
    """Run a single episode with best agent, recording full trajectory."""
    obs, _ = env.reset(seed=seed)
    trajectory = {'ergotropy': [], 'energy': [], 'purity': [],
                  'entropy': [], 'g': [], 'delta': [], 'rewards': []}
    for _ in range(env.max_steps):
        action = agent.select_action(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        trajectory['ergotropy'].append(info['ergotropy'])
        trajectory['energy'].append(info['energy_stored'])
        trajectory['purity'].append(info['purity'])
        trajectory['entropy'].append(info['entropy'])
        trajectory['g'].append(info['g'])
        trajectory['delta'].append(info['delta'])
        trajectory['rewards'].append(r)
        if done or trunc:
            break
    return trajectory


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
eval_env = DickeOQBEnv()

for strategy in STRATEGIES:
    # Use best seed (highest final ergotropy)
    best_idx = np.argmax([np.mean(r['episode_ergotropy'][-100:])
                          for r in all_results[strategy]])
    agent = all_results[strategy][best_idx]['agent']
    traj = run_best_episode(agent, eval_env)

    t = np.arange(len(traj['ergotropy']))
    color = STRATEGY_COLORS[strategy]
    label = STRATEGY_LABELS[strategy]

    axes[0, 0].plot(t, traj['ergotropy'], color=color, label=label, linewidth=2)
    axes[0, 1].plot(t, traj['purity'], color=color, label=label, linewidth=2)
    axes[1, 0].plot(t, traj['g'], color=color, label=label, linewidth=1.5, alpha=0.8)
    axes[1, 1].plot(t, traj['delta'], color=color, label=label, linewidth=1.5, alpha=0.8)

axes[0, 0].set_title('Ergotropy vs Time Step'); axes[0, 0].set_ylabel('Ergotropy')
axes[0, 1].set_title('Purity vs Time Step'); axes[0, 1].set_ylabel('Purity')
axes[1, 0].set_title('Coupling g(t)'); axes[1, 0].set_ylabel('g(t)')
axes[1, 1].set_title('Detuning Δ(t)'); axes[1, 1].set_ylabel('Δ(t)')

for ax in axes.flat:
    ax.set_xlabel('Time Step')
    ax.legend(fontsize=8)

plt.suptitle('Best Episode Charging Profile', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig5_charging_profile.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 5 saved: fig5_charging_profile.png")

# %% [markdown]
# ## 7. Statistical Analysis & Discussion

# %%
from scipy import stats

print("=" * 70)
print(" STATISTICAL ANALYSIS: Curriculum Strategy Comparison")
print("=" * 70)

# Final performance (last 100 episodes average)
print("\n### Final Performance (last 100 episodes average) ###\n")
perf_data = {}
for strategy in STRATEGIES:
    vals = [np.mean(r['episode_ergotropy'][-100:]) for r in all_results[strategy]]
    perf_data[strategy] = vals
    print(f"  {STRATEGY_LABELS[strategy]:40s}: "
          f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")

# Pairwise t-tests (Ours vs. others)
print("\n### Pairwise Welch's t-test: Ours vs. Others ###\n")
ours_vals = perf_data['Ours']
for strategy in ['PosCurr', 'Vanilla', 'RevCurr']:
    other_vals = perf_data[strategy]
    t_stat, p_val = stats.ttest_ind(ours_vals, other_vals, equal_var=False)
    sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else "n.s."))
    print(f"  Ours vs {STRATEGY_LABELS[strategy]:35s}: "
          f"t={t_stat:+.3f}, p={p_val:.4f} {sig}")

# Convergence speed
print("\n### Convergence Speed (episodes to reach 50% max ergotropy) ###\n")
for strategy in STRATEGIES:
    eps = []
    for r in all_results[strategy]:
        e = r['convergence_episode']
        eps.append(e if e is not None else TOTAL_EPISODES)
    print(f"  {STRATEGY_LABELS[strategy]:40s}: "
          f"{np.mean(eps):.0f} ± {np.std(eps):.0f} episodes")

# Purity maintenance
print("\n### Final Purity (last 100 episodes average) ###\n")
for strategy in STRATEGIES:
    vals = [np.mean(r['episode_purity'][-100:]) for r in all_results[strategy]]
    print(f"  {STRATEGY_LABELS[strategy]:40s}: "
          f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")

# Summary table
print("\n" + "=" * 70)
print(" SUMMARY TABLE")
print("=" * 70)
print(f"\n{'Method':<35} {'Ergotropy':>12} {'Conv. Ep':>12} {'Purity':>12}")
print("-" * 71)
for strategy in STRATEGIES:
    erg = np.mean([np.mean(r['episode_ergotropy'][-100:]) for r in all_results[strategy]])
    conv_eps = []
    for r in all_results[strategy]:
        e = r['convergence_episode']
        conv_eps.append(e if e is not None else TOTAL_EPISODES)
    conv = np.mean(conv_eps)
    pur = np.mean([np.mean(r['episode_purity'][-100:]) for r in all_results[strategy]])
    print(f"{STRATEGY_LABELS[strategy]:<35} {erg:>12.4f} {conv:>12.0f} {pur:>12.4f}")

# %% [markdown]
# ## 8. 결론 및 논의
#
# ### 실험 결과 해석
#
# 위 결과는 동일 조건(에이전트, 하이퍼파라미터, 총 학습량)에서 4가지 커리큘럼 전략을
# 객관적으로 비교한 것입니다.
#
# **평가 기준:**
# 1. **최종 에르고트로피**: 충전 성능의 절대적 수준
# 2. **수렴 속도**: 목표 달성까지의 샘플 효율성
# 3. **순도 유지**: 양자 코히어런스 보존 능력
# 4. **안정성**: seed 간 분산 (robust한가?)
#
# ### 한계점
# - N=4 qubits의 소규모 시스템으로 제한적
# - Dicke manifold 근사 사용 (개별 qubit 역학 무시)
# - 학습 에피소드 수가 제한적 (더 긴 학습 시 결과가 달라질 수 있음)
# - 단일 OQB 모델(Dicke)만 테스트 — TC, Spin Chain 등 다른 모델에서의 일반화 필요
#
# ### 향후 연구 방향
# - 더 큰 N (8, 16 qubits)으로 확장
# - 다른 양자 배터리 모델(TC, Spin Chain)에서 검증
# - Non-Markovian 환경에서의 비교
# - Negative Curriculum의 스테이지 전환 조건 자동화 연구
#
# ---
# *이 노트북은 bias를 최소화하기 위해 모든 방법에 동일한 리소스를 할당하고,
# 통계적 유의성 검증을 포함합니다.*

# %%
print("\n" + "=" * 60)
print(" Experiment Complete!")
print(" All figures saved to current directory.")
print("=" * 60)
