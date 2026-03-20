"""
QECCT (Quantum Error Correction Code Transformer) - Model Definitions
======================================================================
Paper: "Deep Quantum Error Correction" (Choukroun & Wolf, AAAI-24)
This module contains all model components for reproducing the QECCT architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================
# 1. Toric Code Utilities
# ============================================================

class ToricCode:
    """
    Toric code on an L×L lattice with periodic boundary conditions.
    Physical qubits sit on edges: n = 2L^2.
    Stabilizers: L^2 vertex (X-type) + L^2 plaquette (Z-type).
    """

    def __init__(self, L: int):
        self.L = L
        self.n = 2 * L * L          # number of physical qubits
        self.n_s = 2 * L * L        # number of syndrome bits
        self.k = 2                   # logical qubits

        self.H_x, self.H_z = self._build_parity_check()
        # Stacked parity check matrix H = [H_z; H_x], shape (2L^2, 2L^2)
        self.H = np.vstack([self.H_z, self.H_x]).astype(np.float32)

        self.L_matrix = self._build_logical_operators()
        self.mask = self._build_mask()

    def _build_parity_check(self):
        L = self.L
        n_qubits_per_type = L * L  # horizontal or vertical edges

        # Vertex operators (X-stabilizers)
        H_x = np.zeros((L * L, 2 * L * L), dtype=np.float32)
        for r in range(L):
            for c in range(L):
                v = r * L + c
                # Four edges adjacent to vertex (r, c):
                # horizontal edge to right: (r, c)
                h_right = r * L + c
                # horizontal edge to left: (r, (c-1)%L)
                h_left = r * L + (c - 1) % L
                # vertical edge below: (r, c)
                v_below = L * L + r * L + c
                # vertical edge above: ((r-1)%L, c)
                v_above = L * L + ((r - 1) % L) * L + c

                H_x[v, h_right] = 1
                H_x[v, h_left] = 1
                H_x[v, v_below] = 1
                H_x[v, v_above] = 1

        # Plaquette operators (Z-stabilizers)
        H_z = np.zeros((L * L, 2 * L * L), dtype=np.float32)
        for r in range(L):
            for c in range(L):
                p = r * L + c
                # Four edges bordering plaquette (r, c):
                # horizontal edge top: (r, c)
                h_top = r * L + c
                # horizontal edge bottom: ((r+1)%L, c)
                h_bottom = ((r + 1) % L) * L + c
                # vertical edge left: (r, c)
                v_left = L * L + r * L + c
                # vertical edge right: (r, (c+1)%L)
                v_right = L * L + r * L + (c + 1) % L

                H_z[p, h_top] = 1
                H_z[p, h_bottom] = 1
                H_z[p, v_left] = 1
                H_z[p, v_right] = 1

        return H_x, H_z

    def _build_logical_operators(self):
        """Build logical operator matrix L of shape (k, n) = (2, 2L^2)."""
        L = self.L
        logical = np.zeros((2, 2 * L * L), dtype=np.float32)
        # Logical X1: horizontal loop across first row
        for c in range(L):
            logical[0, c] = 1  # horizontal edges in row 0
        # Logical X2: vertical loop along first column
        for r in range(L):
            logical[1, L * L + r * L] = 1  # vertical edges in column 0
        return logical

    def _build_mask(self):
        """
        Build attention mask from parity-check matrix.
        Two syndrome/qubit elements are connected if they share a stabilizer.
        mask[i,j] = 1 if elements i,j share at least one stabilizer.
        Input dimension = n + n_s = 4L^2.
        """
        total = self.n + self.n_s  # 4L^2
        # Build adjacency from H: H is (n_s x n)
        # Augmented system: [qubits | syndromes]
        # qubit i and syndrome j are connected if H[j, i] = 1
        H_full = self.H  # (2L^2 x 2L^2)
        adj = np.zeros((total, total), dtype=np.float32)

        # qubit-syndrome connections
        for j in range(self.n_s):
            for i in range(self.n):
                if H_full[j, i] > 0:
                    adj[i, self.n + j] = 1
                    adj[self.n + j, i] = 1

        # qubit-qubit: connected if they share a stabilizer
        HtH = (H_full.T @ H_full > 0).astype(np.float32)
        adj[:self.n, :self.n] = HtH

        # syndrome-syndrome: connected if they share a qubit
        HHt = (H_full @ H_full.T > 0).astype(np.float32)
        adj[self.n:, self.n:] = HHt

        # Self-connections
        np.fill_diagonal(adj, 1)
        return adj

    def get_syndrome(self, noise: np.ndarray) -> np.ndarray:
        """Compute syndrome s = H @ noise (mod 2)."""
        return (self.H @ noise) % 2

    def sample_independent_noise(self, p: float, batch_size: int = 1) -> np.ndarray:
        """Independent noise: X and Z errors independently with probability p."""
        noise = (np.random.random((batch_size, self.n)) < p).astype(np.float32)
        return noise

    def sample_depolarization_noise(self, p: float, batch_size: int = 1) -> np.ndarray:
        """
        Depolarization noise: P(X)=P(Y)=P(Z)=p/3, P(I)=1-p.
        For the Toric code with separate X/Z stabilizers,
        we model X-errors and Z-errors on each qubit.
        n_physical = L^2 actual qubits, each can have X, Z, or Y=XZ error.
        But with the block structure: first L^2 = X-errors, second L^2 = Z-errors.
        """
        n_phys = self.L * self.L
        noise = np.zeros((batch_size, self.n), dtype=np.float32)
        rand = np.random.random((batch_size, n_phys))
        # X-only error: p/3
        x_error = (rand < p / 3).astype(np.float32)
        # Z-only error: p/3
        z_error = ((rand >= p / 3) & (rand < 2 * p / 3)).astype(np.float32)
        # Y=XZ error: p/3
        y_error = ((rand >= 2 * p / 3) & (rand < p)).astype(np.float32)

        noise[:, :n_phys] = x_error + y_error  # X part (mod 2)
        noise[:, n_phys:] = z_error + y_error   # Z part (mod 2)
        noise = noise % 2
        return noise

    def sample_noisy_syndrome(self, noise: np.ndarray, p_meas: float, T: int):
        """
        Generate T repeated noisy syndrome measurements.
        Returns syndromes of shape (batch, T, n_s).
        """
        batch_size = noise.shape[0]
        syndromes = np.zeros((batch_size, T, self.n_s), dtype=np.float32)
        cumulative_noise = np.zeros_like(noise)

        for t in range(T):
            # Accumulate system noise
            new_noise = (np.random.random((batch_size, self.n)) < 0.0).astype(np.float32)
            if t == 0:
                cumulative_noise = noise.copy()
            else:
                noise_t = self.sample_independent_noise(0.0, batch_size)  # additional noise
                cumulative_noise = (cumulative_noise + noise_t) % 2

            # Compute noiseless syndrome
            s_clean = np.array([self.get_syndrome(cumulative_noise[i]) for i in range(batch_size)])
            # Add measurement error
            meas_error = (np.random.random((batch_size, self.n_s)) < p_meas).astype(np.float32)
            syndromes[:, t, :] = (s_clean + meas_error) % 2

        return syndromes


# ============================================================
# 2. Initial Noise Estimator g_omega
# ============================================================

class NoiseEstimator(nn.Module):
    """
    Shallow network g_ω: {0,1}^n_s -> R^n
    Two FC layers with hidden dim = 5 * n_s, GELU activation.
    """

    def __init__(self, n_syndrome: int, n_physical: int):
        super().__init__()
        hidden = 5 * n_syndrome
        self.net = nn.Sequential(
            nn.Linear(n_syndrome, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_physical),
            nn.Sigmoid()
        )

    def forward(self, syndrome: torch.Tensor) -> torch.Tensor:
        return self.net(syndrome)


# ============================================================
# 3. Masked Multi-Head Self-Attention
# ============================================================

class MaskedMultiHeadAttention(nn.Module):
    """
    Self-attention with binary mask derived from parity-check matrix.
    A_H(Q,K,V) = Softmax(d^{-1/2} (QK^T + g(H))) V
    where g(H) applies -inf to unconnected pairs.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, seq_len) binary mask, 1 = attend, 0 = block
        """
        B, L, D = x.shape

        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention with mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask: where mask == 0, set to -inf
        mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # handle all-masked rows

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)


# ============================================================
# 4. Transformer Decoder Block
# ============================================================

class TransformerBlock(nn.Module):
    """Single Transformer decoder block with masked self-attention + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.0):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.attn = MaskedMultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        h = self.norm1(x)
        h = self.attn(h, mask)
        x = x + self.dropout(h)

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x


# ============================================================
# 5. QECCT Full Architecture
# ============================================================

class QECCT(nn.Module):
    """
    Quantum Error Correction Code Transformer.

    Architecture:
    1. Initial noise estimator g_ω(s) -> estimated noise
    2. Input: h_q(s) = [g_ω(s), s] concatenated, then one-hot embedded
    3. N Transformer blocks with masked self-attention
    4. Pooling at ⌊N/2⌋ layer (for faulty syndrome, i.e., T > 1)
    5. Two FC output layers -> n-dim soft decoded noise

    Args:
        code: ToricCode instance
        N: number of Transformer layers (default 6)
        d_model: embedding dimension (default 128)
        n_heads: number of attention heads (default 8)
        use_faulty: whether to handle faulty syndromes
    """

    def __init__(self, code: ToricCode, N: int = 6, d_model: int = 128,
                 n_heads: int = 8, use_faulty: bool = False):
        super().__init__()
        self.code = code
        self.N_layers = N
        self.d_model = d_model
        self.n = code.n
        self.n_s = code.n_s
        self.input_len = code.n + code.n_s  # |g_ω(s)| + |s|
        self.use_faulty = use_faulty
        self.pool_layer = N // 2  # ⌊N/2⌋

        # Initial noise estimator
        self.noise_estimator = NoiseEstimator(code.n_s, code.n)

        # One-hot style embedding: each element embedded into d-dimensional space
        # Embedding matrix W: (input_len, d_model) - learnable
        self.embedding = nn.Parameter(torch.randn(self.input_len, d_model) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(N)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),  # reduce embedding to 1D per element
        )
        self.output_fc = nn.Linear(self.input_len, code.n)  # project to n physical qubits

        # Register mask as buffer
        mask_tensor = torch.tensor(code.mask, dtype=torch.float32)
        self.register_buffer('mask', mask_tensor)

        # Logical operator matrix
        L_tensor = torch.tensor(code.L_matrix, dtype=torch.float32)
        self.register_buffer('L_matrix', L_tensor)

    def _embed_input(self, syndrome: torch.Tensor) -> torch.Tensor:
        """
        Create input embedding from syndrome.
        h_q(s) = [g_ω(s), s] -> positional embedding via Hadamard product.
        """
        # Noise estimation
        noise_est = self.noise_estimator(syndrome)  # (B, n)

        # Concatenate: [noise_est, syndrome]
        h_q = torch.cat([noise_est, syndrome], dim=-1)  # (B, n + n_s)

        # Positional embedding: Φ = (h_q · 1_d^T) ⊙ W
        # h_q: (B, input_len) -> (B, input_len, 1)
        # W: (input_len, d_model)
        h_expanded = h_q.unsqueeze(-1)  # (B, input_len, 1)
        phi = h_expanded * self.embedding.unsqueeze(0)  # (B, input_len, d_model)
        return phi, noise_est

    def forward(self, syndrome: torch.Tensor) -> dict:
        """
        Forward pass.

        Args:
            syndrome: (B, n_s) for perfect measurements or (B, T, n_s) for faulty

        Returns:
            dict with 'prediction' (soft noise), 'noise_est' (g_ω output)
        """
        if self.use_faulty and syndrome.dim() == 3:
            return self._forward_faulty(syndrome)
        else:
            return self._forward_perfect(syndrome)

    def _forward_perfect(self, syndrome: torch.Tensor) -> dict:
        """Perfect syndrome measurement (no repetitions)."""
        phi, noise_est = self._embed_input(syndrome)

        x = phi
        for block in self.blocks:
            x = block(x, self.mask)

        # Output projection
        out = self.output_proj(x).squeeze(-1)  # (B, input_len)
        prediction = torch.sigmoid(self.output_fc(out))  # (B, n)

        return {'prediction': prediction, 'noise_est': noise_est}

    def _forward_faulty(self, syndrome: torch.Tensor) -> dict:
        """Faulty syndrome measurement with T repetitions."""
        B, T, n_s = syndrome.shape

        # Process each time step independently
        all_embeddings = []
        all_noise_est = []
        for t in range(T):
            phi_t, ne_t = self._embed_input(syndrome[:, t, :])
            all_embeddings.append(phi_t)
            all_noise_est.append(ne_t)

        # Stack: (B, T, input_len, d_model)
        x = torch.stack(all_embeddings, dim=1)
        noise_est_avg = torch.stack(all_noise_est, dim=1).mean(dim=1)

        # Apply first half of blocks to each time step
        for i, block in enumerate(self.blocks[:self.pool_layer]):
            x_list = []
            for t in range(T):
                x_list.append(block(x[:, t], self.mask))
            x = torch.stack(x_list, dim=1)

        # Average pooling over time dimension at ⌊N/2⌋
        x = x.mean(dim=1)  # (B, input_len, d_model)

        # Apply remaining blocks
        for block in self.blocks[self.pool_layer:]:
            x = block(x, self.mask)

        # Output projection
        out = self.output_proj(x).squeeze(-1)
        prediction = torch.sigmoid(self.output_fc(out))

        return {'prediction': prediction, 'noise_est': noise_est_avg}


# ============================================================
# 6. Loss Functions
# ============================================================

class QECCTLoss(nn.Module):
    """
    Combined loss: L = λ_BER * L_BER + λ_LER * L_LER + λ_g * L_g

    - L_BER: BCE(f_θ(s), ε) — bit error rate
    - L_LER: BCE(Λ(L, bin(f_θ(s))), Lε) — logical error rate (differentiable)
    - L_g: BCE(g_ω(s), ε) — noise estimator loss
    """

    def __init__(self, L_matrix: torch.Tensor,
                 lambda_ber: float = 0.5,
                 lambda_ler: float = 1.0,
                 lambda_g: float = 0.5):
        super().__init__()
        self.register_buffer('L_matrix', L_matrix)
        self.lambda_ber = lambda_ber
        self.lambda_ler = lambda_ler
        self.lambda_g = lambda_g

    @staticmethod
    def bipolar_map(x: torch.Tensor) -> torch.Tensor:
        """ϕ(u) = 1 - 2u, maps {0,1} -> {+1,-1}"""
        return 1 - 2 * x

    @staticmethod
    def bipolar_inv(x: torch.Tensor) -> torch.Tensor:
        """ϕ^{-1}(u) = (1 - u) / 2"""
        return (1 - x) / 2

    def differentiable_logical(self, prediction: torch.Tensor,
                                L_matrix: torch.Tensor) -> torch.Tensor:
        """
        Differentiable XOR via bipolar mapping:
        Λ(L, x)_i = ϕ^{-1}(Π_j ϕ(L_{ij} · x_j))

        For sigmoid-binarized predictions.
        """
        # prediction: (B, n), L_matrix: (k, n)
        # bipolar of prediction
        pred_bipolar = self.bipolar_map(prediction)  # (B, n)

        # For each logical operator row
        k, n = L_matrix.shape
        results = []
        for i in range(k):
            L_row = L_matrix[i]  # (n,)
            # Only multiply elements where L[i,j] = 1
            # ϕ(L_{ij} · x_j) = ϕ(x_j) if L_{ij}=1, else ϕ(0)=1
            bipolar_L = self.bipolar_map(L_row)  # (n,)
            # Product: Π_j (pred_bipolar_j if L[i,j]=1 else 1)
            # = Π_j pred_bipolar_j^{L[i,j]}

            # Use log-space for numerical stability
            log_abs = torch.log(torch.abs(pred_bipolar) + 1e-10)
            sign = torch.sign(pred_bipolar)

            # Weighted sum in log space
            weighted_log = (L_row.unsqueeze(0) * log_abs)  # (B, n)
            weighted_sign = torch.where(
                L_row.unsqueeze(0) > 0,
                sign,
                torch.ones_like(sign)
            )

            # Product of signs
            total_sign = torch.prod(weighted_sign, dim=-1)  # (B,)
            total_log = weighted_log.sum(dim=-1)  # (B,)

            product = total_sign * torch.exp(total_log)
            result = self.bipolar_inv(product)  # (B,)
            results.append(result)

        return torch.stack(results, dim=-1)  # (B, k)

    def forward(self, prediction: torch.Tensor, noise_est: torch.Tensor,
                target_noise: torch.Tensor) -> dict:
        """
        Compute combined loss.

        Args:
            prediction: soft decoded noise from QECCT (B, n)
            noise_est: initial noise estimate from g_ω (B, n)
            target_noise: ground truth noise (B, n)
        """
        # L_BER: BCE(prediction, target)
        l_ber = F.binary_cross_entropy(prediction, target_noise)

        # L_g: BCE(noise_est, target)
        l_g = F.binary_cross_entropy(noise_est, target_noise)

        # L_LER: differentiable logical error
        pred_logical = self.differentiable_logical(prediction, self.L_matrix)
        target_logical = self.differentiable_logical(target_noise, self.L_matrix)
        l_ler = F.binary_cross_entropy(
            pred_logical.clamp(1e-7, 1 - 1e-7),
            target_logical.clamp(0, 1)
        )

        total = self.lambda_ber * l_ber + self.lambda_ler * l_ler + self.lambda_g * l_g

        return {
            'total': total,
            'ber': l_ber,
            'ler': l_ler,
            'g': l_g
        }


# ============================================================
# 7. Evaluation Metrics
# ============================================================

def compute_ber(prediction: np.ndarray, target: np.ndarray) -> float:
    """Bit Error Rate: fraction of incorrectly decoded bits."""
    pred_binary = (prediction > 0.5).astype(np.float32)
    return np.mean(pred_binary != target)


def compute_ler(prediction: np.ndarray, target: np.ndarray,
                L_matrix: np.ndarray) -> float:
    """
    Logical Error Rate: fraction of samples with at least one
    logical qubit error, up to logical operator equivalence.
    """
    pred_binary = (prediction > 0.5).astype(np.float32)
    # Residual error
    residual = ((pred_binary + target) % 2)  # what's left after correction

    # Check if residual is in the codespace (i.e., L @ residual = 0 mod 2)
    logical_error = (L_matrix @ residual.T) % 2  # (k, B)
    # Error if any logical qubit is flipped
    has_error = np.any(logical_error > 0, axis=0)  # (B,)
    return np.mean(has_error)
