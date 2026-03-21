"""
QECCT Student Model + Knowledge Distillation for On-Device SoC
===============================================================
Lightweight student architecture + KD training + pruning + quantization utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from typing import Dict, List, Tuple, Optional

from qecct_models import ToricCode, QECCT, QECCTLoss, NoiseEstimator, compute_ber, compute_ler
from qecct_models import MaskedMultiHeadAttention, TransformerBlock


# ============================================================
# 1. Lightweight Student QECCT
# ============================================================

class QECCTStudent(nn.Module):
    """
    Lightweight QECCT for on-device SoC deployment.
    Key differences from Teacher:
    - Fewer layers (N=2 vs 6)
    - Smaller embedding (d=32~64 vs 128)
    - Fewer heads (2~4 vs 8)
    - Shared embedding weights option
    - Optional Linear Attention
    """

    def __init__(self, code: ToricCode, N: int = 2, d_model: int = 32,
                 n_heads: int = 4, use_linear_attn: bool = False,
                 use_faulty: bool = False):
        super().__init__()
        self.code = code
        self.N_layers = N
        self.d_model = d_model
        self.n = code.n
        self.n_s = code.n_s
        self.input_len = code.n + code.n_s
        self.use_faulty = use_faulty
        self.pool_layer = N // 2
        self.use_linear_attn = use_linear_attn

        # Compact noise estimator (smaller hidden)
        hidden = 3 * code.n_s  # 3x instead of 5x
        self.noise_estimator = nn.Sequential(
            nn.Linear(code.n_s, hidden),
            nn.GELU(),
            nn.Linear(hidden, code.n),
            nn.Sigmoid()
        )

        # Embedding
        self.embedding = nn.Parameter(torch.randn(self.input_len, d_model) * 0.02)

        # Transformer blocks (with optional linear attention)
        if use_linear_attn:
            self.blocks = nn.ModuleList([
                LinearTransformerBlock(d_model, n_heads) for _ in range(N)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads) for _ in range(N)
            ])

        # Output
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.output_fc = nn.Linear(self.input_len, code.n)

        mask_tensor = torch.tensor(code.mask, dtype=torch.float32)
        self.register_buffer('mask', mask_tensor)
        L_tensor = torch.tensor(code.L_matrix, dtype=torch.float32)
        self.register_buffer('L_matrix', L_tensor)

    def _embed_input(self, syndrome):
        noise_est = self.noise_estimator(syndrome)
        h_q = torch.cat([noise_est, syndrome], dim=-1)
        h_expanded = h_q.unsqueeze(-1)
        phi = h_expanded * self.embedding.unsqueeze(0)
        return phi, noise_est

    def forward(self, syndrome, return_hidden=False):
        if self.use_faulty and syndrome.dim() == 3:
            return self._forward_faulty(syndrome)

        phi, noise_est = self._embed_input(syndrome)
        x = phi
        hiddens = []
        for block in self.blocks:
            x = block(x, self.mask)
            if return_hidden:
                hiddens.append(x)

        out = self.output_proj(x).squeeze(-1)
        prediction = torch.sigmoid(self.output_fc(out))

        result = {'prediction': prediction, 'noise_est': noise_est}
        if return_hidden:
            result['hiddens'] = hiddens
        return result

    def _forward_faulty(self, syndrome):
        B, T, n_s = syndrome.shape
        all_emb, all_ne = [], []
        for t in range(T):
            phi_t, ne_t = self._embed_input(syndrome[:, t, :])
            all_emb.append(phi_t)
            all_ne.append(ne_t)

        x = torch.stack(all_emb, dim=1)
        noise_est_avg = torch.stack(all_ne, dim=1).mean(dim=1)

        for block in self.blocks[:self.pool_layer]:
            x_list = [block(x[:, t], self.mask) for t in range(T)]
            x = torch.stack(x_list, dim=1)

        x = x.mean(dim=1)
        for block in self.blocks[self.pool_layer:]:
            x = block(x, self.mask)

        out = self.output_proj(x).squeeze(-1)
        prediction = torch.sigmoid(self.output_fc(out))
        return {'prediction': prediction, 'noise_est': noise_est_avg}


# ============================================================
# 2. Linear Attention Block (O(n) complexity)
# ============================================================

class LinearAttention(nn.Module):
    """Kernel-based linear attention: O(n) complexity via ELU feature map."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def feature_map(self, x):
        return F.elu(x) + 1  # ELU + 1 as kernel feature map

    def forward(self, x, mask=None):
        B, L, D = x.shape
        Q = self.feature_map(self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2))
        K = self.feature_map(self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2))
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Linear attention: O(n·d²) instead of O(n²·d)
        KV = torch.matmul(K.transpose(-2, -1), V)  # (B, h, d_k, d_k)
        Z = 1.0 / (torch.matmul(Q, K.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6)
        out = torch.matmul(Q, KV) * Z

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)


class LinearTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attn = LinearAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.norm1(x)
        h = self.attn(h, mask)
        x = x + self.dropout(h)
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x


# ============================================================
# 3. Knowledge Distillation Loss
# ============================================================

class KDLoss(nn.Module):
    """
    Knowledge Distillation loss combining:
    1. Task loss (student's own BER+LER+g loss)
    2. Output KD loss (soft target from teacher predictions)
    3. Attention transfer (learnable linear projection for dimension matching)
    """

    def __init__(self, L_matrix, teacher_d=128, student_d=32,
                 temperature=3.0, alpha_task=0.5, alpha_kd=0.3,
                 alpha_attn=0.2, lambda_ber=0.5, lambda_ler=1.0, lambda_g=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha_task = alpha_task
        self.alpha_kd = alpha_kd
        self.alpha_attn = alpha_attn
        self.task_loss = QECCTLoss(L_matrix, lambda_ber, lambda_ler, lambda_g)

        # Learnable 1x1 linear projection for teacher→student dimension matching
        # Avoids information loss from adaptive_avg_pool1d
        if teacher_d != student_d:
            self.dim_projector = nn.Linear(teacher_d, student_d, bias=False)
        else:
            self.dim_projector = None

    def _project_teacher_hidden(self, t_h, s_d):
        """Project teacher hidden states to student dimension via learned linear map."""
        if self.dim_projector is not None and t_h.shape[-1] != s_d:
            return self.dim_projector(t_h)  # (B, seq, teacher_d) -> (B, seq, student_d)
        return t_h

    def forward(self, student_out, teacher_out, target_noise):
        # 1. Task loss
        task = self.task_loss(student_out['prediction'], student_out['noise_est'], target_noise)

        # 2. Output KD (soft target)
        T = self.temperature
        s_logits = torch.log(student_out['prediction'].clamp(1e-7, 1-1e-7) /
                             (1 - student_out['prediction'].clamp(1e-7, 1-1e-7)))
        t_logits = torch.log(teacher_out['prediction'].clamp(1e-7, 1-1e-7) /
                             (1 - teacher_out['prediction'].clamp(1e-7, 1-1e-7)))

        kd_loss = F.mse_loss(torch.sigmoid(s_logits / T), torch.sigmoid(t_logits / T)) * (T * T)

        # 3. Attention transfer with learnable projection
        attn_loss = torch.tensor(0.0, device=target_noise.device)
        if 'hiddens' in student_out and 'hiddens' in teacher_out:
            s_hiddens = student_out['hiddens']
            t_hiddens = teacher_out['hiddens']
            n_match = min(len(s_hiddens), len(t_hiddens))
            for i in range(n_match):
                s_h = s_hiddens[i]
                t_h = teacher_out['hiddens'][-(n_match - i)]
                # Learnable linear projection instead of avg pooling
                t_h = self._project_teacher_hidden(t_h, s_h.shape[-1])
                attn_loss = attn_loss + F.mse_loss(
                    F.normalize(s_h.pow(2).mean(dim=-1), dim=-1),
                    F.normalize(t_h.pow(2).mean(dim=-1), dim=-1)
                )
            attn_loss = attn_loss / max(n_match, 1)

        total = (self.alpha_task * task['total'] +
                 self.alpha_kd * kd_loss +
                 self.alpha_attn * attn_loss)

        return {
            'total': total,
            'task': task['total'],
            'kd': kd_loss,
            'attn': attn_loss,
            'ber': task['ber'],
            'ler': task['ler'],
        }


# ============================================================
# 4. KD Trainer
# ============================================================

class KDTrainer:
    """Knowledge Distillation trainer: Teacher -> Student."""

    def __init__(self, teacher: QECCT, student: QECCTStudent, code: ToricCode,
                 device: torch.device, lr=5e-4, lr_min=5e-7, batch_size=512,
                 noise_type='independent', p_range=(0.01, 0.15),
                 temperature=3.0, alpha_task=0.5, alpha_kd=0.3, alpha_attn=0.2):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.code = code
        self.device = device
        self.batch_size = batch_size
        self.noise_type = noise_type
        self.p_range = p_range

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

        L_mat = torch.tensor(code.L_matrix, dtype=torch.float32).to(device)
        self.criterion = KDLoss(
            L_mat, teacher_d=teacher.d_model, student_d=student.d_model,
            temperature=temperature, alpha_task=alpha_task,
            alpha_kd=alpha_kd, alpha_attn=alpha_attn
        ).to(device)
        # Optimizer includes both student params and projector params
        all_params = list(student.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        self.lr = lr
        self.lr_min = lr_min

    def _sample_batch(self):
        p = np.random.uniform(self.p_range[0], self.p_range[1])
        if self.noise_type == 'independent':
            noise = self.code.sample_independent_noise(p, self.batch_size)
        else:
            noise = self.code.sample_depolarization_noise(p, self.batch_size)
        syndromes = np.array([self.code.get_syndrome(noise[i]) for i in range(self.batch_size)])
        return (torch.tensor(syndromes, dtype=torch.float32).to(self.device),
                torch.tensor(noise, dtype=torch.float32).to(self.device))

    def train_epoch(self, n_batches=5000):
        self.student.train()
        totals = {'total': 0, 'task': 0, 'kd': 0, 'attn': 0, 'ber': 0, 'ler': 0}

        for _ in range(n_batches):
            syn, noise = self._sample_batch()
            self.optimizer.zero_grad()

            with torch.no_grad():
                t_out = self.teacher(syn)
            s_out = self.student(syn, return_hidden=True)

            losses = self.criterion(s_out, t_out, noise)
            losses['total'].backward()
            self.optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()

        return {k: v / n_batches for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self, p_range, n_samples=10000):
        self.student.eval()
        results = {'p': p_range, 'ber': [], 'ler': []}
        for p in p_range:
            if self.noise_type == 'independent':
                noise = self.code.sample_independent_noise(p, n_samples)
            else:
                noise = self.code.sample_depolarization_noise(p, n_samples)
            syns = np.array([self.code.get_syndrome(noise[i]) for i in range(n_samples)])
            preds = []
            for i in range(0, n_samples, 512):
                s = torch.tensor(syns[i:i+512], dtype=torch.float32).to(self.device)
                preds.append(self.student(s)['prediction'].cpu().numpy())
            preds = np.concatenate(preds, axis=0)
            results['ber'].append(compute_ber(preds, noise))
            results['ler'].append(compute_ler(preds, noise, self.code.L_matrix))
        return results

    def train(self, n_epochs=200, n_batches_per_epoch=5000, eval_every=10,
              eval_p_range=None, eval_n_samples=10000, verbose=True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=self.lr_min)
        history = {'epoch': [], 'train_loss': [], 'kd_loss': [], 'eval_results': []}
        if eval_p_range is None:
            eval_p_range = [0.02, 0.05, 0.08, 0.10, 0.12]

        for epoch in range(1, n_epochs + 1):
            import time; start = time.time()
            m = self.train_epoch(n_batches_per_epoch)
            elapsed = time.time() - start
            scheduler.step()

            history['epoch'].append(epoch)
            history['train_loss'].append(m['total'])
            history['kd_loss'].append(m['kd'])

            if verbose:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{n_epochs} | Total: {m['total']:.4f} "
                      f"(Task: {m['task']:.4f}, KD: {m['kd']:.4f}, Attn: {m['attn']:.4f}) | "
                      f"LR: {lr:.2e} | {elapsed:.1f}s")

            if epoch % eval_every == 0:
                r = self.evaluate(eval_p_range, eval_n_samples)
                history['eval_results'].append({'epoch': epoch, 'results': r})
                if verbose:
                    for i, p in enumerate(eval_p_range):
                        print(f"  p={p:.3f}: BER={r['ber'][i]:.6f}, LER={r['ler'][i]:.6f}")

        return history


# ============================================================
# 5. Structured Pruning
# ============================================================

def apply_structured_pruning(model, prune_ratio=0.5):
    """
    Apply L1-norm based STRUCTURED pruning (neuron/channel-level) to Linear layers.
    Uses ln_structured with dim=0 (output neuron pruning) to enable real
    inference speedup on SoC hardware, unlike unstructured pruning which
    only creates sparse weight matrices without actual compute reduction.
    Returns pruned model copy and pruning stats.
    """
    import torch.nn.utils.prune as prune

    pruned_model = copy.deepcopy(model)
    total_params_before = sum(p.numel() for p in pruned_model.parameters())
    pruned_count = 0

    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            # Structured pruning: remove entire output neurons (dim=0)
            # based on L1-norm of each neuron's weight vector.
            # This allows actual row removal -> smaller dense matrices
            # -> real latency reduction on SoC without sparse HW support.
            prune.ln_structured(module, name='weight', amount=prune_ratio, n=1, dim=0)
            prune.remove(module, 'weight')
            pruned_count += 1

    total_params_after = sum((p != 0).sum().item() for p in pruned_model.parameters())
    sparsity = 1 - total_params_after / total_params_before

    stats = {
        'total_before': total_params_before,
        'nonzero_after': total_params_after,
        'sparsity': sparsity,
        'layers_pruned': pruned_count,
    }
    return pruned_model, stats


# ============================================================
# 6. INT8 Quantization
# ============================================================

def apply_dynamic_quantization(model):
    """Apply PyTorch dynamic quantization (INT8) to Linear layers.
    Falls back to original model on unsupported platforms (e.g., macOS ARM)."""
    try:
        quantized = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        return quantized
    except RuntimeError as e:
        print(f"⚠️ Dynamic quantization not supported on this platform: {e}")
        print("   Returning original model (simulated quantization for size estimation).")
        return model


def measure_model_size(model):
    """Measure model size in bytes."""
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.tell()
    return size_bytes


# ============================================================
# 7. Paper-Ready Visualization Functions
# ============================================================

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11


def plot_model_comparison_bar(teacher_params, student_params, pruned_params=None,
                               quantized_size_mb=None, save_path=None):
    """Bar chart: parameter counts comparison (Teacher vs Student vs Pruned)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Parameter count
    labels = ['Teacher\n(QECCT)', 'Student\n(KD)']
    values = [teacher_params, student_params]
    colors = ['#2196F3', '#4CAF50']
    if pruned_params is not None:
        labels.append('Student\n+Pruning')
        values.append(pruned_params)
        colors.append('#FF9800')

    bars = axes[0].bar(labels, values, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                     f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    axes[0].set_ylabel('Number of Parameters', fontsize=12)
    axes[0].set_title('(a) Model Size Comparison', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # (b) Compression ratio
    ratios = [1.0, student_params/teacher_params]
    ratio_labels = ['Teacher', 'Student (KD)']
    ratio_colors = ['#2196F3', '#4CAF50']
    if pruned_params:
        ratios.append(pruned_params/teacher_params)
        ratio_labels.append('Student+Prune')
        ratio_colors.append('#FF9800')

    bars2 = axes[1].barh(ratio_labels, ratios, color=ratio_colors, edgecolor='white', height=0.5)
    for bar, r in zip(bars2, ratios):
        axes[1].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                     f'{r:.1%}', va='center', fontweight='bold', fontsize=10)
    axes[1].set_xlabel('Relative Size (vs Teacher)', fontsize=12)
    axes[1].set_title('(b) Compression Ratio', fontsize=13, fontweight='bold')
    axes[1].set_xlim(0, 1.3)
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_ler_teacher_student(teacher_results, student_results, mwpm_results=None,
                              pruned_results=None, code_L=3, save_path=None):
    """LER comparison: Teacher vs Student vs MWPM (paper Figure)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if mwpm_results:
        ax.semilogy(mwpm_results['p'], mwpm_results['ler'], 'D-',
                    color='#9E9E9E', label='MWPM', linewidth=2, markersize=7)
    ax.semilogy(teacher_results['p'], teacher_results['ler'], 'o-',
                color='#2196F3', label='Teacher (QECCT)', linewidth=2.5, markersize=7)
    ax.semilogy(student_results['p'], student_results['ler'], 's--',
                color='#4CAF50', label='Student (KD)', linewidth=2.5, markersize=7)
    if pruned_results:
        ax.semilogy(pruned_results['p'], pruned_results['ler'], '^:',
                    color='#FF9800', label='Student+Pruning', linewidth=2, markersize=7)

    ax.set_xlabel('Physical Error Rate (p)', fontsize=13)
    ax.set_ylabel('Logical Error Rate (LER)', fontsize=13)
    ax.set_title(f'Toric Code L={code_L}: Decoder Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_kd_training_curves(history, save_path=None):
    """KD training loss decomposition over epochs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['epoch'], history['train_loss'], '-', color='#E91E63',
            label='Total Loss', linewidth=2.5)
    ax.plot(history['epoch'], history['kd_loss'], '--', color='#00BCD4',
            label='KD Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Knowledge Distillation Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_compression_summary_table(metrics_dict, save_path=None):
    """
    Table-style figure summarizing compression results.
    metrics_dict: {method_name: {params, size_mb, ler_at_threshold, speedup}}
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    methods = list(metrics_dict.keys())
    columns = ['Method', 'Params', 'Size (KB)', 'LER @p=0.10', 'Compression']
    cell_data = []
    for m in methods:
        d = metrics_dict[m]
        cell_data.append([
            m,
            f"{d.get('params', 'N/A'):,}" if isinstance(d.get('params'), int) else str(d.get('params', 'N/A')),
            f"{d.get('size_kb', 0):.1f}",
            f"{d.get('ler_010', 0):.4f}",
            f"{d.get('compression', 1.0):.1f}×",
        ])

    table = ax.table(cellText=cell_data, colLabels=columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(columns)):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Alternate row colors
    for i in range(1, len(methods) + 1):
        color = '#E3F2FD' if i % 2 == 1 else '#FFFFFF'
        for j in range(len(columns)):
            table[i, j].set_facecolor(color)

    plt.title('Model Compression Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_latency_comparison(methods, latencies_ms, save_path=None):
    """Horizontal bar chart of inference latency per sample."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    bars = ax.barh(methods, latencies_ms, color=colors[:len(methods)],
                   edgecolor='white', height=0.5)
    for bar, v in zip(bars, latencies_ms):
        ax.text(bar.get_width() + max(latencies_ms)*0.02,
                bar.get_y() + bar.get_height()/2,
                f'{v:.3f}ms', va='center', fontweight='bold')
    ax.set_xlabel('Inference Latency (ms/sample)', fontsize=12)
    ax.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
