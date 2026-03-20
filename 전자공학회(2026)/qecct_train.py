"""
QECCT Training & Evaluation Utilities
======================================
Training loop, MWPM baseline, and plotting utilities.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import time
import math

from qecct_models import ToricCode, QECCT, QECCTLoss, compute_ber, compute_ler


# ============================================================
# 1. MWPM Baseline (via PyMatching)
# ============================================================

def evaluate_mwpm(code: ToricCode, p_range: List[float],
                  noise_type: str = 'independent',
                  n_samples: int = 10000) -> Dict[str, List[float]]:
    """
    Evaluate MWPM decoder using PyMatching.
    Requires: pip install pymatching

    Args:
        code: ToricCode instance
        p_range: list of physical error rates to evaluate
        noise_type: 'independent' or 'depolarization'
        n_samples: number of test samples
    """
    try:
        from pymatching import Matching
    except ImportError:
        print("PyMatching not installed. Run: pip install pymatching")
        print("Returning empty results.")
        return {'p': p_range, 'ber': [0.0]*len(p_range), 'ler': [0.0]*len(p_range)}

    ber_list, ler_list = [], []

    for p in p_range:
        total_bit_errors = 0
        total_logical_errors = 0

        for _ in range(n_samples):
            if noise_type == 'independent':
                noise = code.sample_independent_noise(p, 1)[0]
            else:
                noise = code.sample_depolarization_noise(p, 1)[0]

            syndrome = code.get_syndrome(noise)

            # MWPM decode X and Z separately
            L = code.L
            n_half = L * L

            # X-stabilizer decoding (first half of H, first half of noise)
            try:
                m_x = Matching(code.H_x[:, :code.n].astype(np.uint8))
                correction_x = m_x.decode(syndrome[:n_half].astype(np.uint8))
            except Exception:
                correction_x = np.zeros(code.n, dtype=np.uint8)

            # Z-stabilizer decoding
            try:
                m_z = Matching(code.H_z[:, :code.n].astype(np.uint8))
                correction_z = m_z.decode(syndrome[n_half:].astype(np.uint8))
            except Exception:
                correction_z = np.zeros(code.n, dtype=np.uint8)

            correction = ((correction_x + correction_z) % 2).astype(np.float32)
            residual = ((correction + noise) % 2)

            # BER
            total_bit_errors += np.sum(residual > 0)

            # LER
            logical = (code.L_matrix @ residual) % 2
            if np.any(logical > 0):
                total_logical_errors += 1

        ber_list.append(total_bit_errors / (n_samples * code.n))
        ler_list.append(total_logical_errors / n_samples)

    return {'p': p_range, 'ber': ber_list, 'ler': ler_list}


# ============================================================
# 2. Training Loop
# ============================================================

class Trainer:
    """
    QECCT Trainer following the paper's setup:
    - Adam optimizer, lr=5e-4, cosine decay to 5e-7
    - 512 samples per minibatch
    - Random noise sampling in test range per batch
    """

    def __init__(self, model: QECCT, code: ToricCode,
                 device: torch.device,
                 lr: float = 5e-4,
                 lr_min: float = 5e-7,
                 batch_size: int = 512,
                 noise_type: str = 'independent',
                 p_range: Tuple[float, float] = (0.01, 0.15),
                 lambda_ber: float = 0.5,
                 lambda_ler: float = 1.0,
                 lambda_g: float = 0.5):
        self.model = model.to(device)
        self.code = code
        self.device = device
        self.batch_size = batch_size
        self.noise_type = noise_type
        self.p_range = p_range

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = QECCTLoss(
            L_matrix=torch.tensor(code.L_matrix, dtype=torch.float32).to(device),
            lambda_ber=lambda_ber,
            lambda_ler=lambda_ler,
            lambda_g=lambda_g
        ).to(device)

        # Scheduler will be set when training starts
        self.lr = lr
        self.lr_min = lr_min

    def _sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of noisy data."""
        # Random physical error rate in range
        p = np.random.uniform(self.p_range[0], self.p_range[1])

        if self.noise_type == 'independent':
            noise = self.code.sample_independent_noise(p, self.batch_size)
        else:
            noise = self.code.sample_depolarization_noise(p, self.batch_size)

        # Compute syndrome
        syndromes = np.array([self.code.get_syndrome(noise[i]) for i in range(self.batch_size)])

        noise_t = torch.tensor(noise, dtype=torch.float32).to(self.device)
        syndrome_t = torch.tensor(syndromes, dtype=torch.float32).to(self.device)

        return syndrome_t, noise_t

    def train_epoch(self, n_batches: int = 5000) -> Dict[str, float]:
        """Train for one epoch (n_batches minibatches)."""
        self.model.train()
        total_loss = 0
        total_ber_loss = 0
        total_ler_loss = 0
        total_g_loss = 0

        for step in range(n_batches):
            syndrome, target_noise = self._sample_batch()

            self.optimizer.zero_grad()
            output = self.model(syndrome)

            losses = self.criterion(
                output['prediction'],
                output['noise_est'],
                target_noise
            )

            losses['total'].backward()
            self.optimizer.step()

            total_loss += losses['total'].item()
            total_ber_loss += losses['ber'].item()
            total_ler_loss += losses['ler'].item()
            total_g_loss += losses['g'].item()

        n = n_batches
        return {
            'total': total_loss / n,
            'ber': total_ber_loss / n,
            'ler': total_ler_loss / n,
            'g': total_g_loss / n,
        }

    @torch.no_grad()
    def evaluate(self, p_range: List[float], n_samples: int = 10000) -> Dict:
        """Evaluate model at different physical error rates."""
        self.model.eval()
        results = {'p': p_range, 'ber': [], 'ler': []}

        for p in p_range:
            if self.noise_type == 'independent':
                noise = self.code.sample_independent_noise(p, n_samples)
            else:
                noise = self.code.sample_depolarization_noise(p, n_samples)

            syndromes = np.array([self.code.get_syndrome(noise[i]) for i in range(n_samples)])

            # Process in batches
            all_preds = []
            bs = min(512, n_samples)
            for i in range(0, n_samples, bs):
                s_batch = torch.tensor(syndromes[i:i+bs], dtype=torch.float32).to(self.device)
                out = self.model(s_batch)
                all_preds.append(out['prediction'].cpu().numpy())

            predictions = np.concatenate(all_preds, axis=0)
            ber = compute_ber(predictions, noise)
            ler = compute_ler(predictions, noise, self.code.L_matrix)
            results['ber'].append(ber)
            results['ler'].append(ler)

        return results

    def train(self, n_epochs: int = 200, n_batches_per_epoch: int = 5000,
              eval_every: int = 10, eval_p_range: List[float] = None,
              eval_n_samples: int = 10000, verbose: bool = True) -> Dict:
        """
        Full training loop with cosine decay scheduling.
        """
        # Setup cosine decay scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=self.lr_min
        )

        history = {'epoch': [], 'train_loss': [], 'eval_results': []}

        if eval_p_range is None:
            eval_p_range = [0.02, 0.05, 0.08, 0.10, 0.12]

        for epoch in range(1, n_epochs + 1):
            start = time.time()
            train_metrics = self.train_epoch(n_batches_per_epoch)
            elapsed = time.time() - start

            scheduler.step()

            history['epoch'].append(epoch)
            history['train_loss'].append(train_metrics['total'])

            if verbose:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{n_epochs} | "
                      f"Loss: {train_metrics['total']:.4f} "
                      f"(BER: {train_metrics['ber']:.4f}, "
                      f"LER: {train_metrics['ler']:.4f}, "
                      f"G: {train_metrics['g']:.4f}) | "
                      f"LR: {lr:.2e} | Time: {elapsed:.1f}s")

            if epoch % eval_every == 0:
                eval_results = self.evaluate(eval_p_range, eval_n_samples)
                history['eval_results'].append({
                    'epoch': epoch,
                    'results': eval_results
                })
                if verbose:
                    for i, p in enumerate(eval_p_range):
                        print(f"  p={p:.3f}: BER={eval_results['ber'][i]:.6f}, "
                              f"LER={eval_results['ler'][i]:.6f}")

        return history


# ============================================================
# 3. Plotting Utilities
# ============================================================

def plot_ler_comparison(qecct_results: Dict, mwpm_results: Dict,
                        code_L: int, noise_type: str,
                        save_path: Optional[str] = None):
    """Plot LER comparison between QECCT and MWPM."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.semilogy(mwpm_results['p'], mwpm_results['ler'],
                'o-', label='MWPM', linewidth=2, markersize=6)
    ax.semilogy(qecct_results['p'], qecct_results['ler'],
                's--', label='QECCT', linewidth=2, markersize=6)

    ax.set_xlabel('Physical Error Rate (p)', fontsize=12)
    ax.set_ylabel('Logical Error Rate (LER)', fontsize=12)
    ax.set_title(f'L={code_L} Toric Code - {noise_type.capitalize()} Noise', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training loss curve."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(history['epoch'], history['train_loss'], linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_multi_L_comparison(all_results: Dict[int, Dict],
                             metric: str = 'ler',
                             save_path: Optional[str] = None):
    """Plot LER/BER for multiple lattice sizes."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, (L, data) in enumerate(sorted(all_results.items())):
        if 'mwpm' in data:
            ax.semilogy(data['mwpm']['p'], data['mwpm'][metric],
                        'o-', color=colors[idx], label=f'MWPM L={L}',
                        linewidth=2, markersize=5)
        if 'qecct' in data:
            ax.semilogy(data['qecct']['p'], data['qecct'][metric],
                        's--', color=colors[idx], label=f'QECCT L={L}',
                        linewidth=2, markersize=5)

    ax.set_xlabel('Physical Error Rate (p)', fontsize=12)
    ax.set_ylabel(f'{"Logical" if metric == "ler" else "Bit"} Error Rate', fontsize=12)
    ax.set_title(f'{metric.upper()} vs Physical Error Rate', fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
