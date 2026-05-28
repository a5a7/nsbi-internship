#!/usr/bin/env python3
"""
run_measurement.py
──────────────────
NLL-based signal strength measurement using pre-scored observed events.
Reads ~/model_scores.pt (produced by score_both_models.py) and generates:
  1. Per-model NLL breakdown (t_rate, t_shape, t_total) — 1×2 figure
  2. Model comparison: CARL vs EveNet total NLL — single figure

Run (WSL, nsbi-venv activated):
  python3 /mnt/c/Users/Unnat/nsbi-internship/sessions/day2/nsbi-tutorial/scripts/run_measurement.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
HOME      = Path.home()
DATA_DIR  = Path('/mnt/c/Users/Unnat/nsbi-internship')
NSBI_DIR  = Path('/mnt/c/Users/Unnat/nsbi-internship/sessions/day2/nsbi-tutorial')
PLOTS_DIR = NSBI_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

SCORES_PT = HOME / 'observed_scores.pt'
XS_JSON   = DATA_DIR / 'ggzz4l_xs.json'

LUMI = 300.0  # ifb


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLL computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_t_shape(r_sig, r_sbi, n_obs, mu_space,
                    xs_sig, xs_sbi, xs_bkg, xs_int,
                    chunk=20_000):
    """
    t_shape[mu] = -2 * sum_i( n_i * log( r_sbi_over_bkg(x_i, mu) ) )

    r_sbi_over_bkg(x, mu) = [ xs_sig*(mu-√μ)*r_sig(x) + xs_sbi*√μ*r_sbi(x) + xs_bkg*(1-√μ) ]
                            / [ xs_sig*mu + xs_int*√μ + xs_bkg ]

    Computed in event chunks to keep memory bounded.
    """
    sqrt_mu  = torch.sqrt(mu_space)
    mult_sig = mu_space - sqrt_mu          # [M]
    mult_sbi = sqrt_mu                     # [M]
    mult_bkg = 1.0 - sqrt_mu              # [M]
    denom    = xs_sig * mu_space + xs_int * sqrt_mu + xs_bkg  # [M]

    N = len(n_obs)
    t = torch.zeros(len(mu_space))
    for start in range(0, N, chunk):
        rs = r_sig[start:start + chunk]   # [c]
        rb = r_sbi[start:start + chunk]   # [c]
        ni = n_obs[start:start + chunk]   # [c]
        numer = (xs_sig * mult_sig[None, :] * rs[:, None]
               + xs_sbi * mult_sbi[None, :] * rb[:, None]
               + xs_bkg * mult_bkg[None, :])                   # [c, M]
        r_mu = (numer / denom[None, :]).clamp(min=1e-9)        # [c, M]
        t   += -2.0 * (ni[:, None] * torch.log(r_mu)).sum(0)  # [M]
    return t


def compute_t_rate(n_obs_total, n_files, mu_space, xs_sig, xs_bkg, xs_int):
    """Poisson rate term summed over n_files independent experiments.

    Each experiment predicts nu_mu events; the combined rate term is:
      t_rate = n_files * nu_mu - N_total * log(nu_mu)
    This is minimised at nu_mu = N_total / n_files = N_avg, i.e. at mu_true.
    """
    nu_mu = (xs_sig * mu_space
           + xs_int * torch.sqrt(mu_space)
           + xs_bkg) * LUMI                # [M]
    return n_files * nu_mu - n_obs_total * torch.log(nu_mu)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Plotting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_nll(ax, mu_space, t_rate, t_shape, t_total, title):
    """Plot t_rate, t_shape, t_total on one axis, all shifted to min=0."""
    def shift(t): return t - t.min()

    mu = mu_space.numpy()
    ax.plot(mu, shift(t_rate).numpy(),   label='$t_{\\rm rate}$',  linestyle=':')
    ax.plot(mu, shift(t_shape).numpy(),  label='$t_{\\rm shape}$', linestyle='--')
    ax.plot(mu, shift(t_total).numpy(),  label='$t$',              linewidth=2)

    mu_hat = mu_space[t_total.argmin()].item()
    ax.scatter(mu_hat, 0, color='red', zorder=5, label=f'$\\hat{{\\mu}}={mu_hat:.2f}$')

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label='$1\\sigma$')
    ax.axhline(4.0, color='gray', linestyle=':',  linewidth=0.8, label='$2\\sigma$')

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 10)
    ax.set_xlabel('$\\mu$')
    ax.set_ylabel('$-2\\log\\lambda(\\mu)$')
    ax.legend(fontsize=8)
    ax.set_title(title)

    # Print 1σ interval
    t_norm = shift(t_total).numpy()
    in_1sig = mu[t_norm < 1.0]
    if len(in_1sig) > 0:
        print(f'  {title}: μ̂={mu_hat:.3f}, 1σ=[{in_1sig[0]:.3f}, {in_1sig[-1]:.3f}]')
    else:
        print(f'  {title}: μ̂={mu_hat:.3f}, 1σ interval not found in [0,4]')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # ── Load scores & cross sections ──────────────────────────────────────────
    print(f'Loading {SCORES_PT} ...')
    scores = torch.load(SCORES_PT, map_location='cpu')
    r_sbi_carl   = scores['r_sbi_over_bkg_carl'].float()
    r_sig_carl   = scores['r_sig_over_bkg_carl'].float()
    r_sbi_evenet = scores['r_sbi_over_bkg_evenet'].float()
    r_sig_evenet = scores['r_sig_over_bkg_evenet'].float()
    n_obs        = scores['n'].float()
    N_obs_total  = n_obs.sum()
    n_files      = int(scores.get('n_files', 1))
    print(f'  {len(n_obs):,} events, N_obs={N_obs_total:.1f}, n_files={n_files}')

    print(f'\nLoading {XS_JSON} ...')
    with open(XS_JSON) as f:
        xs = json.load(f)
    xs_sig = float(np.prod(xs['sig']))
    xs_bkg = float(np.prod(xs['bkg']))
    xs_int = float(np.prod(xs['int']))
    xs_sbi = float(np.prod(xs['sbi']))
    print(f'  xs_sig={xs_sig:.4f} fb  xs_bkg={xs_bkg:.4f} fb  xs_int={xs_int:.4f} fb')
    print(f'  nu_sig(μ=1)={xs_sig*LUMI:.1f}  nu_bkg={xs_bkg*LUMI:.1f}  nu_sbi(μ=1)={xs_sbi*LUMI:.1f}')

    mu_space = torch.linspace(0.0, 4.0, 401)

    # ── Rate term (same for both models) ─────────────────────────────────────
    print('\nComputing rate term ...')
    t_rate = compute_t_rate(N_obs_total, n_files, mu_space, xs_sig, xs_bkg, xs_int)

    # ── Shape terms ───────────────────────────────────────────────────────────
    print('Computing CARL shape term ...')
    t_shape_carl = compute_t_shape(
        r_sig_carl, r_sbi_carl, n_obs, mu_space,
        xs_sig, xs_sbi, xs_bkg, xs_int)

    print('Computing EveNet shape term ...')
    t_shape_evenet = compute_t_shape(
        r_sig_evenet, r_sbi_evenet, n_obs, mu_space,
        xs_sig, xs_sbi, xs_bkg, xs_int)

    t_carl   = t_rate + t_shape_carl
    t_evenet = t_rate + t_shape_evenet

    # ── Plot 1: per-model breakdown ────────────────────────────────────────────
    print('\nResults:')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_nll(axes[0], mu_space, t_rate, t_shape_carl,   t_carl,   'CARL')
    plot_nll(axes[1], mu_space, t_rate, t_shape_evenet, t_evenet, 'EveNet')
    plt.suptitle('NSBI Measurement: $-2\\log\\lambda(\\mu)$', fontsize=13)
    plt.tight_layout()
    out = PLOTS_DIR / 'nll_breakdown.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved → {out}')

    # ── Plot 2: CARL vs EveNet comparison ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    mu = mu_space.numpy()

    def shift(t): return (t - t.min()).numpy()

    ax.plot(mu, shift(t_carl),   label='CARL',   linewidth=2)
    ax.plot(mu, shift(t_evenet), label='EveNet', linewidth=2, linestyle='--')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label='$1\\sigma$')
    ax.axhline(4.0, color='gray', linestyle=':',  linewidth=0.8, label='$2\\sigma$')
    ax.set_xlim(0, 4); ax.set_ylim(0, 10)
    ax.set_xlabel('$\\mu$')
    ax.set_ylabel('$-2\\log\\lambda(\\mu)$')
    ax.legend()
    ax.set_title('CARL vs EveNet: Total NLL')
    plt.tight_layout()
    out = PLOTS_DIR / 'nll_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out}')

    print('\nDone. Plots in:', PLOTS_DIR)


if __name__ == '__main__':
    main()
