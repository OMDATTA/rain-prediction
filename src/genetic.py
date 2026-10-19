"""Genetic algorithm (soft computing) that tunes the hybrid ensemble.

Chromosome = [w_lr, w_gbm, w_fuzzy, threshold]. Weights are normalised to sum
to 1. Fitness = F1-score of RainTomorrow on the validation set (rain is the
minority class, so F1 is a better target than accuracy).
"""
from __future__ import annotations

import numpy as np


def _f1(y, pred):
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


MIN_FUZZY = 0.10  # keep the interpretable expert-rule component in the blend


def decode(ch):
    w = np.abs(ch[:3]) + 1e-9
    w = w / w.sum()
    if w[2] < MIN_FUZZY:  # constraint repair: re-scale ML weights, floor fuzzy weight
        w[:2] = w[:2] / w[:2].sum() * (1 - MIN_FUZZY)
        w[2] = MIN_FUZZY
    thr = float(np.clip(ch[3], 0.15, 0.85))
    return w, thr


def fitness(ch, P, y):
    w, thr = decode(ch)
    return _f1(y, (P @ w >= thr).astype(int))


def optimise(P: np.ndarray, y: np.ndarray, pop_size: int = 40, generations: int = 60,
             mutation: float = 0.15, elite: int = 4, seed: int = 42, verbose: bool = True):
    """P: (n_samples, 3) probabilities from [LR, GBM, Fuzzy]. Returns dict + history."""
    rng = np.random.default_rng(seed)
    pop = np.column_stack([rng.random((pop_size, 3)), rng.uniform(0.25, 0.6, pop_size)])
    history = []
    for gen in range(generations):
        fit = np.array([fitness(c, P, y) for c in pop])
        order = np.argsort(-fit)
        pop, fit = pop[order], fit[order]
        history.append(float(fit[0]))
        if verbose and gen % 10 == 0:
            w, t = decode(pop[0])
            print(f"gen {gen:3d}  best F1={fit[0]:.4f}  w={np.round(w, 3)}  thr={t:.3f}")
        children = [pop[i].copy() for i in range(elite)]
        while len(children) < pop_size:
            # tournament selection
            a, b = (pop[min(rng.integers(0, pop_size, 3))] for _ in range(2))
            # blend (BLX-alpha) crossover
            alpha = rng.random(4)
            child = alpha * a + (1 - alpha) * b
            # gaussian mutation
            mask = rng.random(4) < 0.5
            child = child + mask * rng.normal(0, mutation, 4)
            child[:3] = np.abs(child[:3])
            children.append(child)
        pop = np.array(children)
    fit = np.array([fitness(c, P, y) for c in pop])
    best = pop[int(np.argmax(fit))]
    w, thr = decode(best)
    return {
        "weights": {"logistic_regression": round(float(w[0]), 4), "gbm": round(float(w[1]), 4),
                    "fuzzy": round(float(w[2]), 4)},
        "threshold": round(thr, 4),
        "val_f1": round(float(fit.max()), 4),
    }, history
