"""Fits three multisensory integration models (discrete responses) to each observer.

Data: five files data/datasub/DataSub1-5.txt, each a 7x5 matrix of counts (successes
out of 24) where a "success" is responding "d" along a b→d speech continuum.

Rows:
    0: auditory-only stimuli (5 auditory levels, columns 1..5)
    1: visual-only stimuli   (5 visual levels, columns 1..5)
    2-6: audiovisual stimuli; row gives visual level, column auditory level

Models (all assume binomial responses):
    - early strong fusion: logit p = b0 + bA * audio + bV * visual
    - strong fusion probability matching: combine auditory and visual log-odds
      (logit p_AV = bAV + logit p_A + logit p_V)
    - late strong fusion: mix unimodal probabilities with weight w (p_AV = w*p_A+(1-w)*p_V)

The script prints negative log-likelihood (NLL) on the full data and a 7-fold
leave-one-row-out cross-validation NLL for each model and observer.
Run directly: python MultisensoryIntegrationDiscrete.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import minimize

DATA_DIR = Path(__file__).resolve().parent / "data" / "datasub"
LEVELS = np.arange(1, 6, dtype=float)  # stimulus positions along the b→d continuum
TRIALS_PER_CELL = 24.0


@dataclass(frozen=True)
class Trial:
    audio: float | None  # stimulus level (1..5) or None if absent
    visual: float | None  # stimulus level (1..5) or None if absent
    successes: float  # number of "d" responses
    total: float  # number of trials (24 here)
    row_id: int  # which matrix row the observation came from (for CV splits)


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


# --- Model definitions -----------------------------------------------------

def early_strong_fusion(params: np.ndarray, audio: float | None, visual: float | None) -> float:
    """Fused at feature level: one logistic on weighted sum of audio and visual cues."""

    bias, beta_a, beta_v = params
    a = audio if audio is not None else 0.0
    v = visual if visual is not None else 0.0
    return float(logistic(bias + beta_a * a + beta_v * v))


def prob_match_fusion(params: np.ndarray, audio: float | None, visual: float | None) -> float:
    """Probability matching: combine log-odds from separate unimodal channels."""

    bias_a, slope_a, bias_v, slope_v, bias_av = params
    if audio is not None:
        p_a = logistic(bias_a + slope_a * audio)
    else:
        p_a = None
    if visual is not None:
        p_v = logistic(bias_v + slope_v * visual)
    else:
        p_v = None

    if p_a is not None and p_v is None:
        return float(p_a)
    if p_v is not None and p_a is None:
        return float(p_v)

    logit_a = np.log(p_a / (1 - p_a))
    logit_v = np.log(p_v / (1 - p_v))
    return float(logistic(bias_av + logit_a + logit_v))


def late_strong_fusion(params: np.ndarray, audio: float | None, visual: float | None) -> float:
    """Late fusion: mix unimodal probabilities with a fixed weight w."""

    bias_a, slope_a, bias_v, slope_v, logit_w = params
    w = logistic(logit_w)  # convert to 0-1 weight

    if audio is not None:
        p_a = logistic(bias_a + slope_a * audio)
    else:
        p_a = None
    if visual is not None:
        p_v = logistic(bias_v + slope_v * visual)
    else:
        p_v = None

    if p_a is not None and p_v is None:
        return float(p_a)
    if p_v is not None and p_a is None:
        return float(p_v)

    return float(w * p_a + (1.0 - w) * p_v)


# --- Data handling ---------------------------------------------------------

def load_subject(path: Path) -> List[Trial]:
    """Parse one subject matrix into a flat list of Trial entries."""

    mat = np.loadtxt(path)
    if mat.shape != (7, 5):
        raise ValueError(f"Expected 7x5 matrix, got {mat.shape} for {path}")

    trials: List[Trial] = []

    # Row 0: auditory-only (columns = auditory level)
    for col, audio in enumerate(LEVELS):
        trials.append(
            Trial(audio=audio, visual=None, successes=mat[0, col], total=TRIALS_PER_CELL, row_id=0)
        )

    # Row 1: visual-only (columns = visual level)
    for col, visual in enumerate(LEVELS):
        trials.append(
            Trial(audio=None, visual=visual, successes=mat[1, col], total=TRIALS_PER_CELL, row_id=1)
        )

    # Rows 2-6: audiovisual (row encodes visual level, column auditory level)
    for r in range(2, 7):
        visual = LEVELS[r - 2]
        for c, audio in enumerate(LEVELS):
            trials.append(
                Trial(audio=audio, visual=visual, successes=mat[r, c], total=TRIALS_PER_CELL, row_id=r)
            )

    return trials


# --- Likelihoods and fitting ----------------------------------------------

def negative_log_likelihood(
    params: np.ndarray, model: Callable[[np.ndarray, float | None, float | None], float], data: Iterable[Trial]
) -> float:
    """Binomial NLL across all trials."""

    eps = 1e-9
    nll = 0.0
    for t in data:
        p = np.clip(model(params, t.audio, t.visual), eps, 1.0 - eps)
        nll -= t.successes * np.log(p) + (t.total - t.successes) * np.log(1.0 - p)
    return float(nll)


def fit_model(
    model: Callable[[np.ndarray, float | None, float | None], float],
    x0: np.ndarray,
    data: Iterable[Trial],
) -> np.ndarray:
    """Optimise NLL; returns best-fit parameters."""

    res = minimize(
        fun=lambda p: negative_log_likelihood(p, model, data),
        x0=x0,
        method="L-BFGS-B",
    )
    if not res.success:
        print(f"Warning: optimisation did not converge ({res.message})")
    return res.x


def cross_validate_row_out(
    model: Callable[[np.ndarray, float | None, float | None], float],
    x0: np.ndarray,
    data: List[Trial],
) -> float:
    """7-fold CV: leave one matrix row out (audio-only, visual-only, or an AV visual level)."""

    row_ids = sorted({t.row_id for t in data})
    total_nll = 0.0
    for rid in row_ids:
        train = [t for t in data if t.row_id != rid]
        val = [t for t in data if t.row_id == rid]
        params = fit_model(model, x0, train)
        total_nll += negative_log_likelihood(params, model, val)
    return total_nll


# --- Experiment runner -----------------------------------------------------

def run_all_observers() -> None:
    subject_files = sorted(DATA_DIR.glob("DataSub*.txt"))
    if not subject_files:
        raise FileNotFoundError(f"No DataSub*.txt files found in {DATA_DIR}")

    models: Dict[str, Tuple[Callable, np.ndarray]] = {
        "early_strong_fusion": (early_strong_fusion, np.array([-5.0, 2.0, 2.0])),
        "prob_match_fusion": (prob_match_fusion, np.array([-5.0, 2.0, -5.0, 2.0, 0.0])),
        "late_strong_fusion": (late_strong_fusion, np.array([-5.0, 2.0, -5.0, 2.0, 0.0])),
    }

    for path in subject_files:
        subj = path.stem
        trials = load_subject(path)
        print(f"\n=== {subj} ===")
        for name, (model_fn, x0) in models.items():
            params = fit_model(model_fn, x0, trials)
            full_nll = negative_log_likelihood(params, model_fn, trials)
            cv_nll = cross_validate_row_out(model_fn, x0, trials)
            print(f"{name:22s} | NLL: {full_nll:8.2f} | CV NLL: {cv_nll:8.2f} | params: {params}")


if __name__ == "__main__":
    run_all_observers()
