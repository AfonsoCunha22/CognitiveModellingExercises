"""Strong fusion model for continuous responses (ventriloquist exercise).

Steps covered in code:
1) Visualise data with probability density histograms and Gaussian fits.
2) Fit the strong fusion model (six free parameters) by maximising likelihood.
3) Check for over-fitting with 8-fold leave-one-stimulus-out cross-validation.

Run directly to reproduce the figures and parameter estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


# File with 8 rows (stimuli) x 20 columns (responses in degrees).
DATA_PATH = (
    Path(__file__).resolve().parent / "data" / "VentriloquistExercise4_5_5.txt"
)


@dataclass(frozen=True)
class Stimulus:
    """Keep stimulus metadata together."""

    audio_angle: float | None  # None denotes "no auditory stimulus"
    visual_angle: float | None  # None denotes "no visual stimulus"
    label: str


# Mapping from row index to the stimulus combination described in the exercise.
# Row numbers follow the table in the statement: 1..8 -> 0..7 here.
STIMULI: Dict[int, Stimulus] = {
    0: Stimulus(audio_angle=0, visual_angle=None, label="A0"),
    1: Stimulus(audio_angle=30, visual_angle=None, label="A30"),
    2: Stimulus(audio_angle=None, visual_angle=0, label="V0"),
    3: Stimulus(audio_angle=None, visual_angle=30, label="V30"),
    4: Stimulus(audio_angle=0, visual_angle=0, label="A0V0"),
    5: Stimulus(audio_angle=0, visual_angle=30, label="A0V30"),
    6: Stimulus(audio_angle=30, visual_angle=0, label="A30V0"),
    7: Stimulus(audio_angle=30, visual_angle=30, label="A30V30"),
}


def load_responses(path: Path = DATA_PATH) -> Dict[int, np.ndarray]:
    """Load responses; return a dict keyed by stimulus id."""

    data = np.loadtxt(path)
    if data.shape[0] != 8:
        raise ValueError(f"Expected 8 stimulus rows, found {data.shape[0]}")
    return {idx: row for idx, row in enumerate(data)}


def basic_stats(responses: Dict[int, np.ndarray]) -> Dict[int, Tuple[float, float]]:
    """Compute sample mean and sd for each stimulus."""

    return {
        stim: (vals.mean(), vals.std(ddof=1)) for stim, vals in responses.items()
    }


def plot_histograms(
    responses: Dict[int, np.ndarray],
    stats: Dict[int, Tuple[float, float]],
    title: str,
    model_params: np.ndarray | Dict[int, np.ndarray] | None = None,
) -> None:
    """Histogram per stimulus; optionally overlay a Gaussian model curve.

    model_params:
        - None -> only data histograms.
        - np.ndarray -> one parameter set for all stimuli (full-data fit).
        - dict[int, np.ndarray] -> per-stimulus parameter sets (CV predictions).
    """

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=False)
    axes = axes.ravel()

    for stim_id, ax in enumerate(axes):
        stim = STIMULI[stim_id]
        vals = responses[stim_id]
        mu, sd = stats[stim_id]

        # Probability density histogram (height = proportion / bin width).
        bins = np.arange(vals.min() - 2, vals.max() + 3, 2)
        ax.hist(vals, bins=bins, density=True, histtype="step", lw=2, color="black")

        # Sample Gaussian overlay from raw data.
        x = np.linspace(bins[0], bins[-1], 200)
        ax.plot(x, norm.pdf(x, mu, sd), "k:", label="sample Gaussian")

        # Optional model prediction overlay.
        if model_params is not None:
            params_for_stim = (
                model_params[stim_id]
                if isinstance(model_params, dict)
                else model_params
            )
            model_mu, model_sd = stimulus_prediction(params_for_stim, stim_id)
            ax.plot(
                x,
                norm.pdf(x, model_mu, model_sd),
                "r-",
                lw=2,
                label="model fit" if stim_id == 0 else None,
            )

        ax.set_title(stim.label)
        ax.set_xlabel("Azimuth angle (deg)")
        ax.set_ylabel("Probability density")
        ax.axvline(0, color="gray", lw=0.5)

    fig.suptitle(title)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    plt.show()


def stimulus_prediction(params: np.ndarray, stim_id: int) -> Tuple[float, float]:
    """Return (mean, sd) predicted by the strong fusion model for one stimulus."""

    mu_a0, mu_a30, mu_v0, mu_v30, sigma_a, sigma_v = params
    if sigma_a <= 0 or sigma_v <= 0:
        raise ValueError("Sigmas must be positive.")

    stim = STIMULI[stim_id]

    # Unimodal cases: pick the mean/variance of the presented modality.
    if stim.audio_angle is not None and stim.visual_angle is None:
        mu = mu_a0 if stim.audio_angle == 0 else mu_a30
        return mu, sigma_a
    if stim.visual_angle is not None and stim.audio_angle is None:
        mu = mu_v0 if stim.visual_angle == 0 else mu_v30
        return mu, sigma_v

    # Audiovisual: reliability-weighted average of auditory and visual estimates.
    mu_a = mu_a0 if stim.audio_angle == 0 else mu_a30
    mu_v = mu_v0 if stim.visual_angle == 0 else mu_v30
    precision_a = 1.0 / (sigma_a**2)
    precision_v = 1.0 / (sigma_v**2)
    combined_mu = (precision_a * mu_a + precision_v * mu_v) / (
        precision_a + precision_v
    )
    combined_sd = np.sqrt(1.0 / (precision_a + precision_v))
    return combined_mu, combined_sd


def negative_log_likelihood(
    params: np.ndarray, responses: Dict[int, np.ndarray]
) -> float:
    """Sum negative log likelihood over all stimulus/response pairs."""

    sigma_a, sigma_v = params[-2], params[-1]
    if sigma_a <= 0 or sigma_v <= 0:
        return np.inf  # invalid region

    nll = 0.0
    for stim_id, vals in responses.items():
        mu, sd = stimulus_prediction(params, stim_id)
        nll -= norm.logpdf(vals, loc=mu, scale=sd).sum()
    return nll


def initial_guess(responses: Dict[int, np.ndarray]) -> np.ndarray:
    """Derive sensible starting values from sample means/SDs.

    Uses all available stimuli for each modality/location so that it still works
    when one stimulus is held out during cross-validation.
    """

    def mean_for(modality: str, angle: float) -> float:
        collected: list[np.ndarray] = []
        for stim_id, vals in responses.items():
            stim = STIMULI[stim_id]
            if modality == "a" and stim.audio_angle == angle:
                collected.append(vals)
            if modality == "v" and stim.visual_angle == angle:
                collected.append(vals)
        if not collected:
            raise ValueError(f"No data for modality {modality} angle {angle}")
        return np.concatenate(collected).mean()

    def sd_for_modality(modality: str) -> float:
        collected: list[np.ndarray] = []
        for stim_id, vals in responses.items():
            stim = STIMULI[stim_id]
            if modality == "a" and stim.audio_angle is not None:
                collected.append(vals)
            if modality == "v" and stim.visual_angle is not None:
                collected.append(vals)
        if not collected:
            raise ValueError(f"No data for modality {modality}")
        return np.concatenate(collected).std(ddof=1)

    mu_a0 = mean_for("a", 0)
    mu_a30 = mean_for("a", 30)
    mu_v0 = mean_for("v", 0)
    mu_v30 = mean_for("v", 30)
    sigma_a = max(sd_for_modality("a"), 1e-3)
    sigma_v = max(sd_for_modality("v"), 1e-3)
    return np.array([mu_a0, mu_a30, mu_v0, mu_v30, sigma_a, sigma_v])


def fit_strong_fusion(responses: Dict[int, np.ndarray]):
    """Fit model parameters by minimising negative log likelihood."""

    x0 = initial_guess(responses)
    bounds = [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (1e-3, None),  # sigma_a
        (1e-3, None),  # sigma_v
    ]

    result = minimize(
        fun=lambda p: negative_log_likelihood(p, responses),
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
    )
    return result


def cross_validate(
    responses: Dict[int, np.ndarray],
) -> Tuple[float, Dict[int, np.ndarray]]:
    """8-fold leave-one-stimulus-out cross-validation."""

    total_val_nll = 0.0
    params_by_stim: Dict[int, np.ndarray] = {}

    for holdout in responses:
        train = {k: v for k, v in responses.items() if k != holdout}
        fit = fit_strong_fusion(train)
        params_by_stim[holdout] = fit.x

        val_nll = negative_log_likelihood(fit.x, {holdout: responses[holdout]})
        total_val_nll += val_nll
        print(
            f"Fold {holdout+1} ({STIMULI[holdout].label}) - "
            f"train status: {fit.message}, validation NLL: {val_nll:.2f}"
        )

    return total_val_nll, params_by_stim


def main() -> None:
    responses = load_responses()
    stats = basic_stats(responses)

    print("Sample means/SDs per stimulus:")
    for stim_id, (mu, sd) in stats.items():
        print(f"{stim_id+1}: {STIMULI[stim_id].label:5s} mu={mu:6.2f}, sigma={sd:6.2f}")

    # 1) Visualise raw data with sample Gaussians.
    plot_histograms(responses, stats, title="Observed data (sample Gaussian overlays)")

    # 2) Fit strong fusion model on all data.
    full_fit = fit_strong_fusion(responses)
    print("\nFull-data fit (strong fusion, 6 parameters):")
    print(f"Success: {full_fit.success}, message: {full_fit.message}")
    param_names = ["mu_a0", "mu_a30", "mu_v0", "mu_v30", "sigma_a", "sigma_v"]
    for name, val in zip(param_names, full_fit.x):
        print(f"  {name:7s} = {val:8.3f}")
    print(f"Negative log likelihood: {full_fit.fun:.2f}")

    # Plot model overlay using fitted parameters.
    plot_histograms(
        responses,
        stats,
        title="Strong fusion model fit",
        model_params=full_fit.x,
    )

    # 3) Cross-validation to check over-fitting.
    cv_total_nll, params_by_stim = cross_validate(responses)
    print(f"\nTotal cross-validation NLL (8 folds): {cv_total_nll:.2f}")

    # Plot CV predictions: each histogram uses the parameters from the fold
    # where that stimulus was held out.
    plot_histograms(
        responses,
        stats,
        title="Cross-validated predictions (per-stimulus parameters)",
        model_params=params_by_stim,
    )

    print(
        "\nIf the visual/auditory variances differ noticeably and the AV means "
        "shift toward the more reliable modality, that indicates enhancement "
        "and the ventriloquist illusion."
    )


if __name__ == "__main__":
    main()
