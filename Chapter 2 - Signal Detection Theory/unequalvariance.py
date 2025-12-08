import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

N_EXPERIMENTS = 100
N_SIGNAL_TRIALS = 50
N_NOISE_TRIALS = 50
MU_SIGNAL = 1.0
SIGMA_SIGNAL = 0.8  # unequal variance vs. noise sd = 1

# Decision criteria for four ordered confidence categories:
# noise high, noise low, yes low, yes high
CRITERIA = [-0.6, 0.0, 0.6]


def simulate_confidence_responses(
    rng,
    criteria=CRITERIA,
    mu_signal=MU_SIGNAL,
    sigma_signal=SIGMA_SIGNAL,
    n_signal=N_SIGNAL_TRIALS,
    n_noise=N_NOISE_TRIALS,
):
    """Simulate latent evidence and map it to 4 confidence categories."""
    evidence_signal = rng.normal(loc=mu_signal, scale=sigma_signal, size=n_signal)
    evidence_noise = rng.normal(loc=0.0, scale=1.0, size=n_noise)
    bins = [-np.inf, *criteria, np.inf]
    signal_ratings = np.digitize(evidence_signal, bins=bins) - 1  # 0..3
    noise_ratings = np.digitize(evidence_noise, bins=bins) - 1
    return signal_ratings, noise_ratings


def rating_counts(signal_ratings, noise_ratings, n_levels=4):
    """Return counts per rating level for signal and noise."""
    sig_counts = np.bincount(signal_ratings, minlength=n_levels)
    noise_counts = np.bincount(noise_ratings, minlength=n_levels)
    return sig_counts, noise_counts


def cumulative_rates(sig_counts, noise_counts):
    """Compute cumulative hit/FA rates from most liberal to most conservative."""
    sig_total = sig_counts.sum()
    noise_total = noise_counts.sum()
    # thresholds at each boundary from high to low
    sig_cum = np.cumsum(sig_counts[::-1])[::-1]
    noise_cum = np.cumsum(noise_counts[::-1])[::-1]
    # drop the last element (all responses) to get k-1 ROC points
    sig_cum = sig_cum[:-1]
    noise_cum = noise_cum[:-1]
    # log-linear correction to avoid 0/1
    hit_rates = (sig_cum + 0.5) / (sig_total + 1)
    fa_rates = (noise_cum + 0.5) / (noise_total + 1)
    return hit_rates, fa_rates


def fit_unequal_variance_params(hit_rates, fa_rates):
    """Fit zROC slope (s) and intercept (d_a) via linear regression."""
    zH = norm.ppf(hit_rates)
    zF = norm.ppf(fa_rates)
    slope, intercept = np.polyfit(zF, zH, 1)
    s_hat = slope
    d_a_hat = intercept
    return d_a_hat, s_hat


def roc_auc(hit_rates, fa_rates):
    """Trapezoidal AUC including (0,0) and (1,1)."""
    fa = np.concatenate(([0.0], fa_rates, [1.0]))
    hr = np.concatenate(([0.0], hit_rates, [1.0]))
    # sort by fa in case of any monotonicity issues
    order = np.argsort(fa)
    fa = fa[order]
    hr = hr[order]
    return np.trapz(hr, fa)


def run_experiments():
    rng = np.random.default_rng(123)
    d_a_estimates = []
    s_estimates = []
    auc_estimates = []

    for _ in range(N_EXPERIMENTS):
        sig_r, noise_r = simulate_confidence_responses(rng)
        sig_counts, noise_counts = rating_counts(sig_r, noise_r)
        hit_rates, fa_rates = cumulative_rates(sig_counts, noise_counts)
        d_a_hat, s_hat = fit_unequal_variance_params(hit_rates, fa_rates)
        auc_hat = roc_auc(hit_rates, fa_rates)

        d_a_estimates.append(d_a_hat)
        s_estimates.append(s_hat)
        auc_estimates.append(auc_hat)

    return (
        np.array(d_a_estimates),
        np.array(s_estimates),
        np.array(auc_estimates),
    )


def plot_distributions(d_a_vals, s_vals, auc_vals):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(d_a_vals, bins=15, color="#4682b4", edgecolor="black", alpha=0.8)
    axes[0].axvline(MU_SIGNAL, color="tomato", linestyle="--", label="true d_a")
    axes[0].set_title("d_a estimates")
    axes[0].legend()

    axes[1].hist(s_vals, bins=15, color="#6c9f58", edgecolor="black", alpha=0.8)
    axes[1].axvline(SIGMA_SIGNAL, color="tomato", linestyle="--", label="true s")
    axes[1].set_title("slope (s) estimates")
    axes[1].legend()

    axes[2].hist(auc_vals, bins=15, color="#cc7a00", edgecolor="black", alpha=0.8)
    true_auc = norm.cdf(MU_SIGNAL / np.sqrt(1 + SIGMA_SIGNAL ** 2))
    axes[2].axvline(true_auc, color="tomato", linestyle="--", label="true AUC")
    axes[2].set_title("AUC estimates")
    axes[2].legend()

    for ax in axes:
        ax.set_ylabel("count")
    fig.tight_layout()
    plt.show()
    return fig


def main():
    d_a_vals, s_vals, auc_vals = run_experiments()
    true_auc = norm.cdf(MU_SIGNAL / np.sqrt(1 + SIGMA_SIGNAL ** 2))
    print("Estimated parameters across 100 experiments (unequal variance, 4 ratings)")
    print(f"d_a: mean {d_a_vals.mean():.3f}, sd {d_a_vals.std(ddof=1):.3f}, true {MU_SIGNAL}")
    print(f"s (slope): mean {s_vals.mean():.3f}, sd {s_vals.std(ddof=1):.3f}, true {SIGMA_SIGNAL}")
    print(f"AUC: mean {auc_vals.mean():.3f}, sd {auc_vals.std(ddof=1):.3f}, true {true_auc:.3f}")
    plot_distributions(d_a_vals, s_vals, auc_vals)


if __name__ == "__main__":
    main()
