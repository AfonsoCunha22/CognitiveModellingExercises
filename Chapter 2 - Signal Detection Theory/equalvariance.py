import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

N_EXPERIMENTS = 100
N_SIGNAL_TRIALS = 50
N_NOISE_TRIALS = 50
SIGNAL_MEAN = 1.0


def simulate_observer(
    rng,
    criterion,
    n_experiments=N_EXPERIMENTS,
    n_signal=N_SIGNAL_TRIALS,
    n_noise=N_NOISE_TRIALS,
    d_prime=SIGNAL_MEAN,
    signal_sd=1.0,
):
    counts = []
    for _ in range(n_experiments):
        signal = rng.normal(loc=d_prime, scale=signal_sd, size=n_signal)
        noise = rng.normal(loc=0.0, scale=1.0, size=n_noise)

        hits = int(np.sum(signal > criterion))
        misses = n_signal - hits
        false_alarms = int(np.sum(noise > criterion))
        correct_rejections = n_noise - false_alarms

        counts.append((hits, misses, false_alarms, correct_rejections))
    return counts


def summarize_counts(counts):
    arr = np.array(counts, dtype=float)
    hit_rates = arr[:, 0] / (arr[:, 0] + arr[:, 1])
    false_alarm_rates = arr[:, 2] / (arr[:, 2] + arr[:, 3])
    yes_rates = (arr[:, 0] + arr[:, 2]) / (arr[:, 0] + arr[:, 1] + arr[:, 2] + arr[:, 3])

    return {
        "mean_hit_rate": hit_rates.mean(),
        "mean_false_alarm_rate": false_alarm_rates.mean(),
        "mean_yes_rate": yes_rates.mean(),
    }


def loglinear_rates(hits, misses, false_alarms, correct_rejections):
    """Apply log-linear correction to avoid 0/1 rates before z-transform."""
    hit_rate = (hits + 0.5) / (hits + misses + 1)
    false_alarm_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)
    return hit_rate, false_alarm_rate


def d_prime_from_counts(hits, misses, false_alarms, correct_rejections):
    hit_rate, false_alarm_rate = loglinear_rates(hits, misses, false_alarms, correct_rejections)
    return norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)


def compute_d_primes(counts):
    return np.array([d_prime_from_counts(*c) for c in counts])


def plot_histograms(dprime_by_observer):
    n_obs = len(dprime_by_observer)
    fig, axes = plt.subplots(1, n_obs, figsize=(4 * n_obs, 3.5), sharey=True)
    if n_obs == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, dprime_by_observer.items()):
        ax.hist(values, bins=15, color="#4682b4", edgecolor="black", alpha=0.8)
        ax.axvline(SIGNAL_MEAN, color="tomato", linestyle="--", linewidth=1.5, label="true d'")
        ax.set_title(f"{name}")
        ax.set_xlabel("d' estimate")
        ax.set_ylabel("count")
        ax.legend()

    fig.tight_layout()
    plt.show()
    return fig


def main():
    rng = np.random.default_rng(123)
    observers = {
        "yes_biased": -0.7,   # favors yes responses
        "no_biased": 0.7,     # favors no responses
        "mild_bias": 0.1,     # near-neutral criterion
    }

    def run_scenario(signal_sd, label):
        print(f"=== {label} (signal sd = {signal_sd}) ===")
        dprime_results = {}

        for name, criterion in observers.items():
            counts = simulate_observer(rng, criterion, signal_sd=signal_sd)
            summary = summarize_counts(counts)
            d_primes = compute_d_primes(counts)
            dprime_results[name] = d_primes

            print(f"{name} (c={criterion:+.1f})")
            print(f"  mean hit rate: {summary['mean_hit_rate']:.3f}")
            print(f"  mean false alarm rate: {summary['mean_false_alarm_rate']:.3f}")
            print(f"  mean yes rate: {summary['mean_yes_rate']:.3f}")
            print(
                "  first experiment counts (hits, misses, false alarms, correct rejections): "
                f"{counts[0]}"
            )
            print(f"  d' mean (100 experiments): {d_primes.mean():.3f} +/- {d_primes.std(ddof=1):.3f}")
            print()

        plot_histograms(dprime_results)

    # Equal-variance generating model
    run_scenario(signal_sd=1.0, label="Equal-variance generation")
    # Unequal-variance generating model (sigma = 0.8), still analyzed with equal-variance d'
    run_scenario(signal_sd=0.8, label="Unequal-variance generation (sd=0.8)")


if __name__ == "__main__":
    main()
