import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import norm

# Data: 3-AFC task, 30 trials per intensity
INTENSITIES = np.array([5, 10, 15, 20, 25, 30], dtype=float)
CORRECT_COUNTS = np.array([12, 11, 19, 27, 30, 30], dtype=float)
N_TRIALS = 30
P_GUESS = 1.0 / 3.0  # fixed guessing rate for 3-AFC


# Psychometric forms (Equations 3.15, 3.17, 3.19)
# Parameterization: alpha ~ c_I (threshold), beta ~ 1 / sigma_I (slope)
def psi_315(x, alpha, beta):
    return norm.cdf((x - alpha) * beta)


def psi_317(x, alpha, beta, p_guess=P_GUESS):
    core = norm.cdf((x - alpha) * beta)
    return (1 - p_guess) * core + p_guess


def psi_319(x, alpha, beta, lapse, p_guess=P_GUESS):
    core = norm.cdf((x - alpha) * beta)
    return (1 - p_guess - lapse) * core + p_guess


def nll(params, model, x, k, n, p_guess=P_GUESS):
    """Negative log likelihood for binomial counts."""
    # Unpack params depending on model
    if model == "315":
        alpha, beta = params
        lapse = 0.0
        if beta <= 0:
            return np.inf
        p = psi_315(x, alpha, beta)
    elif model == "317":
        alpha, beta = params
        lapse = 0.0
        if beta <= 0:
            return np.inf
        p = psi_317(x, alpha, beta, p_guess=p_guess)
    else:  # "319"
        alpha, beta, lapse = params
        if beta <= 0 or lapse < 0 or lapse > 0.5:
            return np.inf
        p = psi_319(x, alpha, beta, lapse, p_guess=p_guess)

    p = np.clip(p, 1e-6, 1 - 1e-6)
    log_lik = k * np.log(p) + (n - k) * np.log(1 - p)
    return -np.sum(log_lik)


def fit_model(model, x, k, n, initial):
    res = optimize.minimize(
        nll,
        x0=np.array(initial, dtype=float),
        args=(model, x, k, n),
        method="Nelder-Mead",
        options={"maxiter": 5000, "disp": False},
    )
    return res


def aic(nll_val, n_params):
    return 2 * n_params + 2 * nll_val


def summarize_fit(model, x, k, n):
    if model == "315":
        init = [np.median(x), 0.2]
        n_params = 2
    elif model == "317":
        init = [np.median(x), 0.2]
        n_params = 2  # p_guess is fixed at 1/3
    else:  # "319"
        init = [np.median(x), 0.2, 0.02]
        n_params = 3
    res = fit_model(model, x, k, n, init)
    params = res.x
    nll_val = nll(params, model, x, k, n)
    return {
        "model": model,
        "params": params,
        "nll": nll_val,
        "aic": aic(nll_val, n_params),
        "success": res.success,
    }


def evaluate_fits(fits, x_dense):
    curves = {}
    for f in fits:
        m = f["model"]
        if m == "315":
            alpha, beta = f["params"]
            curves[m] = psi_315(x_dense, alpha, beta)
        elif m == "317":
            alpha, beta = f["params"]
            curves[m] = psi_317(x_dense, alpha, beta, p_guess=P_GUESS)
        else:
            alpha, beta, lapse = f["params"]
            curves[m] = psi_319(x_dense, alpha, beta, lapse, p_guess=P_GUESS)
    return curves


def run_analysis(correct_counts, label):
    print(f"=== {label} ===")
    fits = [
        summarize_fit("315", INTENSITIES, correct_counts, N_TRIALS),
        summarize_fit("317", INTENSITIES, correct_counts, N_TRIALS),
        summarize_fit("319", INTENSITIES, correct_counts, N_TRIALS),
    ]
    fits_sorted = sorted(fits, key=lambda d: d["aic"])
    for f in fits_sorted:
        p = f["params"]
        if f["model"] == "319":
            alpha, beta, lapse = p
            extra = f", lapse={lapse:.3f}"
        else:
            alpha, beta = p
            extra = ""
        print(
            f"Model {f['model']}: nll={f['nll']:.3f}, AIC={f['aic']:.3f}, "
            f"alpha={alpha:.3f}, beta={beta:.3f}{extra}, success={f['success']}"
        )
    best = fits_sorted[0]
    print(f"Lowest AIC: model {best['model']}")

    # Plot
    x_dense = np.linspace(INTENSITIES.min() - 2, INTENSITIES.max() + 2, 300)
    curves = evaluate_fits(fits_sorted, x_dense)
    styles = {"315": {"linestyle": "--"}, "317": {"linestyle": "-"}, "319": {"linestyle": ":"}}
    plt.figure(figsize=(7, 4))
    for name, y in curves.items():
        plt.plot(x_dense, y, label=f"model {name}", **styles.get(name, {}))
    plt.scatter(INTENSITIES, correct_counts / N_TRIALS, color="black", zorder=5, label="data")
    plt.xlabel("Intensity (dB)")
    plt.ylabel("Proportion correct")
    plt.title(label)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Original data
    run_analysis(CORRECT_COUNTS, "Original data (3AFC)")
    # Lapse at highest intensity: one miss at 30 dB (29/30 correct)
    lapse_counts = CORRECT_COUNTS.copy()
    lapse_counts[-1] = 29
    run_analysis(lapse_counts, "With lapse at 30 dB (29/30)")


if __name__ == "__main__":
    main()
