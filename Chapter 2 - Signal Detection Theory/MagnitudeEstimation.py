import numpy as np
import matplotlib.pyplot as plt

# Stevens' power law parameters for two modalities
STEVENS_CASES = {
    "brightness_like (a=0.33)": 0.33,  # exponent < 1
    "electric_shock_like (a=3.3)": 3.3,  # exponent > 1
}

def simulate_stevens(is_levels, a, k=10.0):
    """Simulate perceived intensity via Stevens' law: I_p = k * I_s^a."""
    return k * (is_levels ** a)

def fit_fechner(is_levels, ip_values):
    """Fit Fechner's law: I_p = m * log(I_s) + b via linear regression."""
    x = np.log(is_levels)
    A = np.column_stack([x, np.ones_like(x)])
    params, *_ = np.linalg.lstsq(A, ip_values, rcond=None)
    m, b = params
    return m, b

def run_case(label, exponent, is_levels):
    ip_sim = simulate_stevens(is_levels, exponent)
    m, b = fit_fechner(is_levels, ip_sim)
    ip_fit = m * np.log(is_levels) + b

    print(f"{label}: Stevens exponent a={exponent}")
    print(f"  Fechner fit: slope={m:.3f}, intercept={b:.3f}")

    plt.figure(figsize=(6, 4))
    plt.scatter(is_levels, ip_sim, color="black", label="Simulated (Stevens)")
    plt.plot(is_levels, ip_fit, color="tomato", label="Fechner fit")
    plt.xlabel("Physical intensity I_s")
    plt.ylabel("Perceived intensity I_p")
    plt.title(f"{label}: Stevens vs. Fechner fit")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    is_levels = np.arange(1, 11, dtype=float)
    for label, a in STEVENS_CASES.items():
        run_case(label, a, is_levels)

if __name__ == "__main__":
    main()
