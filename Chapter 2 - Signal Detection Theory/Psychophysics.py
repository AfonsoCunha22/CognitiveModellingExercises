import random
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, mannwhitneyu
import tkinter as tk
from tkinter import ttk

# Paths to image sets (relative to this script)
DATA_ROOT = Path(__file__).resolve().parent / "data"
IMAGE_SETS = {
    "age": DATA_ROOT / "age",
    "sly": DATA_ROOT / "sly",
}


def list_images(set_name):
    folder = IMAGE_SETS[set_name]
    return sorted(folder.glob("*.png"))


def run_rating_session(set_name, repeats=10, output_csv=None):
    """GUI rating session: show images randomly (no filenames), collect 1-10 ratings."""
    imgs = list_images(set_name)
    trials = []
    for img in imgs:
        for _ in range(repeats):
            trials.append(img)
    random.shuffle(trials)

    responses = {img.name: [] for img in imgs}

    root = tk.Tk()
    root.title(f"Rate images: {set_name}")

    img_label = ttk.Label(root, text="")
    img_label.pack(pady=5)

    canvas = tk.Label(root)
    canvas.pack()

    rating_var = tk.IntVar(value=5)
    scale = ttk.Scale(root, from_=1, to=10, orient="horizontal", variable=rating_var)
    scale.pack(fill="x", padx=20, pady=10)
    scale_label = ttk.Label(root, text="Use slider (1-10), then click Next")
    scale_label.pack()

    status = ttk.Label(root, text="")
    status.pack(pady=5)

    photo_cache = {"img": None}
    idx = {"i": 0}

    def show_current():
        i = idx["i"]
        if i >= len(trials):
            root.quit()
            return
        path = trials[i]
        img_label.config(text=f"Image {i+1} of {len(trials)}")
        # Hide filename; show only anonymized index
        photo = tk.PhotoImage(file=str(path))
        photo_cache["img"] = photo
        canvas.config(image=photo)
        rating_var.set(5)
        status.config(text="")

    def on_next():
        i = idx["i"]
        if i >= len(trials):
            return
        path = trials[i]
        responses[path.name].append(rating_var.get())
        idx["i"] += 1
        if idx["i"] >= len(trials):
            status.config(text="Finished. Closing window...")
            root.after(500, root.quit)
        else:
            show_current()

    next_btn = ttk.Button(root, text="Next", command=on_next)
    next_btn.pack(pady=10)

    show_current()
    root.mainloop()
    root.destroy()

    df = pd.DataFrame(
        [[name, *responses[name]] for name in imgs],
        columns=["image"] + [f"r{i+1}" for i in range(repeats)],
    )
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved ratings to {output_csv}")
    return df


def synthesize_ratings(set_name, repeats=10, noise_sd=0.8, seed=123):
    """
    Create a synthetic rating dataset with variability:
    - Base latent score = ordinal index (1..N)
    - Add Gaussian noise
    - Clip to [1,10]
    """
    rng = np.random.default_rng(seed)
    imgs = list_images(set_name)
    n = len(imgs)
    bases = np.arange(1, n + 1, dtype=float)
    responses = {}
    for base, img in zip(bases, imgs):
        noisy = rng.normal(loc=base, scale=noise_sd, size=repeats)
        clipped = np.clip(noisy, 1, 10)
        responses[img.name] = clipped.tolist()

    df = pd.DataFrame(
        [[name, *responses[name]] for name in responses],
        columns=["image"] + [f"r{i+1}" for i in range(repeats)],
    )
    return df


def load_ratings(csv_path):
    df = pd.read_csv(csv_path)
    rating_cols = [c for c in df.columns if c.startswith("r")]
    ratings = df[rating_cols].values
    return df["image"].tolist(), ratings


def binarize_ratings(ratings, criterion=None):
    """Convert continuous ratings to binary yes/no using a criterion (default: global median)."""
    if criterion is None:
        criterion = np.nanmedian(ratings)
    binary = (ratings > criterion).astype(int)
    return binary, criterion


def loglinear_rates(hits, misses, fas, crs):
    h = (hits + 0.5) / (hits + misses + 1)
    f = (fas + 0.5) / (fas + crs + 1)
    return h, f


def d_prime_from_counts(hits, misses, fas, crs):
    h, f = loglinear_rates(hits, misses, fas, crs)
    return norm.ppf(h) - norm.ppf(f)


def pair_counts(binary, idx_a, idx_b):
    """Treat idx_a as signal and idx_b as noise."""
    a = binary[idx_a]
    b = binary[idx_b]
    hits = (a == 1).sum()
    misses = (a == 0).sum()
    fas = (b == 1).sum()
    crs = (b == 0).sum()
    return hits, misses, fas, crs


def compute_pairwise_dprime(binary, images, neighbours_only=True):
    n = len(images)
    pairs = []
    if neighbours_only:
        pairs = [(i, i + 1) for i in range(n - 1)]
    else:
        pairs = list(combinations(range(n), 2))
    results = []
    for i, j in pairs:
        hits, misses, fas, crs = pair_counts(binary, i, j)
        dp = d_prime_from_counts(hits, misses, fas, crs)
        results.append({"pair": (images[i], images[j]), "d_prime": dp})
    return results


def auc_from_ratings(ratings_a, ratings_b):
    """AUC = P(r_a > r_b) using Mann-Whitney U / rank-based AUC."""
    u_stat, _ = mannwhitneyu(ratings_a, ratings_b, alternative="two-sided")
    n1 = len(ratings_a)
    n2 = len(ratings_b)
    return u_stat / (n1 * n2)


def compute_pairwise_auc(ratings, images, neighbours_only=True):
    n = len(images)
    pairs = []
    if neighbours_only:
        pairs = [(i, i + 1) for i in range(n - 1)]
    else:
        pairs = list(combinations(range(n), 2))
    results = []
    for i, j in pairs:
        auc = auc_from_ratings(ratings[i], ratings[j])
        results.append({"pair": (images[i], images[j]), "auc": auc})
    return results


def summarize_results(dprimes, aucs, label):
    print(f"=== {label} ===")
    if dprimes:
        vals = [d["d_prime"] for d in dprimes]
        print(f"d' (n={len(vals)}): mean={np.mean(vals):.3f}, sd={np.std(vals, ddof=1):.3f}")
        for d in dprimes:
            print(f"  {d['pair'][0]} vs {d['pair'][1]}: d'={d['d_prime']:.3f}")
    if aucs:
        vals = [a["auc"] for a in aucs]
        print(f"AUC (n={len(vals)}): mean={np.mean(vals):.3f}, sd={np.std(vals, ddof=1):.3f}")
        for a in aucs:
            print(f"  {a['pair'][0]} vs {a['pair'][1]}: AUC={a['auc']:.3f}")


def main():
    # Choose which set to rate/analyze (safe defaults if stdin not available)
    try:
        choice = input("Choose image set (age/sly) [age]: ").strip().lower()
    except EOFError:
        choice = ""
    set_name = choice if choice in IMAGE_SETS else "age"
    csv_path = DATA_ROOT / f"{set_name}_ratings.csv"

    try:
        mode = input("Mode: [s]ynthetic / [i]nteractive / [l]oad existing? [s]: ").strip().lower() or "s"
    except EOFError:
        mode = "s"

    if mode.startswith("l"):
        if csv_path.exists():
            images, ratings = load_ratings(csv_path)
            print(f"Loaded ratings from {csv_path}")
        else:
            print("No ratings file found. Switch to synthetic.")
            mode = "s"

    if mode.startswith("i"):
        df = run_rating_session(set_name, repeats=10, output_csv=csv_path)
        images = df["image"].tolist()
        ratings = df[[c for c in df.columns if c.startswith("r")]].values
    elif mode.startswith("s"):
        df = synthesize_ratings(set_name, repeats=10, noise_sd=0.8, seed=123)
        df.to_csv(csv_path, index=False)
        print(f"Synthesized ratings saved to {csv_path}")
        images = df["image"].tolist()
        ratings = df[[c for c in df.columns if c.startswith("r")]].values
    else:
        # default to load existing if available
        if csv_path.exists():
            images, ratings = load_ratings(csv_path)
            print(f"Loaded ratings from {csv_path}")
        else:
            print("No ratings file found. Generating synthetic ratings.")
            df = synthesize_ratings(set_name, repeats=10, noise_sd=0.8, seed=123)
            df.to_csv(csv_path, index=False)
            images = df["image"].tolist()
            ratings = df[[c for c in df.columns if c.startswith("r")]].values

    # Binarize ratings (criterion = global median)
    binary, crit = binarize_ratings(ratings)
    print(f"Binarization criterion: {crit:.3f} (global median)")

    # d' and AUC for neighbouring pairs
    dprimes_neigh = compute_pairwise_dprime(binary, images, neighbours_only=True)
    aucs_neigh = compute_pairwise_auc(ratings, images, neighbours_only=True)
    summarize_results(dprimes_neigh, aucs_neigh, label=f"{set_name} (neighbours)")

    # Optional: d' and AUC for all pairs (may be unstable if responses are extreme)
    dprimes_all = compute_pairwise_dprime(binary, images, neighbours_only=False)
    aucs_all = compute_pairwise_auc(ratings, images, neighbours_only=False)
    summarize_results(dprimes_all, aucs_all, label=f"{set_name} (all pairs)")


if __name__ == "__main__":
    main()
