# Generalized-Prony-IR

```python
#!/usr/bin/env python3
"""
Generalized Prony Method with automatic stabilization and export.

Usage:
    python gprony_wav_full.py impulse_response.wav --order 20 --plot --stabilize
"""

import argparse
import json
import csv
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.signal import freqz

# -------------------------------------------------------------
# 1. Load and preprocess WAV
# -------------------------------------------------------------
def load_impulse_response(path, normalize=True, window_length=None):
    x, fs = sf.read(path)

    # Mono if needed
    if hasattr(x, "ndim") and x.ndim > 1:
        x = x.mean(axis=1)

    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("Empty audio file or failed to read samples.")

    # Normalize safely
    peak = np.max(np.abs(x))
    if normalize and peak > 0:
        x = x / peak

    # Use max peak as IR start
    n0 = int(np.argmax(np.abs(x)))

    # Default: 100 ms window
    if window_length is None:
        window_length = int(0.1 * fs)

    end = min(n0 + window_length, len(x))
    if end - n0 < 4:
        raise ValueError(
            "Impulse window too short after peak detection. "
            "Check the input file or adjust window_length."
        )

    h = x[n0:end]
    return h, fs

# -------------------------------------------------------------
# 2. Hankel utilities
# -------------------------------------------------------------
def hankel_matrix(h, M):
    """
    Build Hankel matrix from impulse response h for order M.
    Shape: (L, M+1) with L = len(h) - M.
    """
    N = len(h)
    if M >= N:
        raise ValueError(f"Order M={M} must be < len(h)={N}.")
    L = N - M
    if L <= 0:
        raise ValueError("Not enough samples to build Hankel matrix.")
    H = np.empty((L, M + 1))
    for i in range(L):
        H[i, :] = h[i : i + M + 1]
    return H

# -------------------------------------------------------------
# 3. Order estimation & Prony solver
# -------------------------------------------------------------
def estimate_order(h, max_order=50, tol_ratio=1e-2):
    """
    Estimate model order from singular value gaps of Hankel matrix.
    tol_ratio: if s_k / s_{k+1} > 1/tol_ratio, pick k as order.
    """
    N = len(h)
    if N < 4:
        raise ValueError("Impulse response too short for order estimation.")

    max_order = int(min(max_order, N - 2))
    if max_order < 1:
        raise ValueError("max_order too small for given impulse length.")

    H = hankel_matrix(h, max_order)
    s = svd(H, compute_uv=False)

    ratios = s[:-1] / s[1:]
    threshold = 1.0 / tol_ratio

    idx = np.where(ratios > threshold)[0]
    if idx.size == 0:
        M_est = max(1, max_order // 2)
    else:
        M_est = int(idx[0] + 1)

    M_est = max(1, min(M_est, max_order))
    return M_est, s

def generalized_prony(h, M):
    """
    Solve generalized Prony using SVD of Hankel matrix.
    Returns polynomial p such that H p ≈ 0 and p[-1] = 1.
    """
    H = hankel_matrix(h, M)
    _, _, Vh = svd(H, full_matrices=False)
    p = Vh[-1, :]

    if np.isclose(p[-1], 0.0):
        raise RuntimeError("Degenerate Prony solution: last polynomial coefficient is ~0.")

    p = p / p[-1]
    return p

def difference_equation(p):
    """
    Convert Prony polynomial p (H p = 0, p[-1] = 1) to AR coefficients a
    for the all-pole filter:

        A(z) = 1 + a1 z^{-1} + ... + aM z^{-M}

    and the corresponding difference equation:

        h[n] + a1 h[n-1] + ... + aM h[n-M] = 0  (for n >= M)

    Mapping:
        a_m = p[M-m],  m = 1..M
    """
    M = len(p) - 1
    a = p[M-1::-1]   # [p[M-1], p[M-2], ..., p[0]]
    return a

# -------------------------------------------------------------
# 4. Stability handling
# -------------------------------------------------------------
def stabilize_poles(a, max_radius=0.98):
    """
    Reflect unstable poles inside circle of radius max_radius.

    Input:
        a : AR coefficients for A(z) = 1 + a1 z^{-1} + ... + aM z^{-M}
    Returns:
        a_stable    : stabilized AR coefficients
        roots       : original poles
        stable_roots: stabilized poles
    """
    roots = np.roots(np.r_[1.0, a])
    radii = np.abs(roots)

    stable_roots = roots.copy()
    mask = radii >= 1.0
    if np.any(mask):
        stable_roots[mask] = (max_radius / radii[mask]) * roots[mask]

    # Given poles z_k, A(z) = ∏ (1 - z_k z^{-1}),
    # so z^M A(z) = ∏ (z - z_k) = np.poly(z_k)
    # and [1, a1, ..., aM] = np.poly(stable_roots)
    poly_coeffs = np.poly(stable_roots)
    a_stable = poly_coeffs[1:]
    return a_stable, roots, stable_roots

# -------------------------------------------------------------
# 5. Reconstruction & metrics
# -------------------------------------------------------------
def reconstruct_from_diff_eq(h, a):
    """
    Reconstruct impulse response using AR coefficients a to match length of h.
    Uses first M samples of h as initial conditions:

        h_rec[n] = -sum_{k=1}^M a_k h_rec[n-k]
    """
    M = len(a)
    N = len(h)

    if N <= M:
        return np.array(h, copy=True)

    h_rec = np.zeros(N, dtype=float)
    h_rec[:M] = h[:M]

    for n in range(M, N):
        h_rec[n] = -np.dot(a, h_rec[n-M:n][::-1])

    return h_rec

def error_metrics(h, h_rec):
    """
    Return MSE and SNR(dB) with protection against NaN/Inf.
    """
    h = np.asarray(h, dtype=float)
    h_rec = np.asarray(h_rec, dtype=float)

    if h.shape != h_rec.shape:
        raise ValueError("Original and reconstructed responses must have same length.")

    if np.any(~np.isfinite(h_rec)):
        return {"MSE": float("inf"), "SNR_dB": float("-inf")}

    err = h - h_rec
    num = float(np.sum(h**2))
    den = float(np.sum(err**2))
    mse = den / len(h)

    if den == 0.0:
        snr = float("inf")
    elif num == 0.0:
        snr = float("-inf")
    else:
        snr = 10.0 * np.log10(num / den)

    return {"MSE": float(mse), "SNR_dB": float(snr)}

# -------------------------------------------------------------
# 6. Plots
# -------------------------------------------------------------
def plot_impulse_fit(h, h_rec):
    plt.figure()
    plt.plot(h, label="Measured")
    plt.plot(h_rec, "--", label="Reconstructed")
    plt.legend()
    plt.title("Impulse Response Fit")
    plt.xlabel("Sample")
    plt.grid(True, alpha=0.3)

def plot_pz(a):
    poles = np.roots(np.r_[1.0, a])
    plt.figure()
    plt.scatter(np.real(poles), np.imag(poles), color="r", label="Poles")
    uc = plt.Circle((0, 0), 1, color="black", fill=False, linestyle="--")
    ax = plt.gca()
    ax.add_artist(uc)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    plt.axis("equal")
    plt.title("Pole Locations")
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.legend()
    plt.grid(True, alpha=0.3)

def plot_freq_response(a, fs):
    w, h = freqz([1.0], np.r_[1.0, a], worN=2048)

    freqs = w * fs / (2.0 * np.pi)
    mag = np.abs(h)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(freqs, mag_db)
    plt.title("Magnitude Response (dB)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(freqs, np.unwrap(np.angle(h)))
    plt.title("Phase Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (rad)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

# -------------------------------------------------------------
# 7. Export results
# -------------------------------------------------------------
def export_results(a, poles, fs, metrics, prefix="gprony_result", raw_poles=None):
    data = {
        "sampling_rate": float(fs),
        "coefficients": [float(x) for x in a],
        "poles_real": [float(np.real(p)) for p in poles],
        "poles_imag": [float(np.imag(p)) for p in poles],
        "metrics": {
            k: (float(v) if isinstance(v, (int, float)) and np.isfinite(v) else str(v))
            for k, v in metrics.items()
        },
    }

    if raw_poles is not None:
        data["raw_poles_real"] = [float(np.real(p)) for p in raw_poles]
        data["raw_poles_imag"] = [float(np.imag(p)) for p in raw_poles]

    json_path = f"{prefix}.json"
    csv_path = f"{prefix}.csv"

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["a_index", "a_value"])
        for i, ai in enumerate(a, 1):
            writer.writerow([i, ai])

        writer.writerow([])
        writer.writerow(["Pole_real", "Pole_imag"])
        for p in poles:
            writer.writerow([np.real(p), np.imag(p)])

        if raw_poles is not None:
            writer.writerow([])
            writer.writerow(["Raw_pole_real", "Raw_pole_imag"])
            for p in raw_poles:
                writer.writerow([np.real(p), np.imag(p)])

    print(f"\n📁 Results exported to {json_path} and {csv_path}")

# -------------------------------------------------------------
# 8. Main pipeline
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", help="Path to impulse response WAV")
    parser.add_argument(
        "--order",
        type=int,
        default=None,
        help="Model order (if omitted, estimated automatically)",
    )
    parser.add_argument(
        "--maxorder",
        type=int,
        default=50,
        help="Maximum order for automatic estimation",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show diagnostic plots",
    )
    parser.add_argument(
        "--stabilize",
        action="store_true",
        help="Reflect unstable poles inside unit circle",
    )
    parser.add_argument(
        "--export_prefix",
        default="gprony_result",
        help="Prefix for exported JSON/CSV files",
    )
    args = parser.parse_args()

    # Load impulse response
    h, fs = load_impulse_response(args.wav)
    print(f"Loaded {args.wav} ({len(h)} samples @ {fs} Hz)")

    # Determine model order
    if args.order is None:
        M_est, _ = estimate_order(h, max_order=args.maxorder)
        print(f"Estimated model order ≈ {M_est}")
    else:
        M_est = int(args.order)
        if M_est < 1:
            raise ValueError("Order must be at least 1.")
        if M_est >= len(h):
            M_adj = len(h) - 2
            if M_adj < 1:
                raise ValueError(
                    f"Requested order {M_est} too large for impulse length {len(h)}."
                )
            print(
                f"Requested order {M_est} too high for data; using {M_adj} instead."
            )
            M_est = M_adj
        else:
            print(f"Using model order = {M_est}")

    # Generalized Prony solution
    p = generalized_prony(h, M_est)
    a = difference_equation(p)

    print("\nDifference-equation coefficients (a₁…a_M):")
    for i, ai in enumerate(a, 1):
        print(f"  a{i:02d} = {ai:.6g}")

    # Stability analysis on raw model
    raw_poles = np.roots(np.r_[1.0, a])
    magnitudes = np.abs(raw_poles)
    max_mag = float(magnitudes.max()) if magnitudes.size > 0 else 0.0

    print(f"\nMax |pole| = {max_mag:.3f}")

    if np.any(magnitudes >= 1.0):
        print("⚠️  Model unstable: some poles are on or outside the unit circle.")
        if args.stabilize:
            a_used, old_poles, stable_poles = stabilize_poles(a)
            poles = stable_poles
            print("✅  Poles reflected inside the unit circle.")
        else:
            a_used = a
            poles = raw_poles
            print("👉  Rerun with --stabilize to enforce stability, if desired.")
    else:
        a_used = a
        poles = raw_poles
        print("✅  All poles strictly inside unit circle.")

    # Reconstruction & evaluation
    h_rec = reconstruct_from_diff_eq(h, a_used)
    metrics = error_metrics(h, h_rec)

    print("\nReconstruction metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and np.isfinite(v):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    # Export results (include raw_poles for inspection)
    export_results(
        a_used,
        poles,
        fs,
        metrics,
        prefix=args.export_prefix,
        raw_poles=raw_poles,
    )

    # Plots
    if args.plot:
        plot_impulse_fit(h, h_rec)
        plot_pz(a_used)
        plot_freq_response(a_used, fs)
        plt.show()

if __name__ == "__main__":
    main()

```
