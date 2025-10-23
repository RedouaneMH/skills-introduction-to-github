# main.py
# DM954 Week 5 — Task 1: Linear Regression (manual + verification)
# Author: Redouane
# Strict, reproducible pipeline with clear outputs and figures.

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# -----------------------------
# A) Data I/O + Inspection
# -----------------------------
def load_raw(csv_path="dirty_regression_data.csv") -> pd.DataFrame:
    """
    Load the raw dataset produced by generate_data.py
    Expected columns: x, y
    """
    df = pd.read_csv(csv_path)
    # Soft sanity checks
    assert 'x' in df.columns and 'y' in df.columns, "CSV must contain columns 'x' and 'y'."
    return df


def inspect(df: pd.DataFrame, title="RAW"):
    print(f"\n=== INSPECT ({title}) ===")
    print("shape:", df.shape)
    print(df.describe(include='all'))
    print("missing x:", df['x'].isna().sum(), " | missing y:", df['y'].isna().sum())


# -----------------------------
# B) Cleaning
# -----------------------------
def clean_xy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where x or y is NaN. Do NOT alter values otherwise.
    This preserves the 'true' noise and outlier structure.
    """
    cleaned = df.dropna(subset=['x', 'y']).reset_index(drop=True)
    cleaned.to_csv("treated_set.dat", index=False)
    return cleaned


# -----------------------------
# C) Manual Linear Regression
# -----------------------------
def manual_means(x: np.ndarray, y: np.ndarray):
    xbar = float(np.mean(x))
    ybar = float(np.mean(y))
    return xbar, ybar


def manual_sums_centered(x: np.ndarray, y: np.ndarray, xbar: float, ybar: float):
    """
    Sxx = Σ (x - xbar)^2
    Sxy = Σ (x - xbar)(y - ybar)
    """
    dx = x - xbar
    dy = y - ybar
    Sxx = float(np.sum(dx * dx))
    Sxy = float(np.sum(dx * dy))
    return Sxx, Sxy


def manual_fit(x: np.ndarray, y: np.ndarray):
    """
    Compute slope b and intercept a with centered least-squares formulas:
      b = Sxy / Sxx
      a = ybar - b * xbar
    Returns (a, b, dict_of_intermediates)
    """
    n = int(len(x))
    assert n == len(y) and n >= 2

    xbar, ybar = manual_means(x, y)
    Sxx, Sxy = manual_sums_centered(x, y, xbar, ybar)

    # Guard: Sxx must be > 0 for identifiable slope
    if Sxx <= 0:
        raise ValueError("Sxx <= 0; cannot compute slope.")

    b = Sxy / Sxx
    a = ybar - b * xbar

    info = {
        "n": n,
        "xbar": xbar,
        "ybar": ybar,
        "Sxx": Sxx,
        "Sxy": Sxy
    }
    return a, b, info


# -----------------------------
# D) Metrics (R^2, SE, t, p)
# -----------------------------
def predictions(x: np.ndarray, a: float, b: float):
    return a + b * x


def regression_metrics(x: np.ndarray, y: np.ndarray, a: float, b: float, info: dict):
    """
    Compute:
      - yhat, residuals
      - SSR = Σ (y - yhat)^2
      - SST = Σ (y - ybar)^2
      - R^2 = 1 - SSR/SST
      - Residual variance: sigma2 = SSR / (n - 2)
      - SE_b = sqrt( sigma2 / Sxx )
      - t_b = b / SE_b
      - p-value (two-sided) via Student t with df = n - 2
    """
    n = info["n"]
    ybar = info["ybar"]
    Sxx = info["Sxx"]

    yhat = predictions(x, a, b)
    resid = y - yhat

    SSR = float(np.sum(resid ** 2))                   # Sum of Squared Residuals (a.k.a. SSE)
    SST = float(np.sum((y - ybar) ** 2))              # Total Sum of Squares
    R2 = float(1.0 - SSR / SST) if SST > 0 else float("nan")

    df = n - 2
    sigma2 = SSR / df                                  # unbiased residual variance
    SE_b = math.sqrt(sigma2 / Sxx)                     # standard error of slope
    t_b = b / SE_b                                     # t-stat for slope
    # two-sided p-value
    p_b = 2.0 * (1.0 - stats.t.cdf(abs(t_b), df=df))

    return {
        "yhat": yhat,
        "resid": resid,
        "SSR": SSR,
        "SST": SST,
        "R2": R2,
        "df": df,
        "sigma2": sigma2,
        "SE_b": SE_b,
        "t_b": t_b,
        "p_b": p_b
    }


# -----------------------------
# E) SciPy verification
# -----------------------------
def scipy_linregress(x: np.ndarray, y: np.ndarray):
    r = stats.linregress(x, y)
    return {
        "a": float(r.intercept),
        "b": float(r.slope),
        "R2": float(r.rvalue ** 2),
        "p_b": float(r.pvalue),
        "SE_b": float(r.stderr),
        "intercept_stderr": float(getattr(r, "intercept_stderr", np.nan))
    }


# -----------------------------
# F) Outliers (IQR on y) + Residual-based
# -----------------------------
def iqr_outlier_mask(series: pd.Series, k: float = 1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (series >= lo) & (series <= hi)


def residual_outlier_mask(resid: np.ndarray, k_sigma: float = 3.0):
    s = float(np.std(resid, ddof=1))
    if s == 0:
        return np.ones_like(resid, dtype=bool)
    return np.abs(resid) <= k_sigma * s


# -----------------------------
# G) Plot helpers
# -----------------------------
def plot_scatter(df: pd.DataFrame, title: str):
    plt.figure()
    plt.scatter(df['x'], df['y'])
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


def plot_with_fit(df: pd.DataFrame, a: float, b: float, title: str):
    xx = np.linspace(df['x'].min(), df['x'].max(), 200)
    yy = a + b * xx
    plt.figure()
    plt.scatter(df['x'], df['y'])
    plt.plot(xx, yy)
    plt.title(title + f"\nŷ = {a:.4f} + {b:.4f}x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


def plot_residuals(x: np.ndarray, resid: np.ndarray, title: str):
    plt.figure()
    plt.scatter(x, resid)
    plt.axhline(0)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("residual (y - ŷ)")
    plt.tight_layout()
    plt.show()


# -----------------------------
# H) Orchestrate
# -----------------------------
def main():
    # 1) Load + Inspect raw
    df_raw = load_raw("dirty_regression_data.csv")
    inspect(df_raw, "RAW")

    # Optional: quick raw plot
    plot_scatter(df_raw.dropna(subset=['x','y']), "Raw (non-NaN only)")

    # 2) Clean
    df = clean_xy(df_raw)
    inspect(df, "CLEANED")
    plot_scatter(df, "Cleaned Scatter")

    # 3) Manual fit
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    a, b, info = manual_fit(x, y)
    metrics = regression_metrics(x, y, a, b, info)

    # 4) SciPy
    sp = scipy_linregress(x, y)

    # 5) Outliers — IQR on y
    mask_iqr = iqr_outlier_mask(df['y'], k=1.5)
    df_iqr = df[mask_iqr].reset_index(drop=True)
    x_iqr, y_iqr = df_iqr['x'].to_numpy(), df_iqr['y'].to_numpy()
    a_iqr, b_iqr, info_iqr = manual_fit(x_iqr, y_iqr)
    metrics_iqr = regression_metrics(x_iqr, y_iqr, a_iqr, b_iqr, info_iqr)

    # 6) Outliers — residual-based on initial fit
    resid_mask = residual_outlier_mask(metrics["resid"], k_sigma=3.0)
    df_res = df.loc[resid_mask].reset_index(drop=True)
    x_res, y_res = df_res['x'].to_numpy(), df_res['y'].to_numpy()
    a_res, b_res, info_res = manual_fit(x_res, y_res)
    metrics_res = regression_metrics(x_res, y_res, a_res, b_res, info_res)

    # 7) Plots (model + residuals)
    plot_with_fit(df, a, b, "Manual Linear Regression (Cleaned)")
    plot_residuals(x, metrics["resid"], "Residuals (Initial Manual Fit)")

    # 8) Report to console (accurate, concise)
    print("\n================= REPORT (Console) =================")
    print("== DATA ==")
    print(f"Cleaned rows (x,y valid): {len(df)}")

    print("\n== MANUAL FIT ==")
    print(f"x̄ = {info['xbar']:.6f}, ȳ = {info['ybar']:.6f}")
    print(f"Sxx = {info['Sxx']:.6f}, Sxy = {info['Sxy']:.6f}")
    print(f"b (slope)   = {b:.12f}")
    print(f"a (intercept)= {a:.12f}")
    print(f"R² = {metrics['R2']:.6f}")
    print(f"SE_b = {metrics['SE_b']:.12f}, t = {metrics['t_b']:.6f}, p(two-sided) = {metrics['p_b']:.3e}")

    print("\n== SCIPY CHECK ==")
    print(f"b_scipy = {sp['b']:.12f}, a_scipy = {sp['a']:.12f}")
    print(f"R²_scipy = {sp['R2']:.6f}, p_b_scipy = {sp['p_b']:.3e}, SE_b_scipy = {sp['SE_b']:.12f}")

    print("\n== OUTLIERS (IQR on y) ==")
    print(f"Kept rows: {len(df_iqr)} (of {len(df)})")
    print(f"b_iqr = {b_iqr:.12f}, a_iqr = {a_iqr:.12f}, R²_iqr = {metrics_iqr['R2']:.6f}")

    print("\n== OUTLIERS (|residual| <= 3σ) ==")
    print(f"Kept rows: {len(df_res)} (of {len(df)})")
    print(f"b_res = {b_res:.12f}, a_res = {a_res:.12f}, R²_res = {metrics_res['R2']:.6f}")

    print("\n== FILES WRITTEN ==")
    print("treated_set.dat  (cleaned x,y CSV)")
    print("Figures shown: Cleaned Scatter, Fit with line, Residuals")
    print("====================================================\n")


if __name__ == "__main__":
    main()

