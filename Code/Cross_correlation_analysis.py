"""
Finds the most correlated lag for each indicator using cross-correlation with flood events.
For each indicato(rainfall, runoff, Evapotranspiration(ET), lake levels) correlates its 
value k days earlier with Jnglei(the choosen region) flood events on the 2002-2018 training data,
with and without detrending, and picks the lag with the strongest correlation. Checks multicollinearity 
at those lags(VIF > 10 flagged) prints the LAGs dict for the flood model, and saves CSVs. 


"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))    ## to save it 


OUTCOME_FILE = "flood_outcome_jonglei.parquet"
PREDICTOR_FILE = "predictors_jonglei.parquet"

TRAIN_START = "2002-01-01"
TRAIN_END = "2018-12-31"

MIN_LEAD = 3          #
MIN_PAIRS = 30


LAG_RANGES = {
    "rain_mm":       (3, 60),
    "runoff_mm":     (3, 60),
    "et_mm":         (3, 60),
    "lake_victoria": (30, 250),
    "lake_kyoga":    (30, 250),
    "lake_albert":   (14, 250),
}
INDICATORS = list(LAG_RANGES)
MAX_LAG = max(hi for _, hi in LAG_RANGES.values())



def detrend(s):
    """Remove a straight-line trend (fitted on the non-missing values)."""
    valid = s.notna()
    t = np.arange(len(s), dtype=float)
    out = pd.Series(np.nan, index=s.index)
    if valid.sum() < 2:
        return out
    slope, intercept = np.polyfit(t[valid], s[valid], 1)
    out[valid] = s[valid] - (slope * t[valid] + intercept)
    return out


def corr_at_lag(flood, x, lag):
    """Correlation between x on day t - lag and flood on day t. Shifts by DATE, not position."""
    x_lagged = x.shift(lag, freq="D").reindex(flood.index)
    pair = pd.concat([flood, x_lagged], axis=1).dropna()
    if len(pair) < MIN_PAIRS or pair.iloc[:, 0].std() < 1e-12 or pair.iloc[:, 1].std() < 1e-12:
        return np.nan
    return pair.iloc[:, 0].corr(pair.iloc[:, 1])


# Loading Training years
flood = pd.read_parquet(OUTCOME_FILE)["flood"].astype(float)
pred = pd.read_parquet(PREDICTOR_FILE)

pred = pred.loc[pd.Timestamp(TRAIN_START) - pd.Timedelta(days=MAX_LAG):TRAIN_END]
flood = flood.loc[TRAIN_START:TRAIN_END]
print(f"Flood days used: {flood.notna().sum()} ({flood.index.min().date()} to {flood.index.max().date()}), "
      f"event rate {flood.mean():.3f}")



all_rows, summary, curves = [], [], {}

for version in ["Not detrended", "Detrended"]:
    for col, (lo, hi) in LAG_RANGES.items():
        lo = max(lo, MIN_LEAD)
        lags = np.arange(lo, hi + 1)
        x = pred[col] if version == "Not detrended" else detrend(pred[col])

        r = pd.Series([corr_at_lag(flood, x, k) for k in lags], index=lags)
        curves[(version, col)] = r
        for k, v in r.items():
            all_rows.append({"version": version, "indicator": col, "lag": int(k), "r": v})

        if r.notna().any():
            best = int(r.abs().idxmax())
            summary.append({"version": version, "indicator": col, "search_range": f"{lo}-{hi}",
                            "best_lag": best, "r": r[best]})
        else:
            summary.append({"version": version, "indicator": col, "search_range": f"{lo}-{hi}",
                            "best_lag": np.nan, "r": np.nan})

summary = pd.DataFrame(summary)
for version, block in summary.groupby("version", sort=False):
    print(f"\n{version}:")
    print(block[["indicator", "search_range", "best_lag", "r"]].round(3).to_string(index=False))


# Computing VIF
def vif_table(columns):
    """VIF = 1 / (1 - R^2), with R^2 from regressing each column on all the others."""
    data = pd.concat(columns, axis=1).dropna()
    z = (data - data.mean()) / data.std()
    out = {}
    for c in z.columns:
        others = np.column_stack([z.drop(columns=c).values, np.ones(len(z))])
        coef, *_ = np.linalg.lstsq(others, z[c].values, rcond=None)
        resid = z[c].values - others @ coef
        r2 = 1 - resid @ resid / (z[c].values @ z[c].values)
        out[c] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(out), len(data)


for version in ["Not detrended", "Detrended"]:
    lag_of = summary[summary["version"] == version].set_index("indicator")["best_lag"]
    lagged = []
    for col in INDICATORS:
        if pd.isna(lag_of.get(col, np.nan)):
            continue
        k = int(lag_of[col])
        x = pred[col] if version == "Not detrended" else detrend(pred[col])
        lagged.append(x.shift(k, freq="D").reindex(flood.index).rename(f"{col}_lag{k}"))
    vif, n = vif_table(lagged)
    print(f"\nVIF {version} ({n} complete training days, each indicator at its best lag):")
    print(vif.round(2).to_string())
    high = vif[vif > 10]
    if len(high):
        print(f"  -> VIF > 10: {list(high.index)}. Drop or combine one of these before modelling.")
    vif.rename("VIF").to_csv(f"cross_correlation_vif_{'detrended' if version == 'Detrended' else 'raw'}.csv")


best = summary[summary["version"] == "Detrended"].set_index("indicator")["best_lag"]
print("\nLAGS for flood_model.py (detrended best lags):")
print("LAGS = {")
for col in INDICATORS:
    if pd.notna(best.get(col, np.nan)):
        print(f'    "{col}": [{int(best[col])}],')
print("}")


# Saving everything
pd.DataFrame(all_rows).to_csv("cross_correlation_all_lags.csv", index=False)
summary.to_csv("cross_correlation_best_lags.csv", index=False)

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
for ax, col in zip(axes.ravel(), INDICATORS):
    for version, style in [("Not detrended", "--"), ("Detrended", "-")]:
        r = curves[(version, col)]
        ax.plot(r.index, r.values, style, label=version)
    ax.axhline(0, color="grey", lw=0.8)
    k = best.get(col, np.nan)
    if pd.notna(k):
        ax.axvline(int(k), color="black", lw=1, ls=":")
        ax.set_title(f"{col}: best lag {int(k)} days")
    else:
        ax.set_title(col)
    ax.set_xlabel("lag (days before flood)")
    ax.set_ylabel("correlation")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Cross-correlation with flood-expansion events, per search range (training years 2002-2018)")
plt.tight_layout()
plt.savefig("cross_correlation_curves.png", dpi=150)
print("\nSaved cross_correlation_all_lags.csv, cross_correlation_best_lags.csv, "
      "cross_correlation_vif_*.csv and cross_correlation_curves.png")