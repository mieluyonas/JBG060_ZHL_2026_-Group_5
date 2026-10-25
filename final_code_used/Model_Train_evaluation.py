"""

Fits a logistic model that predicts Jonglei flood events from lagged predictors.

Builds a daily predictor for the region(rainfall, runoff, EVapotranspiration(ET), Lake Albert level),
lag them by at least 3 days, and trains on 2002-2018 with HAC standard errors. Reports significance(Wald test), 
effect size(odds ratios, marginal effects), and tests on 2019-2025 aganist traditional climatology and 
persistence baselines(AUC_PR,Brier, Brier Skill Score). 

Inorder to run use the files(import) CDBLhydro and capDBLexposure. and "flood_outcome_jonglei.parquet" and"predictors_jonglei.parquet". 


"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, brier_score_loss

from CDBLhydro import load_rainfall_runoff, process_ET, load_processed_ET, load_lake_stations
from capDBLexposure import load_admin_boundaries

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # Saving it 



OUTCOME_FILE = "flood_outcome_jonglei.parquet"
PREDICTOR_CACHE = "predictors_jonglei.parquet"
REGION_IDS = ["SS0302", "SS0303", "SS0305", "SS0306", "SS0310"]   # Ayod, Bor South, Duk, Fangak, Twic East(we are using these regions counties in our region)

START = "2002-01-01"
CUTOFF = "2019-01-01"     
LEAD = 3           


LAGS = {
    "rain_mm":       [10],
    "runoff_mm":     [10],
    "et_mm":         [9],
    "lake_victoria": [227],
    "lake_kyoga":    [63],
    "lake_albert":   [21],
}
DROP = ["lake_victoria", "lake_kyoga"]  

HAC_MAXLAGS = 30



def region_polygon():
    _, admin2 = load_admin_boundaries()
    return admin2[admin2["adm2_pcode"].isin(REGION_IDS)].dissolve().to_crs(4326)


def build_predictors(region):
    years = np.arange(2000, 2026)

    # Rain + runoff: ERA5 hourly summed to daily 
    minx, miny, maxx, maxy = region.total_bounds
    ds = load_rainfall_runoff(years)
    ds = ds.sel(latitude=slice(maxy + 0.25, miny - 0.25), longitude=slice(minx - 0.25, maxx + 0.25)).compute()
    lon2d, lat2d = np.meshgrid(ds.longitude.values, ds.latitude.values)
    cells = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon2d.ravel(), lat2d.ravel()), crs=4326)
    inside = cells.within(region.geometry.iloc[0]).values.reshape(lat2d.shape)
    print(f"ERA5 cells inside region: {inside.sum()}")
    era5 = pd.DataFrame({
        "rain_mm":   ds["tp"].values[:, inside].mean(axis=1) * 1000,   # m/day -> mm/day
        "runoff_mm": ds["ro"].values[:, inside].mean(axis=1) * 1000,
    }, index=pd.to_datetime(ds.valid_time.values))

    # ET: already daily; taken at the region centre
    centre = region.to_crs(6933).centroid.to_crs(4326).iloc[0]
    lon_c, lat_c = round(centre.x, 3), round(centre.y, 3)
    print(f"ET grid point (region centre): lon {lon_c}, lat {lat_c}")
    for year in years:
        process_ET(year, target_longitude=lon_c, target_latitude=lat_c)
    et = pd.concat(load_processed_ET(years, lon_c, lat_c).values(), ignore_index=True)
    et["date"] = pd.to_datetime(et["date"])
    et = et.set_index("date")["gridcell"].rename("et_mm")

    # Water level: satellite readings carried forward to every day 
    lake_series = {}
    for name, df in load_lake_stations().items():
        col = "height_wrt_ref" if "height_wrt_ref" in df.columns else df.select_dtypes("number").columns[0]
        print(f"Lake {name}: using column '{col}'")
        s = df[col].groupby(df.index.normalize()).mean()
        lake_series[f"lake_{name.lower()}"] = s.resample("D").mean().ffill()

    predictors = era5.join(et, how="left").join(pd.DataFrame(lake_series), how="left")
    predictors.index.name = "date"
    return predictors


if os.path.exists(PREDICTOR_CACHE):
    predictors = pd.read_parquet(PREDICTOR_CACHE)
    print(f"Loaded cached predictors: {predictors.shape}")
else:
    predictors = build_predictors(region_polygon())
    predictors.to_parquet(PREDICTOR_CACHE)
    print(f"Built and saved predictors: {predictors.shape}")

data = pd.read_parquet(OUTCOME_FILE)[["flood"]].join(predictors, how="left")



features, feature_of = [], {}
for col, lags in LAGS.items():
    if col in DROP:
        continue
    for k in lags:
        assert k >= LEAD, f"Lag {k} for {col} is shorter than LEAD ({LEAD})"
        name = f"{col}_lag{k}"
        data[name] = data[col].shift(k)
        features.append(name)
        feature_of[name] = col


G = min(k for col, lags in LAGS.items() if col not in DROP for k in lags)
data["flood_lag_g"] = data["flood"].shift(G)



data = data.loc[START:].dropna()
train = data[data.index < CUTOFF]
test = data[data.index >= CUTOFF]
print(f"\nTrain: {train.index.min().date()} to {train.index.max().date()} ({len(train)} days, event rate {train['flood'].mean():.3f})")
print(f"Test:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} days, event rate {test['flood'].mean():.3f})")

mu, sd = train[features].mean(), train[features].std()     # training years only
Xtr = sm.add_constant((train[features] - mu) / sd)
Xte = sm.add_constant((test[features] - mu) / sd)



model = sm.GLM(train["flood"], Xtr, family=sm.families.Binomial()).fit(
    cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS}
)
print(f"\nModel fitted on {int(model.nobs)} days (HAC maxlags = {HAC_MAXLAGS})")


# Significance 
ci = model.conf_int()
per_lag = pd.DataFrame({
    "beta": model.params,
    "SE": model.bse,
    "z": model.params / model.bse,
    "p_value": model.pvalues,
}).drop(index="const")
per_lag["significant_5%"] = per_lag["p_value"] < 0.05

joint = []
for col in dict.fromkeys(feature_of.values()):
    terms = [f for f in features if feature_of[f] == col]
    w = model.wald_test(", ".join(f"{t} = 0" for t in terms), scalar=True)
    joint.append({"indicator": col, "n_lags": len(terms),
                  "wald_chi2": float(w.statistic), "p_value": float(w.pvalue)})
joint = pd.DataFrame(joint).set_index("indicator")

print("\nStep 6a - Wald test per lag:")
print(per_lag.round(3).to_string())
print("\nStep 6b - Joint Wald test per indicator (all lags = 0):")
print(joint.round(4).to_string())


# Magnitude 
p_train = model.predict(Xtr)
scale = (p_train * (1 - p_train)).mean()            
magnitude = pd.DataFrame({
    "odds_ratio": np.exp(model.params),
    "OR_95%_low": np.exp(ci[0]),
    "OR_95%_high": np.exp(ci[1]),
    "marginal_effect_pp": model.params * scale * 100,  
}).drop(index="const")
print("\nStep 7 - Magnitude per +1 SD:")
print(magnitude.round(3).to_string())


# Evaluation of model
y = test["flood"].values
p_model = model.predict(Xte).values

# 1. Traditional climatology
md_train = train.index.strftime("%m-%d")      
md_test = test.index.strftime("%m-%d")
floods_per_day = train["flood"].groupby(md_train).sum()
years_per_day = train["flood"].groupby(md_train).count()
clim_by_day = floods_per_day / years_per_day
p_clim = pd.Series(md_test).map(clim_by_day).fillna(train["flood"].mean()).values   # e.g. 29 Feb -> overall rate

# 2. Traditional persistence
p_pers = test["flood_lag_g"].values.astype(float)

names = ["Traditional climatology (same day, past years)",
         f"Traditional persistence (flood {G} days ago)",
         "Logistic model"]
preds = [p_clim, p_pers, p_model]

scores = pd.DataFrame({
    "AUC_PR": [average_precision_score(y, p) for p in preds],
    "Brier":  [brier_score_loss(y, p) for p in preds],
}, index=names)

# Brier Skill Score of the model against each baseline: > 0 means the model is better
scores["model_BSS_vs_this"] = 1 - scores.loc["Logistic model", "Brier"] / scores["Brier"]
scores.loc["Logistic model", "model_BSS_vs_this"] = np.nan

print(f"\nTest {CUTOFF[:4]}-2025 ({len(y)} days, event rate {y.mean():.3f}; "
      f"AUC-PR of a random guess = {y.mean():.3f}):")
print(scores.round(3).to_string())
print("\nmodel_BSS_vs_this > 0: the model beats that baseline; < 0: the baseline is better.")