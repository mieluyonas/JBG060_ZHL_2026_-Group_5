"""
Creates a daily flood-event label per county. Satellite flood pixels are joined to counties, smoothed(7-day mean)
and compared with 7 days earlier. A day is a flood event if the growth is above  the county's 90th percentile 
(from2002-2018 training data), and  at least 9 pixels. saves every countyoutputs and  aper-county summary.

""" 

import pandas as pd 
import numpy as np
import geopandas as gpd 
import time
import os

from CDBLhydro import load_flood_masks
from capDBLexposure import load_admin_boundaries 

os.chdir(os.path.dirname(os.path.abspath(__file__)))

cache = "flood_joined_counties.parquet"
Output_counties = "flood_outcome_counties.parquet"
Output_region = "flood_outcome_jonglei.parquet"


Pixel_size_KM = 0.232
region_ids = ["SS0302", "SS0303", "SS0305", "SS0306","SS0310" ]


cutoff = "2019-01-01" # trains before this date and test after
Start_output = "2002-01-01"
Smooth_days = 7   # rolling mean window
Growth_days = 7 # compare this with the early days
EPS_PX = 9   
MIN_GAIN_PX = 9 
PCTL = 0.90  # event threshold percentile(for training set)
MIN_FLOOD_DAYS_SHARE = 0.10


# Loading the counties boundries(admin-2)
admin1, admin2 = load_admin_boundaries()
counties = admin2.copy()
counties["county_id"] = counties["adm2_pcode"]
print(f'Number of counties: {len(counties)}')


pixel_area_km2 = Pixel_size_KM ** 2   #  Calculating Total picel per county
if 'area_sqkm' in counties.columns:
    counties["total_pixels"] = counties["area_sqkm"] / pixel_area_km2
else:
    counties["total_pixels"] = counties.to_crs(epsg=6933).geometry.area / 1e6 / pixel_area_km2

if os.path.exists(cache):
    joined = pd.read_parquet(cache)
    print(f"loaded chach join: {len(joined):,} pixel days")
else: 
    minx,miny, maxx, maxy = counties.total_bounds
    bbox = {"lat_min": miny, "lat_max": maxy, "lon_min": minx, "lon_max": maxx}
    flood_data = load_flood_masks(np.arange(2000, 2026), bbox=bbox)
    print(f"Total flood pixel-days: {len(flood_data):,}")

    flood_gdf = gpd.GeoDataFrame(
        flood_data,
        geometry=gpd.points_from_xy(flood_data["lon"], flood_data["lat"]),
        crs="EPSG:4326"
    ).to_crs(epsg=6933)

    counties_proj = counties.to_crs(epsg=6933).copy()

    counties_proj["geometry"] = counties_proj["geometry"].simplify(100, preserve_topology=True)     
    print("starting spatial join")
    t0 = time.time()

    joined = gpd.sjoin( flood_gdf, counties_proj[["county_id", "adm2_name", "geometry"]],
        how="inner", predicate="within")

    print(f"Spatial join done in {time.time() - t0:.1f} sec. Matched {len(joined):,} pixel-days.") ## It was talking long so i want to see the time it took each time to run
    
    joinedoined = pd.DataFrame(joined.drop(columns='geometry'))
    joined.to_parquet(cache)

joined["date"] = pd.to_datetime(joined["date"])

full_dates = pd.date_range(joined["date"].min(), joined["date"].max(), freq="D")

panel = pd.MultiIndex.from_product(
    [counties["county_id"], full_dates], names=["county_id", "date"]
).to_frame(index=False)

panel = panel.merge(counties[["county_id", "adm2_name", "total_pixels"]], on = "county_id", how= 'left')

daily_counts = joined.groupby(["county_id", "date"]).size().rename("flagged_pixels").reset_index()

panel = panel.merge(daily_counts, on = ["county_id", "date"], how= 'left')
panel["flagged_pixels"] = panel["flagged_pixels"].fillna(0)
panel = panel.sort_values(["county_id", "date"]).reset_index(drop=True)

def add_expansion(df, group_col):
    g = df.groupby(group_col)
    df["flooded_share"] = df["flagged_pixels"] / df["total_pixels"]
    df["share_smooth"] = g["flooded_share"].transform(lambda s: s.rolling(Smooth_days, min_periods=1).mean())
    prev = df.groupby(group_col)["share_smooth"].shift(Growth_days)
    eps = EPS_PX / df["total_pixels"]
    df["rel_expansion"] = np.log((df["share_smooth"] + eps) / (prev + eps))
    df["gain_px"] = (df["share_smooth"] - prev) * df["total_pixels"]
 
    train = (df["date"] >= Start_output) & (df["date"] < cutoff)
    thr = df[train].groupby(group_col)["rel_expansion"].quantile(PCTL).rename("threshold")
    df = df.merge(thr, left_on=group_col, right_index=True, how="left")
    df["flood"] = ((df["rel_expansion"] > df["threshold"]) & (df["gain_px"] >= MIN_GAIN_PX)).astype(int)
    return df

panel = add_expansion(panel, "county_id")
panel["period"] = np.where(panel["date"] < cutoff, "train", "test")
panel.loc[panel["date"] < Start_output, "period"] = "excluded"


use = panel[panel["period"] != "excluded"]
summary = use.groupby(["county_id", "adm2_name"]).agg(
    threshold=("threshold", "first"),
    flooded_days_train=("flagged_pixels", lambda s: (s[use.loc[s.index, "period"] == "train"] > 0).mean()),
    event_rate_train=("flood", lambda s: s[use.loc[s.index, "period"] == "train"].mean()),
    event_rate_test=("flood", lambda s: s[use.loc[s.index, "period"] == "test"].mean()),
    events_total=("flood", "sum"),
).reset_index()
summary["reliable"] = summary["flooded_days_train"] >= MIN_FLOOD_DAYS_SHARE
summary = summary.sort_values("events_total", ascending=False)
 
print("\nPer county (sorted by number of events):")
print(summary.round(3).to_string(index=False))
print(f"\nReliable counties (flooded on >= {MIN_FLOOD_DAYS_SHARE:.0%} of training days): "
      f"{summary['reliable'].sum()} of {len(summary)}")

reg = panel[panel["county_id"].isin(region_ids)]
regional = reg.groupby("date")[["flagged_pixels", "total_pixels"]].sum().reset_index()



country_total = panel.groupby("date")["flagged_pixels"].sum()
missing_dates = country_total.index[country_total == 0]
regional.loc[regional["date"].isin(missing_dates), "flagged_pixels"] = np.nan
print(f"Days with no satellite data (set to missing): {len(missing_dates)}")



regional["region"] = "jonglei"
regional = add_expansion(regional, "region").set_index("date")

regional.loc[regional["rel_expansion"].isna(), "flood"] = np.nan   
r = regional.loc[Start_output:]
print(f"\nJonglei region: threshold {r['threshold'].iloc[0]:.3f}, "
      f"train {r.loc[:'2018-12-31', 'flood'].mean():.3f}, test {r.loc[cutoff:, 'flood'].mean():.3f}")
 
 
# Saving output
out_c = panel.loc[panel["period"] != "excluded",
                  ["county_id", "adm2_name", "date", "flooded_share", "rel_expansion", "flood"]]
out_c = out_c.merge(summary[["county_id", "reliable"]], on="county_id", how="left")
out_c.to_parquet(Output_counties)
 
r[["flooded_share", "share_smooth", "rel_expansion", "flood"]].to_parquet(Output_region)
summary.to_csv("flood_outcome_county_summary.csv", index=False)
 
print(f"\nSaved {Output_counties} ({len(out_c):,} county-days), {Output_region} ({len(r):,} days)"
      f" and flood_outcome_county_summary.csv")
 
