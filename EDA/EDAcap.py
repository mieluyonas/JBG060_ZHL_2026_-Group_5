import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

from CDBLhydro import load_dartmouth_data, load_lake_stations, load_flood_masks, load_rainfall_runoff
from capDBLexposure import load_admin_boundaries, load_worldpop_area, load_GDP, load_ipc_data

discharge_data = load_dartmouth_data()
lake_data = load_lake_stations()

# Rainfall/runoff
years_rainfall = np.arange(2000, 2026)
rainfall_runoff = load_rainfall_runoff(years_rainfall)

# Flood masks 
years_flood = np.arange(2000, 2026)
bbox = {'lat_min': 3.5, 'lat_max': 12.5, 'lon_min': 24.0, 'lon_max': 36.0}
flood_data = load_flood_masks(years_flood, bbox=bbox)

# Population
pop_bbox = {'lon_min': 24, 'lat_min': 3.5, 'lon_max': 36, 'lat_max': 12.5}
pop_data = load_worldpop_area(pop_bbox)

# GDP 
gdp_data = load_GDP()

# IPC
ipc_data = load_ipc_data()

admin1, admin2 = load_admin_boundaries()


import pandas as pd
import matplotlib.pyplot as plt


# 1. EDA for discharge stations

for station_id, df in discharge_data.items():
    print(f"\n=== Station {station_id} ===")
    print(f"Date range   : {df.index.min()} to {df.index.max()}")
    print(f"Number of rows: {len(df)}")
    print(f"Missing values:\n{df.isna().sum()}")
    print(f"\nSummary stats:\n{df.describe()}")

    # Plot discharge over time
    plt.figure(figsize=(10, 3))
    df.plot(ax=plt.gca(), legend=False)
    plt.title(f"Discharge over time - Station {station_id}")
    plt.xlabel("Date")
    plt.ylabel("Discharge (m³/s)")
    plt.tight_layout()
    plt.show()



#2. EDA for lakes

for lake_name, df in lake_data.items():  
    print(f"\n=== Lake {lake_name} ===")
    print(f"Date range   : {df.index.min()} to {df.index.max()}")
    print(f"Number of rows: {len(df)}")
    print(f"Missing values:\n{df.isna().sum()}")
    print(f"\nSummary stats:\n{df.describe()}")

    # We pick the water level column (name differs slightly between lakes)
    level_col = [c for c in df.columns if c == 'height_wrt_ref' or 'level' in c.lower()]
    if level_col:
        plt.figure(figsize=(10, 3))
        df[level_col[0]].plot()
        plt.title(f"Water level over time - Lake {lake_name}")
        plt.xlabel("Date")
        plt.ylabel("Water level")
        plt.tight_layout()
        plt.show()

for lake_name, df in lake_data.items():
    if 'mission' not in df.columns:
        continue

    print(f"\n Lake {lake_name} — by mission ")
    mission_summary = (
        df.reset_index()
          .groupby('mission')['date']
          .agg(rows='count', start='min', end='max')
    )
    print(mission_summary)

path = r'C:\Users\20223778\Downloads\CDBLhydro.py'
with open(path, 'r') as f:
    for line in f:
        if 'mission_names' in line:
            print(repr(line))

    


# 3.  EDA for rainfall & runoff
print(f"\n Rainfall & Runoff")
print(rainfall_runoff)
print(f"Missing precipitation (tp): {rainfall_runoff['tp'].isnull().sum().values}")
print(f"Missing runoff (ro): {rainfall_runoff['ro'].isnull().sum().values}")

# Basin-averaged daily precipitation/runoff, plotted over time
tp_daily_mean = rainfall_runoff['tp'].mean(dim=['latitude', 'longitude'])
ro_daily_mean = rainfall_runoff['ro'].mean(dim=['latitude', 'longitude'])

plt.figure(figsize=(10, 3))
tp_daily_mean.plot()
plt.title("Basin-averaged Precipitation over time")
plt.xlabel("Date")
plt.ylabel("Precipitation (m/day)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 3))
ro_daily_mean.plot()
plt.title("Basin-averaged Runoff over time")
plt.xlabel("Date")
plt.ylabel("Runoff (m/day)")
plt.tight_layout()
plt.show()



# 4) EDA for flood masks'

print(f"\n Flood Masks ")
print(flood_data.head())
print(f"Total rows: {len(flood_data)}")
print(f"Missing values:\n{flood_data.isna().sum()}")
print(f"Recurring floods (type 0): {(flood_data['flood_type'] == 0).sum()}")
print(f"Unusual floods (type 1)  : {(flood_data['flood_type'] == 1).sum()}")

flood_data['year'] = flood_data['date'].dt.year
floods_per_year = flood_data.groupby('year').size()

plt.figure(figsize=(8, 3))
floods_per_year.plot(kind='bar')
plt.title("Flood records per year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 5. EDA for population 

print(f"\n=== Population ===")
pop_df = pd.Series(pop_data).to_frame('population')
print(pop_df)
print(f"Missing values:\n{pop_df.isna().sum()}")

plt.figure(figsize=(8, 3))
pop_df['population'].plot(kind='bar')
plt.title("Total Population by Year (bounding box)")
plt.xlabel("Year")
plt.ylabel("Population")
plt.tight_layout()
plt.show()

#6. EDA for GDP

print(f"\n GDP ")
gdp_df = pd.Series(gdp_data).to_frame('GDP_USD')
print(gdp_df)
print(f"Missing values:\n{gdp_df.isna().sum()}")



# 7. EDA for IPC

print(f"\n IPC Data ")
print(ipc_data.head())
print(f"Shape: {ipc_data.shape}")
print(f"Missing values:\n{ipc_data.isna().sum()}")



# 8. EDA for administrative boundaries

print(f"\n Admin Boundaries ")
print(admin1.head())
print(f"Missing values (admin1):\n{admin1.isna().sum()}")
print(f"Missing values (admin2):\n{admin2.isna().sum()}")

