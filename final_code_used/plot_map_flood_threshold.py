"""
Flood-event threshold map, publication style.

Run this file directly. It must sit in the same folder as capDBLexposure.py and
flood_outcome_county_summary.csv (made by the threshold script). It saves flood_threshold_map.png.

Optional layers (set them in the Run section at the bottom):
    rivers       all rivers (thin lines)
    main_rivers  main Nile system, with a 'name' column for labels
    recurring    frequently flooded land (polygons or points)
    neighbours   neighbouring countries (grey context around South Sudan)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrow
from matplotlib.ticker import FuncFormatter

# Style
# Colour-blind-friendly scale

CMAP = ListedColormap(plt.get_cmap("magma_r")(np.linspace(0.05, 0.85, 256)), name="magma_trim")
BACKGROUND = "#eef2f5"     # light grey-blue page
NEIGHBOUR = "#f7f7f4"      # neighbouring countries (almost white, so it is not confused with no data)
NO_DATA = "#cfdcc8"        # counties with too little flooding: soft sage green (not on the colour scale)
COUNTY_EDGE = "white"
STATE_EDGE = "#7a7a7a"
COUNTRY_EDGE = "#2f5d50"   # dark green outline, like the Swiss example
RIVER = "#2b7bba"
RECURRING = "#2b7bba"
REGION_EDGE = "#1a1a1a"
TEXT = "#333333"
HALO = [pe.withStroke(linewidth=2.5, foreground="white")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def _add_north_arrow(ax, x=0.95, y=0.93):
    ax.annotate("N", xy=(x, y), xytext=(x, y - 0.07), xycoords="axes fraction",
                ha="center", va="center", fontsize=11, weight="bold", color=TEXT,
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.5))


def _add_scale_bar(ax, length_km=100):
    """Simple scale bar in the lower right, correct for the map's mid latitude."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    km_per_deg = 111.32 * np.cos(np.deg2rad((y0 + y1) / 2))
    length_deg = length_km / km_per_deg
    xs = x1 - (x1 - x0) * 0.05 - length_deg
    ys = y0 + (y1 - y0) * 0.04
    h = (y1 - y0) * 0.008
    for i in range(2):
        ax.add_patch(plt.Rectangle((xs + i * length_deg / 2, ys), length_deg / 2, h,
                                   facecolor=TEXT if i == 0 else "white", edgecolor=TEXT,
                                   linewidth=0.8, zorder=12))
    for frac, label in [(0, "0"), (0.5, f"{length_km // 2}"), (1, f"{length_km} km")]:
        ax.text(xs + frac * length_deg, ys + h * 2.2, label, ha="center", va="bottom",
                fontsize=7, color=TEXT, zorder=12)


def plot_threshold_map(counties, states, rivers=None, main_rivers=None, recurring=None,
                       neighbours=None, region_ids=(), vmin=1.0, vmax=3.5,
                       source="Source: NASA MODIS/VIIRS flood masks 2002–2018; OCHA admin boundaries",
                       out="flood_threshold_map.png"):
    counties = counties.to_crs(4326).copy()
    states = states.to_crs(4326)
    country = states.dissolve()

    # Threshold is a log ratio -> weekly growth factor (e.g. 1.226 -> x3.4)
    counties["growth"] = np.exp(counties["threshold"])
    reliable = counties[counties["reliable"]]
    unreliable = counties[~counties["reliable"]]

    fig, ax = plt.subplots(figsize=(11, 9.5), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Context: neighbouring countries
    if neighbours is not None:
        neighbours.to_crs(4326).plot(ax=ax, color=NEIGHBOUR, edgecolor="#bbbbbb", linewidth=0.6, zorder=0)

    # Soft shadow behind the country
    country.translate(0.06, -0.06).plot(ax=ax, color="#000000", alpha=0.12, linewidth=0, zorder=1)

    # Counties
    reliable.plot(ax=ax, column="growth", cmap=CMAP, norm=norm,
                  edgecolor=COUNTY_EDGE, linewidth=0.35, zorder=2)
    unreliable.plot(ax=ax, color=NO_DATA, edgecolor=COUNTY_EDGE, linewidth=0.35, zorder=2)

    # Frequently flooded land
    if recurring is not None:
        rec = recurring.to_crs(4326)
        if rec.geom_type.isin(["Point", "MultiPoint"]).all():
            rec.plot(ax=ax, color=RECURRING, markersize=0.6, alpha=0.5, zorder=3)
        else:
            rec.plot(ax=ax, facecolor="none", edgecolor=RECURRING, hatch="....", linewidth=0, zorder=3)

    # State borders
    states.boundary.plot(ax=ax, color=STATE_EDGE, linewidth=0.7, zorder=4)

    # Rivers
    if rivers is not None:
        rivers.to_crs(4326).plot(ax=ax, color=RIVER, linewidth=0.6, alpha=0.8, zorder=5)
    if main_rivers is not None:
        mr = main_rivers.to_crs(4326)
        mr.plot(ax=ax, color="white", linewidth=3.2, zorder=6)
        mr.plot(ax=ax, color=RIVER, linewidth=1.8, zorder=7)
        if "name" in mr.columns:
            for name, grp in mr.groupby("name"):
                geom = grp.geometry
                line = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
                p = line.interpolate(0.5, normalized=True)
                ax.annotate(name, (p.x, p.y), xytext=(5, 0), textcoords="offset points",
                            fontsize=7.5, style="italic", color=RIVER, zorder=11, path_effects=HALO)

    # Country outline
    country.boundary.plot(ax=ax, color=COUNTRY_EDGE, linewidth=1.8, zorder=8)

    # State labels (small caps-style, muted)
    for _, s in states.iterrows():
        p = s.geometry.representative_point()
        ax.annotate(str(s["adm1_name"]).upper(), (p.x, p.y), ha="center", va="center",
                    fontsize=6.5, color="#5a5a5a", zorder=9, path_effects=HALO)

    # Study region outline and labels
    region = counties[counties["county_id"].isin(region_ids)]
    if len(region):
        region.dissolve().boundary.plot(ax=ax, color=REGION_EDGE, linewidth=1.8, linestyle="-", zorder=10)
        for _, r in region.iterrows():
            p = r.geometry.representative_point()
            ax.annotate(r["adm2_name"], (p.x, p.y), ha="center", va="center", fontsize=8,
                        weight="bold", color=TEXT, zorder=11, path_effects=HALO)

    # Frame to the country with a small margin
    minx, miny, maxx, maxy = country.total_bounds
    padx, pady = (maxx - minx) * 0.04, (maxy - miny) * 0.04
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady * 2.5, maxy + pady)
    ax.set_aspect("equal")
    ax.set_axis_off()

    _add_north_arrow(ax)
    _add_scale_bar(ax)

    # Title and subtitle, left-aligned
    fig.text(0.06, 0.95, "Flood-event threshold per county", fontsize=16, weight="bold",
             color=TEXT, ha="left")
    fig.text(0.06, 0.925, "Weekly growth in flooded area needed to count as a flood event, "
             "South Sudan (training years 2002–2018)", fontsize=10, color="#555555", ha="left")

    # Horizontal colour bar at the bottom, plus a 'no data' box
    cax = fig.add_axes([0.16, 0.075, 0.40, 0.018])
    sm = ScalarMappable(norm=norm, cmap=CMAP)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max")
    cbar.formatter = FuncFormatter(lambda v, _: f"×{v:.1f}")
    cbar.update_ticks()
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=8, length=2, color=TEXT, labelcolor=TEXT)
    cbar.set_label("Weekly growth needed for a flood event", fontsize=8.5, color=TEXT, labelpad=3)
    cbar.ax.xaxis.set_label_position("top")

    nax = fig.add_axes([0.07, 0.075, 0.035, 0.018])
    nax.add_patch(plt.Rectangle((0, 0), 1, 1, color=NO_DATA))
    nax.set_xlim(0, 1); nax.set_ylim(0, 1); nax.set_axis_off()
    fig.text(0.0875, 0.06, "Too little\nflooding", ha="center", va="top", fontsize=7, color=TEXT)

    # Legend for lines and areas
    handles = [Line2D([], [], color=REGION_EDGE, linewidth=1.8, label="Study region (5 Jonglei counties)"),
               Line2D([], [], color=COUNTRY_EDGE, linewidth=1.8, label="National border"),
               Line2D([], [], color=STATE_EDGE, linewidth=0.7, label="State border")]
    if main_rivers is not None or rivers is not None:
        handles.append(Line2D([], [], color=RIVER, linewidth=1.8, label="Rivers (main Nile system bold)"))
    if recurring is not None:
        handles.append(Patch(facecolor="none", edgecolor=RECURRING, hatch="....", label="Frequently flooded land (Sudd)"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.62, 0.035), fontsize=7.5,
               frameon=False, ncol=1)

    # Source note
    fig.text(0.06, 0.02, source, fontsize=7, color="#777777", ha="left")

    fig.savefig(out, dpi=300, facecolor=BACKGROUND, bbox_inches="tight")
    return fig, ax


# ================= Run the map =================
if __name__ == "__main__":
    from capDBLexposure import load_admin_boundaries

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # County and state shapes
    admin1, admin2 = load_admin_boundaries()
    counties = admin2.copy()
    counties["county_id"] = counties["adm2_pcode"]

    # Threshold results from the threshold script
    summary = pd.read_csv("flood_outcome_county_summary.csv")
    counties = counties.merge(summary[["county_id", "threshold", "reliable"]], on="county_id", how="left")
    counties["reliable"] = counties["reliable"].fillna(False).astype(bool)

    plot_threshold_map(
        counties,
        states=admin1,
        rivers=None,        # replace with your rivers GeoDataFrame
        main_rivers=None,   # replace with your main Nile rivers (with a 'name' column)
        recurring=None,     # replace with your frequently flooded land
        neighbours=None,    # optional: neighbouring countries for grey context
        region_ids=["SS0302", "SS0303", "SS0305", "SS0306", "SS0310"],
    )
    print("Saved flood_threshold_map.png")

## clean part

"""
Flood-event threshold map, publication style.

Run this file directly. It must sit in the same folder as capDBLexposure.py and
flood_outcome_county_summary.csv (made by the threshold script). It saves flood_threshold_map.png.

Optional layers (set them in the Run section at the bottom):
    rivers       all rivers (thin lines)
    main_rivers  main Nile system, with a 'name' column for labels
    recurring    frequently flooded land (polygons or points)
    neighbours   neighbouring countries (grey context around South Sudan)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------- Style ----------------
# Colour-blind-safe scale: magma (perceptually uniform, readable with all common colour-vision
# deficiencies), reversed and trimmed so it runs pale yellow -> orange -> deep purple, not black
CMAP = ListedColormap(plt.get_cmap("magma_r")(np.linspace(0.05, 0.85, 256)), name="magma_trim")
BACKGROUND = "#dde4ea"     # grey-blue page (a bit darker, so the grey counties stand out)
NEIGHBOUR = "#f7f7f4"      # neighbouring countries
NO_DATA = "#c6cacd"        # counties below the threshold: plain light grey (not on the colour scale)
COUNTY_EDGE = "white"
STATE_EDGE = "#7a7a7a"
COUNTRY_EDGE = "#3b5b7a"   # dark slate blue, so it is easy to tell apart from the black study region
RIVER = "#2b7bba"
RECURRING = "#2b7bba"
REGION_EDGE = "#1a1a1a"
TEXT = "#333333"
HALO = [pe.withStroke(linewidth=2.5, foreground="white")]
NO_DATA_LABEL = "Below threshold"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def _add_north_arrow(ax, x=0.95, y=0.95):
    ax.annotate("N", xy=(x, y), xytext=(x, y - 0.07), xycoords="axes fraction",
                ha="center", va="center", fontsize=11, weight="bold", color=TEXT,
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.5))


def _add_scale_bar(ax, length_km=100):
    """Simple scale bar in the lower right, correct for the map's mid latitude."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    km_per_deg = 111.32 * np.cos(np.deg2rad((y0 + y1) / 2))
    length_deg = length_km / km_per_deg
    xs = x1 - (x1 - x0) * 0.05 - length_deg
    ys = y0 + (y1 - y0) * 0.04
    h = (y1 - y0) * 0.008
    for i in range(2):
        ax.add_patch(plt.Rectangle((xs + i * length_deg / 2, ys), length_deg / 2, h,
                                   facecolor=TEXT if i == 0 else "white", edgecolor=TEXT,
                                   linewidth=0.8, zorder=12))
    for frac, label in [(0, "0"), (0.5, f"{length_km // 2}"), (1, f"{length_km} km")]:
        ax.text(xs + frac * length_deg, ys + h * 2.2, label, ha="center", va="bottom",
                fontsize=7, color=TEXT, zorder=12)


def _fmt_growth(v):
    """1.0 -> '×1', 1.5 -> '×1.5'"""
    return "×" + f"{v:.1f}".rstrip("0").rstrip(".")


def plot_threshold_map(counties, states, rivers=None, main_rivers=None, recurring=None,
                       neighbours=None, region_ids=(), vmin=1.0, vmax=3.5,
                       source="Source: NASA MODIS/VIIRS flood masks 2002–2018; OCHA admin boundaries",
                       out="flood_threshold_map.png"):
    counties = counties.to_crs(4326).copy()
    states = states.to_crs(4326)
    country = states.dissolve()

    # Threshold is a log ratio -> weekly growth factor (e.g. 1.226 -> x3.4)
    counties["growth"] = np.exp(counties["threshold"])
    reliable = counties[counties["reliable"]]
    unreliable = counties[~counties["reliable"]]

    # Map frame: the country with a small margin
    minx, miny, maxx, maxy = country.total_bounds
    padx, pady = (maxx - minx) * 0.04, (maxy - miny) * 0.04
    view = (minx - padx, maxx + padx, miny - pady * 2.5, maxy + pady)

    # Figure sized to the map's own shape -> no empty space around it
    # Layout, top to bottom (inches): title | map | colour bar | legend row | source
    map_w = 9.0
    map_h = map_w * (view[3] - view[2]) / (view[1] - view[0])
    side, top_h, bot_h = 0.3, 0.85, 1.65
    W, H = map_w + 2 * side, map_h + top_h + bot_h
    fig = plt.figure(figsize=(W, H), facecolor=BACKGROUND)
    ax = fig.add_axes([side / W, bot_h / H, map_w / W, map_h / H])
    ax.set_anchor("C")
    ax.set_facecolor(BACKGROUND)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Context: neighbouring countries
    if neighbours is not None:
        neighbours.to_crs(4326).plot(ax=ax, color=NEIGHBOUR, edgecolor="#bbbbbb", linewidth=0.6, zorder=0)

    # Soft shadow behind the country
    country.translate(0.06, -0.06).plot(ax=ax, color="#000000", alpha=0.12, linewidth=0, zorder=1)

    # Counties with a threshold: coloured by the growth factor
    reliable.plot(ax=ax, column="growth", cmap=CMAP, norm=norm,
                  edgecolor=COUNTY_EDGE, linewidth=0.35, zorder=2)
    # Counties below the threshold: plain light grey
    unreliable.plot(ax=ax, color=NO_DATA, edgecolor=COUNTY_EDGE, linewidth=0.35, zorder=2)

    # Frequently flooded land
    if recurring is not None:
        rec = recurring.to_crs(4326)
        if rec.geom_type.isin(["Point", "MultiPoint"]).all():
            rec.plot(ax=ax, color=RECURRING, markersize=0.6, alpha=0.5, zorder=3)
        else:
            rec.plot(ax=ax, facecolor="none", edgecolor=RECURRING, hatch="....", linewidth=0, zorder=3)

    # State borders
    states.boundary.plot(ax=ax, color=STATE_EDGE, linewidth=0.7, zorder=4)

    # Rivers
    if rivers is not None:
        rivers.to_crs(4326).plot(ax=ax, color=RIVER, linewidth=0.6, alpha=0.8, zorder=5)
    if main_rivers is not None:
        mr = main_rivers.to_crs(4326)
        mr.plot(ax=ax, color="white", linewidth=3.2, zorder=6)
        mr.plot(ax=ax, color=RIVER, linewidth=1.8, zorder=7)
        if "name" in mr.columns:
            for name, grp in mr.groupby("name"):
                geom = grp.geometry
                line = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
                p = line.interpolate(0.5, normalized=True)
                ax.annotate(name, (p.x, p.y), xytext=(5, 0), textcoords="offset points",
                            fontsize=7.5, style="italic", color=RIVER, zorder=11, path_effects=HALO)

    # Country outline
    country.boundary.plot(ax=ax, color=COUNTRY_EDGE, linewidth=1.6, zorder=8)

    # State labels (muted capitals)
    for _, s in states.iterrows():
        p = s.geometry.representative_point()
        ax.annotate(str(s["adm1_name"]).upper(), (p.x, p.y), ha="center", va="center",
                    fontsize=6.5, color="#5a5a5a", zorder=9, path_effects=HALO)

    # Study region outline and labels
    region = counties[counties["county_id"].isin(region_ids)]
    if len(region):
        region.dissolve().boundary.plot(ax=ax, color=REGION_EDGE, linewidth=2.2, zorder=10)
        for _, r in region.iterrows():
            p = r.geometry.representative_point()
            ax.annotate(r["adm2_name"], (p.x, p.y), ha="center", va="center", fontsize=8,
                        weight="bold", color=TEXT, zorder=11, path_effects=HALO)

    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    _add_north_arrow(ax)
    _add_scale_bar(ax)

    # Title and subtitle (centred)
    fig.text(0.5, (H - 0.15) / H, "Flood-event threshold per county", fontsize=16, weight="bold",
             color=TEXT, ha="center", va="top")
    fig.text(0.5, (H - 0.52) / H, "Weekly growth in flooded area needed to count as a flood event, "
             "South Sudan (training years 2002–2018)", fontsize=10, color="#555555",
             ha="center", va="top")

    # ---------------- Colour bar: centred under the map ----------------
    cb_w = 0.45 * map_w
    cb_left = (W - cb_w) / 2
    cb_y, cb_h = 0.95, 0.12
    cax = fig.add_axes([cb_left / W, cb_y / H, cb_w / W, cb_h / H])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
    ticks = np.arange(vmin, vmax + 1e-9, 0.5)
    labels = [_fmt_growth(t) for t in ticks]
    labels[-1] = "≥" + labels[-1]          # darkest colour also covers anything above the top value
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=8, length=2.5, color=TEXT, labelcolor=TEXT)

    cax.text(0.5, 3.6, "How fast must the flooded area grow to count as a flood?",
             transform=cax.transAxes, ha="center", va="bottom", fontsize=10, weight="bold", color=TEXT)
    cax.text(0.5, 1.7, "Growth of the flooded area within one week (×2 = doubles)",
             transform=cax.transAxes, ha="center", va="bottom", fontsize=8, color="#555555")
    cax.text(0, -2.4, "◀ Small rise is enough", transform=cax.transAxes,
             ha="left", va="top", fontsize=8, color="#777777")
    cax.text(1, -2.4, "Needs a big jump ▶", transform=cax.transAxes,
             ha="right", va="top", fontsize=8, color="#777777")

    # 'Below threshold' swatch, left of the colour bar
    sw_w = 0.40
    nax = fig.add_axes([(cb_left - 1.00) / W, cb_y / H, sw_w / W, cb_h / H])
    nax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=NO_DATA, edgecolor="#9a9ea2", linewidth=0.5))
    nax.set_xlim(0, 1); nax.set_ylim(0, 1); nax.set_axis_off()
    nax.text(0.5, -0.9, NO_DATA_LABEL, transform=nax.transAxes, ha="center", va="top",
             fontsize=7.5, color=TEXT, linespacing=1.2)

    # ---------------- Lines legend: one row under the colour bar ----------------
    handles = [Line2D([], [], color=REGION_EDGE, linewidth=2.2, label="Study region (5 Jonglei counties)"),
               Line2D([], [], color=COUNTRY_EDGE, linewidth=1.6, label="National border"),
               Line2D([], [], color=STATE_EDGE, linewidth=0.7, label="State border")]
    if main_rivers is not None or rivers is not None:
        handles.append(Line2D([], [], color=RIVER, linewidth=1.8, label="Rivers"))
    if recurring is not None:
        handles.append(Patch(facecolor="none", edgecolor=RECURRING, hatch="....",
                             label="Frequently flooded land (Sudd)"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.40 / H), ncol=len(handles),
               fontsize=8, frameon=False, handlelength=2.2, columnspacing=2.0)

    # Source note
    fig.text(0.5, 0.08 / H, source, fontsize=6.5, color="#777777", ha="center", va="bottom")

    fig.savefig(out, dpi=300, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {os.path.abspath(out)}")
    return fig, ax


# ================= Run the map =================
if __name__ == "__main__":
    from capDBLexposure import load_admin_boundaries

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # County and state shapes
    admin1, admin2 = load_admin_boundaries()
    counties = admin2.copy()
    counties["county_id"] = counties["adm2_pcode"]

    # Threshold results from the threshold script
    summary = pd.read_csv("flood_outcome_county_summary.csv")
    counties = counties.merge(summary[["county_id", "threshold", "reliable"]], on="county_id", how="left")
    counties["reliable"] = counties["reliable"].fillna(False).astype(bool)

    plot_threshold_map(
        counties,
        states=admin1,
        rivers=None,        # replace with your rivers GeoDataFrame
        main_rivers=None,   # replace with your main Nile rivers (with a 'name' column)
        recurring=None,     # replace with your frequently flooded land
        neighbours=None,    # optional: neighbouring countries for grey context
        region_ids=["SS0302", "SS0303", "SS0305", "SS0306", "SS0310"],
    )
    plt.show()   # opens the map in a window; the saved PNG is the final version

# Plotting the ZOOMED version(study regions(Jonglei state))
"""
Flood hotspot map of the study region: where inside the five Jonglei counties flooding happens
most often on flood-event days (the days the threshold labels as flood events).

For every satellite pixel (~230 m) inside the five counties, it counts on how many flood-event
days that pixel was flooded, as a share of all flood-event days. Dark = flooded on most event days.

Inputs (same folder as this script):
    flood_joined_counties.parquet   flood pixels per county and day (cache made by the threshold script)
    flood_outcome_jonglei.parquet   regional flood label (1 = flood-event day), from the threshold script
    capDBLexposure.py               for the county and state boundaries

Output: flood_region_hotspots.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import PowerNorm, ListedColormap
import shapely
from matplotlib.cm import ScalarMappable

# setting
REGION_IDS = ["SS0302", "SS0303", "SS0305", "SS0306", "SS0310"]   # Ayod, Bor South, Duk, Fangak, Twic East
JOINED_FILE = "flood_joined_counties.parquet"
OUTCOME_FILE = "flood_outcome_jonglei.parquet"
START = "2002-01-01"                     # same start as the label
OUT_FILE = "flood_region_hotspots.png"
PIXEL_DEG = 10 / 4800      # size of one flood-mask pixel (~230 m)
GRID_DEG = 0.01            # heatmap cell size (~1 km)
SMOOTH_CELLS = 1.5         # how much to smooth (in heatmap cells); higher = softer heatmap

# Style
CMAP = ListedColormap(plt.get_cmap("magma_r")(np.linspace(0.0, 0.9, 256)), name="magma_trim").with_extremes(bad=(0, 0, 0, 0))
BACKGROUND = "#eef2f5"
BASE = "#f1efe9"            # county fill under the heatmap (not flooded on event days)
CONTEXT = "#f7f7f4"         # neighbouring counties
CONTEXT_EDGE = "#c8c8c8"
REGION_EDGE = "#1a1a1a"
RIVER = "#2b7bba"
TEXT = "#333333"
HALO = [pe.withStroke(linewidth=3, foreground="white")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def _pct(x):
    """0.0025 -> '0.25%', 0.01 -> '1%'"""
    return f"{x * 100:.2f}".rstrip("0").rstrip(".") + "%"


def flood_share_per_pixel(joined_file=JOINED_FILE, outcome_file=OUTCOME_FILE,
                          region_ids=REGION_IDS, start=START):
    """Share of flood-event days on which each pixel in the region was flooded."""
    label = pd.read_parquet(outcome_file)["flood"]
    label.index = pd.to_datetime(label.index)
    event_days = label[(label == 1) & (label.index >= start)].index
    print(f"Flood-event days: {len(event_days)}")

    print("Reading flood pixels for the five counties (can take a minute)...")
    px = pd.read_parquet(joined_file, columns=["county_id", "date", "lat", "lon"],
                         filters=[("county_id", "in", list(region_ids))])
    px["date"] = pd.to_datetime(px["date"])
    px = px[px["date"].isin(event_days)].drop_duplicates(["date", "lat", "lon"])

    share = px.groupby(["lon", "lat"]).size().rename("n").reset_index()
    share["share"] = share["n"] / len(event_days)
    print(f"Pixels flooded on at least one event day: {len(share):,}")
    return share, len(event_days)


def heat_grid(share, region_geom, grid=GRID_DEG, pixel=PIXEL_DEG, sigma=SMOOTH_CELLS):
    """Turn per-pixel shares into a smooth heatmap grid clipped to the region.
    Each ~1 km cell = average share over all pixels in it (pixels never flooded count as 0),
    then a Gaussian blur so hotspots read as smooth areas."""
    minx, miny, maxx, maxy = region_geom.bounds
    nx, ny = int(np.ceil((maxx - minx) / grid)), int(np.ceil((maxy - miny) / grid))
    ix = ((share["lon"].values - minx) / grid).astype(int).clip(0, nx - 1)
    iy = ((share["lat"].values - miny) / grid).astype(int).clip(0, ny - 1)
    total = np.zeros((ny, nx))
    np.add.at(total, (iy, ix), share["share"].values)
    val = total / max((grid / pixel) ** 2, 1.0)

    r = max(int(3 * sigma), 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    val = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, val)
    val = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, val)

    xs = minx + (np.arange(nx) + 0.5) * grid
    ys = miny + (np.arange(ny) + 0.5) * grid
    X, Y = np.meshgrid(xs, ys)
    val = np.where(shapely.contains_xy(region_geom, X, Y), val, np.nan)
    return val, X, Y, (minx, minx + nx * grid, miny, miny + ny * grid)


def _north_arrow(ax, x=0.95, y=0.97):
    ax.annotate("N", xy=(x, y), xytext=(x, y - 0.07), xycoords="axes fraction", ha="center",
                va="center", fontsize=11, weight="bold", color=TEXT,
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.5))


def _scale_bar(ax, length_km=50):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    length_deg = length_km / (111.32 * np.cos(np.deg2rad((y0 + y1) / 2)))
    xs, ys = x1 - (x1 - x0) * 0.08 - length_deg, y0 + (y1 - y0) * 0.04
    h = (y1 - y0) * 0.008
    for i in range(2):
        ax.add_patch(plt.Rectangle((xs + i * length_deg / 2, ys), length_deg / 2, h,
                                   facecolor=TEXT if i == 0 else "white", edgecolor=TEXT,
                                   linewidth=0.8, zorder=20))
    for frac, label in [(0, "0"), (0.5, f"{length_km // 2}"), (1, f"{length_km} km")]:
        ax.text(xs + frac * length_deg, ys + h * 2.2, label, ha="center", va="bottom",
                fontsize=7, color=TEXT, zorder=20)


def plot_hotspot_map(counties, states, share, n_events, rivers=None, main_rivers=None,
                     region_ids=REGION_IDS, out=OUT_FILE,
                     source="Source: NASA MODIS/VIIRS flood masks (~230 m pixels) 2002–2025; "
                            "OCHA admin boundaries"):
    counties = counties.to_crs(4326).copy()
    states = states.to_crs(4326)
    country = states.dissolve()
    region = counties[counties["county_id"].isin(region_ids)]
    others = counties[~counties["county_id"].isin(region_ids)]
    region_shape = region.dissolve()

    minx, miny, maxx, maxy = region.total_bounds
    padx, pady = (maxx - minx) * 0.30, (maxy - miny) * 0.10
    view = (minx - padx, maxx + padx, miny - pady, maxy + pady)

    # Figure sized to the map's own shape -> no empty space around it
    # Layout, top to bottom (inches): title + inset | map | colour bar | source
    map_w = 8.0
    map_h = map_w * (view[3] - view[2]) / (view[1] - view[0])
    side, top_h, bot_h = 0.3, 0.95, 1.35
    in_w, in_h, gap = 1.35, 0.85, 0.10                             # South Sudan inset size + gap to the map
    W, H = map_w + 2 * side + gap + in_w, map_h + top_h + bot_h      # extra room on the right for the inset
    fig = plt.figure(figsize=(W, H), facecolor=BACKGROUND)
    ax = fig.add_axes([side / W, bot_h / H, map_w / W, map_h / H])
    ax.set_facecolor(BACKGROUND)

    # Context counties
    others.plot(ax=ax, color=CONTEXT, edgecolor=CONTEXT_EDGE, linewidth=0.6, zorder=1)
    for _, r in others.iterrows():
        p = r.geometry.representative_point()
        if view[0] < p.x < view[1] and view[2] < p.y < view[3]:
            ax.annotate(r["adm2_name"], (p.x, p.y), ha="center", va="center", fontsize=7,
                        color="#9a9a9a", style="italic", zorder=2)

    # Study counties: shadow + base fill
    region_shape.translate(0.03, -0.03).plot(ax=ax, color="black", alpha=0.15, linewidth=0, zorder=2)
    region.plot(ax=ax, color=BASE, edgecolor="none", zorder=3)

    # Heatmap: smooth ~1 km grid, clipped to the five counties
    val, X, Y, extent = heat_grid(share, region_shape.geometry.iloc[0])
    vmax = float(np.nanquantile(val, 0.995))
    vmax = max(0.01, np.ceil(vmax * 100) / 100)                     # round up to 1%
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)                  # stretches low values so patterns show
    ax.imshow(val, origin="lower", extent=extent, cmap=CMAP, norm=norm,
              interpolation="bilinear", zorder=4)
    ax.contour(X, Y, np.nan_to_num(val), levels=[0.5 * vmax], colors="#2b0a3d",
               linewidths=0.6, alpha=0.7, zorder=5)                 # outline of the main hotspots

    # Rivers(we try if we can show the rivers in the area)
    if rivers is not None:
        rivers.to_crs(4326).plot(ax=ax, color=RIVER, linewidth=0.7, alpha=0.8, zorder=5)
    if main_rivers is not None:
        mr = main_rivers.to_crs(4326)
        mr.plot(ax=ax, color="white", linewidth=3.2, zorder=6)
        mr.plot(ax=ax, color=RIVER, linewidth=1.8, zorder=7)

    # Borders
    region.boundary.plot(ax=ax, color="white", linewidth=1.4, zorder=8)
    region.boundary.plot(ax=ax, color="#555555", linewidth=0.5, zorder=8)
    states.boundary.plot(ax=ax, color="#8a8a8a", linewidth=0.9, linestyle="--", zorder=8)
    region_shape.boundary.plot(ax=ax, color=REGION_EDGE, linewidth=2.2, zorder=9)

    # County names only (no text boxes)
    for _, r in region.iterrows():
        p = r.geometry.representative_point()
        ax.annotate(r["adm2_name"], (p.x, p.y), ha="center", va="center", fontsize=10,
                    weight="bold", color=TEXT, zorder=16, path_effects=HALO)

    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    _north_arrow(ax)
    _scale_bar(ax)

    # Inset: where the region is in South Sudan (top-right corner, above the map)
    # Attached to the map: right next to its top-right corner, outside the map and away from the title
    iax = ax.inset_axes([1 + gap / map_w, 1 - in_h / map_h, in_w / map_w, in_h / map_h])
    iax.set_anchor("NW")                                            # top-aligned with the map
    iax.set_facecolor("white")
    country.plot(ax=iax, color="#e4e4e0", edgecolor="#666666", linewidth=0.6)
    states.boundary.plot(ax=iax, color="#aaaaaa", linewidth=0.3)
    region_shape.plot(ax=iax, color="#b5367a", edgecolor="black", linewidth=0.6)
    iax.set_aspect("equal")
    iax.set_xticks([]); iax.set_yticks([]); iax.set_xlabel(""); iax.set_ylabel("")
    for sp in iax.spines.values():
        sp.set_edgecolor("#999999"); sp.set_linewidth(0.8)
    iax.set_title("South Sudan", fontsize=7.5, color=TEXT, pad=2)

    # Title, subtitle (left, attached to the map so they stay aligned with it)
    ax.text(0, 1 + (top_h - 0.22) / map_h, "Where flooding happens on flood-event days",
            transform=ax.transAxes, fontsize=15, weight="bold", color=TEXT, ha="left", va="top")
    ax.text(0, 1 + (top_h - 0.55) / map_h, f"Share of the {n_events:,} flood-event days (2002–2025) on which "
            "each location was flooded, five Jonglei counties", transform=ax.transAxes,
            fontsize=9.5, color="#555555", ha="left", va="top", linespacing=1.3)

    # Colour bar: centred under the map 
    cb_w = 0.55 * map_w
    cax = fig.add_axes([(side + (map_w - cb_w) / 2 + 0.3) / W, 0.70 / H, cb_w / W, 0.13 / H])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")

    # Round, easy-to-read numbers (no odd values like 0.7% or 1.3%)
    nice = np.array([0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 50]) / 100
    ticks = nice[nice <= vmax + 1e-9]
    if not np.isclose(ticks[-1], vmax):
        ticks = np.append(ticks, vmax)
    labels = [_pct(t) for t in ticks]
    labels[-1] = "≥" + labels[-1]          
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=8, length=2.5, color=TEXT, labelcolor=TEXT)

    # Clear sentence above the bar
    cax.text(0.5, 3.4, "How often was this spot flooded?", transform=cax.transAxes,
             ha="center", va="bottom", fontsize=10, weight="bold", color=TEXT)
    cax.text(0.5, 1.6, f"% of the {n_events:,} flood-event days (≈1 km cells, smoothed)",
             transform=cax.transAxes, ha="center", va="bottom", fontsize=8, color="#555555")

    # Plain-language ends below the bar
    cax.text(0, -2.3, "◀ Rarely flooded", transform=cax.transAxes,
             ha="left", va="top", fontsize=8, color="#777777")
    cax.text(1, -2.3, "Most often flooded ▶", transform=cax.transAxes,
             ha="right", va="top", fontsize=8, color="#777777")

    # Source note (own line at the very bottom)
    fig.text((side + map_w / 2) / W, 0.08 / H, source + ". Flood-event days: regional threshold label.",
             fontsize=6.5, color="#777777", ha="center", va="bottom")

    fig.savefig(out, dpi=300, facecolor=BACKGROUND, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {os.path.abspath(out)}")
    return fig, ax


# Run the map
if __name__ == "__main__":
    from capDBLexposure import load_admin_boundaries

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    admin1, admin2 = load_admin_boundaries()
    counties = admin2.copy()
    counties["county_id"] = counties["adm2_pcode"]

    share, n_events = flood_share_per_pixel()
    plot_hotspot_map(
        counties,
        states=admin1,
        share=share,
        n_events=n_events,
        rivers=None,        # replace with your rivers GeoDataFrame
        main_rivers=None,   # replace with your main Nile rivers
    )
    plt.show()   # opens the map in a window, if you wantit, the saved PNG is the final version!