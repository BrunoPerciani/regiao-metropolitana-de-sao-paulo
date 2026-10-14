"""
Bivariate choropleth maps: NDVI x Population Density (PD), 2010 and 2022,
plus a bivariate delta map (dNDVI x dPD).

Input : two CSV grids (250 m cells) with columns
        left, top, right, bottom, row_index, col_index, NDVI, DP
Output: Fig_bivariate_NDVI_PD_2010_2022.(png|pdf)
        Fig_bivariate_delta_NDVI_PD.(png|pdf)
        bivariate_class_shares.csv
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle, Patch

F2010, F2022 = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "."

# ---------------------------------------------------------------- palette
# rows = PD (0 low -> 2 high), cols = NDVI (0 low -> 2 high)
PAL = np.array([
    ["#e8e8e8", "#b5d9a8", "#4c9a55"],   # low PD
    ["#e4b1c6", "#b3a68a", "#5c7a3e"],   # medium PD
    ["#c1527e", "#8a5a44", "#4b3f1f"],   # high PD
])
NODATA = "#ffffff"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

# ---------------------------------------------------------------- data
a = pd.read_csv(F2010)[["row_index", "col_index", "left", "top", "NDVI", "DP"]]
b = pd.read_csv(F2022)[["row_index", "col_index", "NDVI", "DP"]]
df = a.merge(b, on=["row_index", "col_index"], suffixes=("_2010", "_2022"))
# Delta on within-year percentile ranks (0-1): robust to unit / sensor offsets
# between the two years. Raw differences kept for reference.
df["dNDVI_raw"] = df.NDVI_2022 - df.NDVI_2010
df["dPD_raw"] = df.DP_2022 - df.DP_2010
df["dNDVI"] = df.NDVI_2022.rank(pct=True) - df.NDVI_2010.rank(pct=True)
df["dPD"] = df.DP_2022.rank(pct=True) - df.DP_2010.rank(pct=True)
DELTA_T = 0.10   # +/- 10 percentile points = "stable" band

nrow, ncol = int(df.row_index.max()), int(df.col_index.max())
x0, y1 = df.left.min(), df.top.max()
CELL = 250.0
extent = [x0, x0 + ncol * CELL, y1 - nrow * CELL, y1]


def tercile(s):
    """0/1/2 class by terciles of the valid values (NaN stays NaN)."""
    q1, q2 = s.quantile([1 / 3, 2 / 3])
    c = pd.Series(np.nan, index=s.index)
    c[s <= q1] = 0
    c[(s > q1) & (s <= q2)] = 1
    c[s > q2] = 2
    return c, (q1, q2)


def to_rgb_grid(cx, cy):
    img = np.ones((nrow, ncol, 3))
    ok = cx.notna() & cy.notna()
    r = df.loc[ok, "row_index"].to_numpy() - 1
    c = df.loc[ok, "col_index"].to_numpy() - 1
    lut = np.array([[to_rgb(PAL[i, j]) for j in range(3)] for i in range(3)])
    img[r, c] = lut[cy[ok].astype(int).to_numpy(), cx[ok].astype(int).to_numpy()]
    return img, ok


def legend(ax, xlab, ylab, xt=("Low", "Medium", "High"), yt=("Low", "Medium", "High")):
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((j, i), 1, 1, fc=PAL[i, j], ec="white", lw=1))
    ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.set_aspect("equal")
    ax.set_xticks([.5, 1.5, 2.5]); ax.set_xticklabels(xt, fontsize=7)
    ax.set_yticks([.5, 1.5, 2.5]); ax.set_yticklabels(yt, fontsize=7)
    ax.set_xlabel(xlab + "  →", fontsize=8)
    ax.set_ylabel(ylab + "  →", fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def decorate(ax, title):
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(.6); s.set_color("#666")
    # scale bar 0-25-50 km
    xs, ys = extent[0] + 0.05 * (extent[1] - extent[0]), extent[2] + 0.05 * (extent[3] - extent[2])
    for k in range(2):
        ax.add_patch(Rectangle((xs + k * 25000, ys), 25000, 1200,
                               fc="black" if k == 0 else "white", ec="black", lw=.6))
    for k, t in enumerate(["0", "25", "50 km"]):
        ax.text(xs + k * 25000, ys + 2000, t, ha="center", va="bottom", fontsize=7)
    # north arrow
    xn, yn = extent[1] - 0.05 * (extent[1] - extent[0]), extent[3] - 0.17 * (extent[3] - extent[2])
    ax.annotate("N", xy=(xn, yn + 9000), xytext=(xn, yn), ha="center", fontsize=9,
                fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))


def change_class(s, t=None):
    t = DELTA_T if t is None else t
    c = pd.Series(np.nan, index=s.index)
    c[s < -t] = 0
    c[(s >= -t) & (s <= t)] = 1
    c[s > t] = 2
    return c, (-t, t)


shares = []

# ---------------------------------------------------------------- Figure 1: 2010 | 2022
fig = plt.figure(figsize=(12, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.42], wspace=0.06)
breaks = {}
for k, yr in enumerate(["2010", "2022"]):
    cx, bx = tercile(df[f"NDVI_{yr}"])
    cy, by = tercile(df[f"DP_{yr}"])
    breaks[yr] = (bx, by)
    img, ok = to_rgb_grid(cx, cy)
    ax = fig.add_subplot(gs[k])
    ax.imshow(img, extent=extent, interpolation="nearest")
    decorate(ax, f"({'ab'[k]}) {yr}")
    t = pd.crosstab(cy[ok], cx[ok], normalize=True) * 100
    for i in range(3):
        for j in range(3):
            shares.append(dict(map=yr, PD_class=["Low", "Medium", "High"][i],
                               NDVI_class=["Low", "Medium", "High"][j], pct_cells=round(t.loc[i, j], 2)))

lax = fig.add_axes([0.80, 0.38, 0.11, 0.30])
legend(lax, "NDVI", "Population density")
fig.legend(handles=[Patch(fc=NODATA, ec="#999", label="No data")],
           loc="upper left", bbox_to_anchor=(0.775, 0.26), frameon=False, fontsize=8)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/Fig_bivariate_NDVI_PD_2010_2022.{ext}", dpi=600, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: delta
cx, bx = change_class(df.dNDVI)
cy, by = change_class(df.dPD)
breaks["delta"] = (bx, by)
img, ok = to_rgb_grid(cx, cy)
fig = plt.figure(figsize=(7.4, 4.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.40], wspace=0.06)
ax = fig.add_subplot(gs[0])
ax.imshow(img, extent=extent, interpolation="nearest")
decorate(ax, "Δ 2010–2022")
lax = fig.add_axes([0.78, 0.40, 0.17, 0.30])
legend(lax, "\u0394NDVI", "\u0394Population density",
       xt=("Loss", "Stable", "Gain"), yt=("Decrease", "Stable", "Increase"))
fig.legend(handles=[Patch(fc=NODATA, ec="#999", label="No data")],
           loc="upper left", bbox_to_anchor=(0.72, 0.28), frameon=False, fontsize=8)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/Fig_bivariate_delta_NDVI_PD.{ext}", dpi=600, bbox_inches="tight")
plt.close(fig)

t = pd.crosstab(cy[ok], cx[ok], normalize=True) * 100
for i in range(3):
    for j in range(3):
        shares.append(dict(map="delta", PD_class=["Decrease", "Stable", "Increase"][i],
                           NDVI_class=["Loss", "Stable", "Gain"][j], pct_cells=round(t.loc[i, j], 2)))
pd.DataFrame(shares).to_csv(f"{OUT}/bivariate_class_shares.csv", index=False)

for k, ((bx1, bx2), (by1, by2)) in breaks.items():
    print(f"{k:6s} NDVI breaks: {bx1:.3f}, {bx2:.3f} | PD breaks: {by1:.3f}, {by2:.3f}")
print(pd.DataFrame(shares).to_string())
