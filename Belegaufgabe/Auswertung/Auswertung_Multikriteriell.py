import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

colors = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#937860",
    "#64B5CD",
    "#7d7d7d",
]

df_pareto = pd.read_csv("pareto_CH4_vs_Vcat.csv")
df_all = pd.read_csv("all_evaluated_points.csv")

pareto_ch4_out = df_pareto["CH4_out"]
all_ch4_out    = df_all["CH4_out"]

pareto_Vcat_out = df_pareto["Vcat_m3"]
all_Vcat_out    = df_all["Vcat_m3"]

# === Hauptplot ===
fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(
    all_ch4_out,
    all_Vcat_out,
    label="Alle Punkte",
    color=colors[-1],
    marker="+",
    s=20
)

ax.scatter(
    pareto_ch4_out,
    pareto_Vcat_out,
    label="Pareto-Front",
    color="red",
    marker="x",
    s=20
)

ax.set_xlabel(r"CH$_4$-out")
ax.set_ylabel("Katalysatorvolumen")

#ax.set_xlim(0, 0.003)
#ax.set_ylim(1e-7, 4.5e-7)

ax.grid()
ax.legend(loc="upper left")

# === Zoom-Inset ===
axins = inset_axes(
    ax,
    width="50%",      # Größe des Zoom-Fensters
    height="70%",
    loc="upper right", # Position im Plot
    borderpad=1
)

axins.scatter(
    all_ch4_out,
    all_Vcat_out,
    color=colors[-1],
    marker="+",
    s=12
)

axins.scatter(
    pareto_ch4_out,
    pareto_Vcat_out,
    color="red",
    marker="x",
    s=16
)

axins.tick_params(
    left=False,
    bottom=False,
    labelleft=False,
    labelbottom=False
)

# >>> ZOOM-BEREICH (hier ggf. feinjustieren) <<<
axins.set_xlim(0.0002, 0.003)
axins.set_ylim(1e-7, 5e-7)

axins.grid()

# Verbindung Hauptplot ↔ Zoom
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4")

#plt.show()
plt.savefig("img/Pareto_front_zoom", dpi=500)