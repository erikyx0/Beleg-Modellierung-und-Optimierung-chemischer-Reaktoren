# plot_results.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = "optimization_history.csv"

def plot_history(csv_path=CSV):
    df = pd.read_csv(csv_path)

    it = df["iteration"].to_numpy()
    ch4 = df["CH4"].to_numpy()
    av = df["cat_area_per_vol_1_per_cm"].to_numpy()
    d = df["diameter_cm"].to_numpy()
    eps = df["porosity"].to_numpy()

    # "best so far" (hilft enorm beim Lesen von Optimierungsverläufen)
    ch4_best = np.minimum.accumulate(ch4)

    # --------- Figure 1: Überblick (4 Panels, shared x) ----------
    fig, ax = plt.subplots(
        4, 1, sharex=True, figsize=(10, 9), constrained_layout=True
    )

    # CH4
    ax[0].plot(it, ch4, marker="o", markersize=3, linewidth=1, alpha=0.8, label="CH4 (aktuell)")
    ax[0].plot(it, ch4_best, linewidth=2, label="CH4 (best so far)")
    ax[0].set_ylabel("CH4 (mol/mol)")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend(loc="upper right")

    # d
    ax[1].plot(it, d, marker="x", markersize=3, linewidth=1, alpha=0.9)
    ax[1].set_ylabel("d (cm)")
    ax[1].grid(True, which="both", alpha=0.3)

    # A/V
    ax[2].plot(it, av, marker="o", markersize=3, linewidth=1, alpha=0.8)
    ax[2].set_ylabel("A/V (1/cm)")
    ax[2].grid(True, which="both", alpha=0.3)

    # Porosity
    ax[3].plot(it, eps, marker="x", markersize=3, linewidth=1, alpha=0.9)
    ax[3].set_ylabel("Porosität (-)")
    ax[3].set_xlabel("Iteration")
    ax[3].grid(True, which="both", alpha=0.3)

    # Markiere bestes gefundenes Design
    best_idx = int(np.argmin(ch4))
    for a in ax:
        a.axvline(it[best_idx], linewidth=1, alpha=0.4)

    fig.suptitle("Optimierungsverlauf (Differential Evolution)", y=1.01)
    fig.savefig("img/plot_cstr_overview.png", dpi=200)

    # --------- Figure 2: Pareto-Feeling: CH4 vs Parameter (scatter) ----------
    fig2, ax2 = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)

    ax2[0].scatter(d, ch4, s=18, alpha=0.7)
    ax2[0].set_xlabel("d (cm)")
    ax2[0].set_ylabel("CH4 (mol/mol)")
    ax2[0].grid(True, alpha=0.3)

    ax2[1].scatter(av, ch4, s=18, alpha=0.7)
    ax2[1].set_xlabel("A/V (1/cm)")
    ax2[1].grid(True, alpha=0.3)

    ax2[2].scatter(eps, ch4, s=18, alpha=0.7)
    ax2[2].set_xlabel("Porosität (-)")
    ax2[2].grid(True, alpha=0.3)

    fig2.suptitle("Zusammenhang: Parameter → CH4", y=1.03)
    fig2.savefig("img/plot_cstr_scatter.png", dpi=200)

    # --------- Optional: nur CH4 groß (für Bericht) ----------
    fig3, ax3 = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax3.plot(it, ch4, marker="o", markersize=3, linewidth=1, alpha=0.8, label="CH4 (aktuell)")
    ax3.plot(it, ch4_best, linewidth=2, label="CH4 (best so far)")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("CH4 (mol/mol)")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(loc="upper right")
    ax3.set_title("Konvergenzverlauf")
    fig3.savefig("img/plot_cstr_ch4.png", dpi=200)

    plt.show()

if __name__ == "__main__":
    plot_history()
