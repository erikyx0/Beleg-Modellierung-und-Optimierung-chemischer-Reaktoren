# optimize_kaskade_pymoo_nsga2_clean.py
# pip install pymoo

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from Kaskade_Klasse import CSTRCascadeModel, cm


class CatMultiObjectiveProblem(Problem):
    """
    Minimize (scaled):
      f1 = CH4 / CH4_SCALE
      f2 = V_cat / VCAT_SCALE
    Optional constraint:
      V_cat <= Vcat_max  ->  G = V_cat - Vcat_max <= 0
    """

    def __init__(self, model: CSTRCascadeModel, xl, xu,
                 CH4_SCALE=1.0, VCAT_SCALE=1e-9, Vcat_max=None):
        self.model = model
        self.Vcat_max = Vcat_max
        self.CH4_SCALE = float(CH4_SCALE)
        self.VCAT_SCALE = float(VCAT_SCALE)

        n_ieq = 1 if Vcat_max is not None else 0
        super().__init__(
            n_var=3,
            n_obj=2,
            n_ieq_constr=n_ieq,
            xl=np.array(xl, dtype=float),
            xu=np.array(xu, dtype=float),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        n = X.shape[0]
        F = np.empty((n, 2), dtype=float)
        G = np.empty((n, 1), dtype=float) if self.Vcat_max is not None else None

        for i in range(n):
            av, d_cm, eps = X[i, :]
            try:
                res = self.model.simulate(av, d_cm, eps, return_profile=False)
                ch4 = float(res["CH4"])
                vcat = float(res["V_cat"])
            except Exception:
                # Differenzierende Strafe -> hilft aus Failure-Regionen rauszukommen
                ch4 = 1e3
                vcat = 1e3 + 1e-6 * (av + d_cm + eps)

            # objectives MUST be minimized
            F[i, 0] = ch4 / (self.CH4_SCALE + 1e-30)
            F[i, 1] = vcat / (self.VCAT_SCALE + 1e-30)

            if G is not None:
                G[i, 0] = vcat - float(self.Vcat_max)

        out["F"] = F
        if G is not None:
            out["G"] = G


def main():
    os.chdir(os.path.dirname(__file__))
    out_dir = "../Auswertung"
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # Model setup
    # -----------------------------
    tc = 800.0
    p = 1 * ct.one_atm
    length = 0.3 * cm
    mass_flow_rate = 1e-6
    yaml_file = "methane_pox_on_pt.yaml"
    n_cstr = 200  # test; später hoch

    model = CSTRCascadeModel(
        yaml_file=yaml_file,
        tc_C=tc,
        p_Pa=p,
        length_m=length,
        mass_flow_rate_kg_s=mass_flow_rate,
        n_cstr=n_cstr,
        gas_comp="CH4:1, O2:0.6, AR:0.1",
        energy_enabled=False,
        surface_name="Pt_surf",
        gas_name="gas",
    )

    # bounds: [A/V (1/cm), d (cm), porosity (-)]
    xl = [1000.0, 1.0, 0.2]
    xu = [2000.0, 3.0, 0.5]

    # Scaling (nur für Optimierung)
    CH4_SCALE = 1.0
    VCAT_SCALE = 1e-9

    # Optional hard constraint
    Vcat_max = None  # z.B. 1e-9

    problem = CatMultiObjectiveProblem(
        model, xl=xl, xu=xu,
        CH4_SCALE=CH4_SCALE, VCAT_SCALE=VCAT_SCALE,
        Vcat_max=Vcat_max
    )

    algo = NSGA2(
        pop_size=50,
        crossover=SBX(prob=0.9, eta=8),
        mutation=PM(eta=10),
        eliminate_duplicates=True,
    )
    termination = get_termination("n_gen", 100)

    res = minimize(problem, algo, termination, seed=1, verbose=True)

    X = res.X            # decision variables
    F_scaled = res.F     # scaled objectives

    # -----------------------------
    # Recompute TRUE objectives for export/plot
    # (because res.F is scaled!)
    # -----------------------------
    CH4_true = np.empty(X.shape[0], dtype=float)
    Vcat_true = np.empty(X.shape[0], dtype=float)
    fail = 0

    for i, (av, d_cm, eps) in enumerate(X):
        # geometry-based V_cat (always available)
        A_cs = (np.pi / 4.0) * (d_cm * cm) ** 2
        V_bed = A_cs * length
        vcat_geom = (1.0 - eps) * V_bed

        try:
            r = model.simulate(av, d_cm, eps, return_profile=False)
            CH4_true[i] = float(r["CH4"])
        except Exception:
            CH4_true[i] = np.nan

        Vcat_true[i] = float(vcat_geom)

    print("\nDiagnostics (TRUE):")
    print("  fails:", fail, "/", len(X))
    print("  CH4 min/max:", np.nanmin(CH4_true), np.nanmax(CH4_true))
    print("  Vcat min/max:", np.nanmin(Vcat_true), np.nanmax(Vcat_true))

    # -----------------------------
    # Export CSV (TRUE values)
    # -----------------------------
    csv_path = os.path.join(out_dir, "pareto_pymoo_nsga2_TRUE.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["CH4_out", "V_cat_m3", "A_over_V_1_per_cm", "diameter_cm", "porosity"])
        for i in range(len(X)):
            w.writerow([CH4_true[i], Vcat_true[i], X[i, 0], X[i, 1], X[i, 2]])

    # -----------------------------
    # Plot Pareto (TRUE)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.scatter(Vcat_true, CH4_true, s=28)
    ax.set_xlabel("V_cat (m³)")
    ax.set_ylabel("CH4_out (mol/mol)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Pareto-Front (NSGA-II): CH4_out vs. V_cat")
    fig.savefig(os.path.join(out_dir, "pareto_front_pymoo_TRUE.png"), dpi=200)

    print("\nSaved:")
    print("  CSV :", csv_path)
    print("  PNG :", os.path.join(out_dir, "pareto_front_pymoo_TRUE.png"))


if __name__ == "__main__":
    main()
