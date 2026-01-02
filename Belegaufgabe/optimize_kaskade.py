# optimize_cstr_cascade.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import cantera as ct
from multiprocessing import freeze_support

from Kaskade_Klasse import CSTRCascadeModel, cm

def main():
    os.chdir(os.path.dirname(__file__))

    tc = 800.0
    p = 1 * ct.one_atm
    length = 0.3 * cm
    mass_flow_rate = 1e-6
    yaml_file = "methane_pox_on_pt.yaml"

    n_cstr = 201

    bounds = [
        (1000.0, 2000.0),  # A/V [1/cm]
        (1.0, 3.0),        # d [cm]
        (0.2, 0.5),        # porosity [-]
    ]

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

    history = []
    max_iter = 100

    def callback(xk, convergence=None):
        if len(history) < max_iter:
            # optional: store xk only; storing fx would cost extra evaluation unless you cache inside objective
            history.append(xk.copy())

    solution = optimize.differential_evolution(
        model.objective_CH4,
        bounds=bounds,
        disp=True,
        maxiter=max_iter,
        callback=callback,
        workers=-1,
        updating="deferred",
    )

    print(solution)
    print("Optimum solution:")
    print(f"CH4 = {solution.fun:.6f}")
    print(f"A/V = {solution.x[0]:.1f} 1/cm")
    print(f"d = {solution.x[1]:.3f} cm")
    print(f"Porosity = {solution.x[2]:.4f}")
    print(f"n_CSTR = {n_cstr}")

if __name__ == "__main__":
    freeze_support()  # safe on Windows; harmless otherwise
    main()
