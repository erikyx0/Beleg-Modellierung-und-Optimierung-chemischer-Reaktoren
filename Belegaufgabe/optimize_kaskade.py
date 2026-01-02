# optimize_cstr_cascade.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import cantera as ct

# eigene Klasse aus subdatei
from Kaskade_Klasse import CSTRCascadeModel, cm

os.chdir(os.path.dirname(__file__))

tc = 800.0
p = 1 * ct.one_atm
length = 0.3 * cm
mass_flow_rate = 1e-6
yaml_file = "methane_pox_on_pt.yaml"

# number of CSTRs
n_cstr = 201

cat_area_per_vol_min = 1000.0
cat_area_per_vol_max = 2000.0
diameter_min = 1.0
diameter_max = 3.0
porosity_min = 0.2
porosity_max = 0.5

bounds = [
    (cat_area_per_vol_min, cat_area_per_vol_max),
    (diameter_min, diameter_max),
    (porosity_min, porosity_max),
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

results = []
max_iter = 100

def callback(xk, convergence=None):
    if len(results) < max_iter:
        results.append(xk.copy())

solution = optimize.differential_evolution(
    model.objective_CH4,
    bounds=bounds,
    disp=True,
    maxiter=max_iter,
    callback=callback,
)

print(solution)
print("Optimum solution:")
print(f"CH4 = {solution.fun:.6f}")
print(f"A/V = {solution.x[0]:.1f} 1/cm")
print(f"d = {solution.x[1]:.3f} cm")
print(f"Porosity = {solution.x[2]:.4f}")
print(f"n_CSTR = {n_cstr}")

# post-processing
obj_values_iterative = []
cat_area_per_vol_values_iterative = []
diameter_values_iterative = []
porosity_values_iterative = []

for x in results:
    obj_values_iterative.append(model.objective_CH4(x))
    cat_area_per_vol_values_iterative.append(x[0])
    diameter_values_iterative.append(x[1])
    porosity_values_iterative.append(x[2])

iters = np.arange(1, len(results) + 1)

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel("Iterations")
ax1.set_ylabel("CH4 (mol/mol)")
ax1.plot(iters, obj_values_iterative, label="CH4 (Iterative)", marker="o")
ax2 = ax1.twinx()
ax2.set_ylabel("d (cm)")
ax2.plot(iters, diameter_values_iterative, label="d (Iterative)", marker="x")
ax2.set_ylim(diameter_min, diameter_max)
fig1.tight_layout()
plt.title("Iterative Results")
fig1.legend(loc="lower right", bbox_to_anchor=(1, 0.5), bbox_transform=ax1.transAxes)
plt.savefig("plot-Aufgabe1_cstr-v1.png", dpi=150)

fig2, ax3 = plt.subplots(figsize=(10, 6))
ax3.set_ylabel("A/V (1/cm)")
ax3.plot(iters, cat_area_per_vol_values_iterative, label="A/V (Iterative)", marker="o")
ax3.set_ylim(cat_area_per_vol_min, cat_area_per_vol_max)
ax4 = ax3.twinx()
ax4.set_ylabel("Porosity (-)")
ax4.plot(iters, porosity_values_iterative, label="Porosity (Iterative)", marker="x")
ax4.set_ylim(porosity_min, porosity_max)
fig2.tight_layout()
plt.title("Iterative Results")
fig2.legend(loc="lower right", bbox_to_anchor=(1, 0.5), bbox_transform=ax3.transAxes)
plt.savefig("plot-Aufgabe1_cstr-v2.png", dpi=150)
