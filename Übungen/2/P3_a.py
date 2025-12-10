import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import cantera as ct
import os

os.chdir(os.path.dirname(__file__))     # changing working directory to this file's directory, 
                                        # allows output to be placed in the same directory without the need for absolute paths

# unit conversion factors to SI
cm = 0.01
minute = 60.0

#######################################################################
# Input Parameters
#######################################################################

tc = 800.0  # Temperature in Celsius
p = 1*ct.one_atm # Pressure
length = 0.3 * cm  # Catalyst bed length
mass_flow_rate = 1.e-6 # Mass flow rate in kg/s

# input file containing the surface reaction mechanism
yaml_file = 'methane_pox_on_pt.yaml' # Input File 

#######################################################################
# Optimization Parameters
#######################################################################

cat_area_per_vol_min = 1000.0   # min bound for Catalyst particle surface area per unit volume
cat_area_per_vol_max = 2000.0   # max bound for Catalyst particle surface area per unit volume
diameter_min = 1.0   # min bound for diameter in cm
diameter_max = 3.0   # max bound for diameter in cm
porosity_min = 0.2   # min bound for porosity
porosity_max = 0.5   # max bound for porosity

#####################################################################

# Objective function to minimize
def objective(params):
    cat_area_per_vol, diameter, porosity = params
    t = tc + 273.15  # convert to Kelvin
    cat_apv = cat_area_per_vol /cm # convert to SI
    area = math.pi/4 * (diameter * cm)**2  # Catalyst bed area
    
    # import the model and set the initial conditions
    surf = ct.Interface(yaml_file, 'Pt_surf') # initial surface conditions 
    surf.TP = t, p # set surface temperature and pressure
    gas = surf.adjacent['gas'] # get initial gas phase conditions
    gas.TPX = t, p, 'CH4:1, O2:0.6, AR:0.1' # set gas phase  
    
    # create a new reactor
    r = ct.FlowReactor(gas) # steady state plug flow reactor
    r.area = area # cross sectional area of the reactor
    r.surface_area_to_volume_ratio = cat_apv * porosity
    r.mass_flow_rate = mass_flow_rate
    r.energy_enabled = False # Energy equation inactive
    
    # Add the reacting surface to the reactor
    rsurf = ct.ReactorSurface(surf, r)
    
    sim = ct.ReactorNet([r]) # Integrate along the length of reactor 
    
    max_c = 0.0
    while sim.distance < length:
        sim.step()
        max_c=max(max_c,rsurf.kinetics.coverages[9])

    ch4 = r.thermo['CH4'].X[0]
    return ch4  
    # return max_c

# Bounds for parameters
bounds = [(cat_area_per_vol_min, cat_area_per_vol_max), 
          (diameter_min, diameter_max), 
          (porosity_min, porosity_max)]

# Iteratively collect the results during optimization
results = []
max_iter = 100
val = 0
def callback(xk, convergence=val):
    if len(results) < max_iter:
        results.append(xk.copy())      

# Optimization using differential evolution algorithm
solution = optimize.differential_evolution(objective, bounds=bounds, 
                                           disp=True, maxiter=max_iter, 
                                           callback=callback)

# Printing the results
print(solution)
print("Optimum solution:")
print(f"CH4 = {solution.fun:.4f}")
print(f"A/V = {solution.x[0]:.1f} 1/cm")
print(f"d = {solution.x[1]:.3f} cm")
print(f"Porosity = {solution.x[2]:.4f}")

# Iteratively calculate the values for ch4
obj_values_iterative = []
cat_area_per_vol_values_iterative = []
diameter_values_iterative = []
porosity_values_iterative = []
for res in results:
    cat_area_per_vol, diameter, porosity = res
    obj_values_iterative.append(objective(res)) 
    cat_area_per_vol_values_iterative.append(cat_area_per_vol)
    diameter_values_iterative.append(diameter)
    porosity_values_iterative.append(porosity)
# Plotting the graph for the iterative results
fig1, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Iterations')
ax1.set_ylabel('CH4 (mol/mol)', color=color)
ax1.plot(np.linspace(1, len(results), len(results)), obj_values_iterative, 
         label='CH4 (Iterative)', marker='o', color=color) 
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('d (cm)', color=color)
ax2.plot(np.linspace(1, len(results), len(results)), diameter_values_iterative,
         label='d (Iterative)', marker='x', color=color)  
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(diameter_min, diameter_max)

fig1.tight_layout()  
plt.title('Iterative Results')
fig1.legend(loc="lower right", bbox_to_anchor=(1,0.5), bbox_transform=ax1.transAxes)
plt.savefig('plot-Aufgabe1-v1.png', dpi=150)

fig2, ax3 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax3.set_ylabel('A/V (1/cm)', color=color)
ax3.plot(np.linspace(1, len(results), len(results)), cat_area_per_vol_values_iterative, 
         label='A/V (Iterative)', marker='o', color=color) 
ax3.tick_params(axis='y', labelcolor=color)
ax3.set_ylim(cat_area_per_vol_min, cat_area_per_vol_max)

ax4 = ax3.twinx()  
color = 'tab:red'
ax4.set_ylabel('Porosity (-)', color=color)
ax4.plot(np.linspace(1, len(results), len(results)), porosity_values_iterative,
         label='Porosity (Iterative)', marker='x', color=color)  
ax4.tick_params(axis='y', labelcolor=color)
ax4.set_ylim(porosity_min, porosity_max)

fig2.tight_layout()  
plt.title('Iterative Results')
fig2.legend(loc="lower right", bbox_to_anchor=(1,0.5), bbox_transform=ax3.transAxes)
plt.savefig('plot-Aufgabe1-v2.png', dpi=150)
