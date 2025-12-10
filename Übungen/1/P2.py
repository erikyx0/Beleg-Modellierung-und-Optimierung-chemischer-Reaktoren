import csv 
import numpy as np 
import matplotlib.pyplot as plt 
import cantera as ct 
import os 

os.chdir(os.path.dirname(__file__))

# unit conversion factors 
cm = 0.01 #m
minute = 60 #s

# Input parameters 
tc = 1000 
length = 0.3 * cm 
area = 1 * cm**2 
cat_area_per_vol = 1000/cm 
velocity = 40 * cm / minute 
porosity = 0.3 

# input file 
yaml_file = "methane_pox_on_pt.yaml"
output_filename = "surf_pfr2_output.csv"

t = 273.15 + tc 

surf = ct.Interface(yaml_file, "Pt_surf")
surf.TP = t, 1*ct.one_atm 
gas = surf.adjacent['gas']
gas.TPX = t, 1*ct.one_atm, "CH4:1, O2:1.5, AR:0.1"

# Reaktor erstellen 
mass_flow_rate = velocity *gas.density * area * porosity

r = ct.FlowReactor(gas)
r.area = area 
r.surface_area_to_volume_ratio = cat_area_per_vol * porosity 
r.mass_flow_rate = mass_flow_rate 
r.energy_enabled = False 

# reacting surface 
rsurf = ct.ReactorSurface(surf, r) 

# Simulation 
sim = ct.ReactorNet([r])

output_data = []
n = 0
while sim.distance < length:
    dist = sim.distance* 1e3 #in mm 
    sim.step()
    n += 1 
    output_data.append(
        [dist, r.T - 273.15, r.thermo.P/ct.one_atm] + list(r.thermo.X) + list(rsurf.kinetics.coverages)
    )


with open(output_filename, "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Distance (mm)", "T (C)", "P (atm)"] + gas.species_names + surf.species_names)
    writer.writerows(output_data)
print("Ergebnisse wurden in csv geschrieben")

data = np.array(output_data)
plt.plot(data[:,0], data[:,7], label = "CO")
plt.plot(data[:,0], data[:,3], label = r"$\mathrm{H_2}$")
plt.plot(data[:,0], data[:,6], label = r"$\mathrm{CH_4}$")
plt.xlabel("Distance (mm)")
plt.ylabel("Mass Fractiom")
plt.grid()
plt.legend(loc="best")
plt.show()