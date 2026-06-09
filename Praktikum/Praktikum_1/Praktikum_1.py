import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plot = True

# Given Parameters
FF = 3000 #N
rho = 7850 # kg/m3
max_stress = 200e6 #N/m2 oder MPa

# Parameters to optimize with minimal and maximum values (min, max)
Di = (0.013,0.016)
Do = (0.017, 0.019)
L = (0.08,0.1)

bounds = [Di, Do, L]

def objective(params):
    Di, Do, L = params
    m = (Do**2-Di**2) * np.pi/4 * L * rho
    return m

def stress_constraint(params):
    Di, Do, L = params
    C_max = FF/2 * L/2
    I = (Do**4 - Di**4) * np.pi / 64
    sigma_m = C_max/I * Do/2
    return max_stress -sigma_m

# initial guess (Average of min and max)
initial_guess = np.array([(Di[0]+Di[1])/2,(Do[0]+Do[1])/2,(L[0]+L[1])/2])

# iterative solution
results = []
max_iter = 300

def callback(xk):
    if len(results) < max_iter:
        results.append(xk)

solution = minimize(objective, initial_guess, method='slsqp', bounds=bounds,constraints={'type': 'ineq', 'fun': stress_constraint},callback=callback, options={'maxiter': max_iter})

Di_opt, Do_opt, L_opt = solution.x
m_minimized  = objective(solution.x)

def stress():
    Cm = (FF/2) * (L_opt/2)
    I = (Do_opt**4 - Di_opt**4) * (math.pi / 64)
    sigma_m_opt = (Cm * Do_opt) / (2* I)
    print(f"𝜎 = {sigma_m_opt/1e6:.0f} MPa")

# Printing the results
print("Optimum solution:")
print(f"𝐷𝑖 = {Di_opt*1000:.1f} mm")
print(f"𝐷𝑜 = {Do_opt*1000:.1f} mm")
print(f"𝐿 = {L_opt*1000:.0f} mm")
print(f"𝑚 = {m_minimized*1000:.1f} g")
stress()

# Iteratively calculate the values for mass and stress
mass_values_iterative = []
sigma_values_iterative = []
for res in results:
    Di, Do, L = res
    mass_values_iterative.append(objective(res)*1000)
    Cm = (FF/2) * (L/2)
    I = (Do**4 - Di**4) * (math.pi / 64)
    sigma_m = (Cm * Do) / (2* I)
    sigma_values_iterative.append(sigma_m / 1e6)

#%% Plotting the graph for the iterative results
if plot:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Mass (g)', color=color)
    ax1.plot(np.linspace(1, len(results), len(results)), mass_values_iterative,
    label='Mass (Iterative)', marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(40, 100)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Sigma (MPa)', color=color)
    ax2.plot(np.linspace(1, len(results), len(results)), sigma_values_iterative,
    label='Sigma (Iterative)', marker='x', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(120, 210)

    fig.tight_layout()
    plt.title('Iterative Results of Mass and Sigma')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.savefig('Iterative_Lösung.png', dpi=300)