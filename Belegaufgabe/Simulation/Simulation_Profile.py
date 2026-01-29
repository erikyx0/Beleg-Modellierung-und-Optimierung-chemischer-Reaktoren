# plot_temperature_profile.py
# Minimal example to run your CSTRCascadeModel and plot the temperature profile

import matplotlib.pyplot as plt

# import your class (adjust module name/path if needed)
from Kaskade_Klasse import CSTRCascadeModel, cm


def main():
    # ---- USER INPUTS ----
    yaml_file = "methane_pox_on_pt.yaml"          # <-- adjust to your Cantera YAML file
    gas_name = "gas"                 # <-- adjust if different in your YAML
    surface_name = "Pt_surf"         # <-- adjust if different in your YAML

    tc_C = 900.0                     # inlet temperature [°C]
    p_Pa = 5e5                       # pressure [Pa]
    length_m = 0.20                  # reactor length [m]
    mdot_kg_s = 0.01                 # mass flow [kg/s]
    n_cstr = 50                      # number of CSTRs (axial resolution)

    gas_comp = "CH4:1, O2:0.6, AR:0.1"

    cat_area_per_vol_per_cm = 200.0  # catalyst area per volume [1/cm] (your definition)
    diameter_cm = 2.0                # reactor diameter [cm]
    porosity = 0.40                  # void fraction [-]

    energy_enabled = True           # set True if you want temperature to evolve

    # ---- BUILD MODEL ----
    model = CSTRCascadeModel(
        yaml_file=yaml_file,
        tc_C=tc_C,
        p_Pa=p_Pa,
        length_m=length_m,
        mass_flow_rate_kg_s=mdot_kg_s,
        n_cstr=n_cstr,
        gas_comp=gas_comp,
        energy_enabled=energy_enabled,
        surface_name=surface_name,
        gas_name=gas_name,
        track_species=("CH4","O2", "H2", "CO"),      # not required for temperature plot
        track_coverages=False,
    )

    # ---- SIMULATE WITH PROFILE ----
    res = model.simulate(
        cat_area_per_vol_per_cm=cat_area_per_vol_per_cm,
        diameter_cm=diameter_cm,
        porosity=porosity,
        return_profile=True,
        T_amb_C=300,
        U_W_m2K=100
    )

    profile = res["profile"]
    T_profile = profile["T"]  # [K]

    CH4_profile = profile["CH4"]
    O2_profile = profile["O2"]
    H2_profile = profile["H2"]
    CO_profile = profile["CO"]

    # ---- BUILD AXIAL COORDINATE ----
    # Each stage corresponds to ~equal slice of the reactor length
    z = [(i + 1) * (length_m / n_cstr) for i in range(n_cstr)]  # [m]

    # ---- PLOT Temperature ----
    plt.figure()
    plt.plot(z, T_profile)
    plt.xlabel("z [m]")
    plt.ylabel("T [K]")
    plt.title("Temperature profile along CSTR cascade (PFR approximation)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- PLOT Species ----
    plt.figure()
    plt.plot(z, CH4_profile, label = "Ch4")
    plt.plot(z, O2_profile, label = "O2")
    plt.plot(z, H2_profile, label = "H2")
    plt.plot(z, CO_profile, label = "CO")
    plt.xlabel("z [m]")
    plt.ylabel("T [K]")
    plt.title("Temperature profile along CSTR cascade (PFR approximation)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- OPTIONAL: PRINT SUMMARY ----
    print(f"T_out  = {res['T_out']:.2f} K")
    print(f"T_max  = {res['T_max']:.2f} K")
    print(f"P_out  = {res['P_out']:.2f} Pa")


if __name__ == "__main__":
    main()
