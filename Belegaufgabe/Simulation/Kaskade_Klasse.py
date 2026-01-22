# cstr_cascade_model.py
import math
import cantera as ct

cm = 0.01


class CSTRCascadeModel:
    """
    PFR approximation via cascade of N CSTRs using a single reactor setup and
    "state marching" (same as the Cantera template):

        1) advance_to_steady_state()
        2) set upstream reservoir state = reactor outlet state
        3) upstream.syncState()
        4) sim.reinitialize()

    This avoids rebuilding Cantera objects N times and is much faster / more stable.

    Geometry mapping (kept consistent with your original FlowReactor approach):
      - Bed volume: V_bed = A_cs * L
      - Void (gas) volume: V_gas = porosity * V_bed
      - Each CSTR volume: V_stage = V_gas / N
      - Your old effective surface-area-to-gas-volume ratio was:
            (A_surf/V_gas) = (cat_apv_SI * porosity)
        so we set per stage:
            A_surf_stage = (cat_apv_SI * porosity) * V_stage

    If your cat_area_per_vol is defined differently, adjust A_surf_stage accordingly.
    """

    def __init__(
        self,
        yaml_file: str,
        tc_C: float,
        p_Pa: float,
        length_m: float,
        mass_flow_rate_kg_s: float,
        n_cstr: int,
        gas_comp: str = "CH4:1, O2:0.6, AR:0.1",
        energy_enabled: bool = False,
        surface_name: str = "Pt_surf",
        gas_name: str = "gas",
        # optional profiling
        track_species: tuple[str, ...] = ("CH4",),
        track_coverages: bool = False,
    ):
        if n_cstr < 1:
            raise ValueError("n_cstr must be >= 1")

        self.yaml_file = yaml_file
        self.t0 = tc_C + 273.15
        self.p0 = p_Pa
        self.length = length_m
        self.mdot = mass_flow_rate_kg_s
        self.n = int(n_cstr)
        self.gas_comp = gas_comp
        self.energy_flag = "on" if energy_enabled else "off"
        self.surface_name = surface_name
        self.gas_name = gas_name
        self.track_species = tuple(track_species)
        self.track_coverages = bool(track_coverages)

    @staticmethod
    def _area_from_diameter_cm(diameter_cm: float) -> float:
        return (math.pi / 4.0) * (diameter_cm * cm) ** 2  # [m^2]

    @staticmethod
    def _cat_apv_to_SI(cat_area_per_vol_per_cm: float) -> float:
        return cat_area_per_vol_per_cm / cm  # 1/cm -> 1/m

    def simulate(
        self,
        cat_area_per_vol_per_cm: float,
        diameter_cm: float,
        porosity: float,
        return_profile: bool = False,
    ) -> dict:
        # --- geometry ---
        A_cs = self._area_from_diameter_cm(diameter_cm)
        V_bed = A_cs * self.length
        V_gas = porosity * V_bed
        V_stage = V_gas / self.n

        cat_apv_SI = self._cat_apv_to_SI(cat_area_per_vol_per_cm)
        A_surf_stage = (cat_apv_SI * porosity) * V_stage

        # --- build objects ONCE ---
        # upstream gas (reservoir)
        gas_in = ct.Solution(self.yaml_file, self.gas_name)
        gas_in.TPX = self.t0, self.p0, self.gas_comp
        upstream = ct.Reservoir(gas_in)

        # reactor gas
        gas_r = ct.Solution(self.yaml_file, self.gas_name)
        gas_r.TPX = self.t0, self.p0, self.gas_comp
        r = ct.IdealGasReactor(gas_r, energy=self.energy_flag, volume=V_stage)

        # surface attached to reactor
        surf = ct.Interface(self.yaml_file, self.surface_name, [gas_r])
        surf.TP = self.t0, self.p0
        rsurf = ct.ReactorSurface(surf, r)
        rsurf.area = A_surf_stage

        # downstream reservoir (state doesn't matter much; use a separate gas object)
        gas_out = ct.Solution(self.yaml_file, self.gas_name)
        gas_out.TPX = self.t0, self.p0, self.gas_comp
        downstream = ct.Reservoir(gas_out)

        # flow devices: fixed mdot, pressure-controlled outlet
        mfc = ct.MassFlowController(upstream, r, mdot=self.mdot)
        _pc = ct.PressureController(r, downstream, primary=mfc, K=1e-5)

        sim = ct.ReactorNet([r])

        # --- optional profiling buffers (per stage) ---
        profile = None
        if return_profile:
            profile = {
                "stage": [],
                "T": [],
                "P": [],
            }
            for sp in self.track_species:
                profile[sp] = []
            if self.track_coverages:
                profile["coverages"] = []

        # --- march through N CSTRs ---
        for i in range(self.n):
            sim.advance_to_steady_state()

            if profile is not None:
                profile["stage"].append(i + 1)
                profile["T"].append(r.T)
                profile["P"].append(r.thermo.P)
                for sp in self.track_species:
                    profile[sp].append(r.thermo[sp].X[0])
                if self.track_coverages:
                    # depending on Cantera version, either rsurf.coverages or surf.coverages is available
                    try:
                        profile["coverages"].append(list(rsurf.coverages))
                    except Exception:
                        profile["coverages"].append(list(surf.coverages))

            # inlet for next stage = outlet of this stage
            # TDY is robust for state transfer
            gas_in.TDY = r.thermo.TDY
            upstream.syncState()
            sim.reinitialize()

        ch4 = r.thermo["CH4"].X[0]

        T_max = None
        if profile is not None:
            T_max = float(max(profile["T"]))

        out = {
            "CH4": float(ch4),
            "T_out": float(r.T),
            "T_max": float(T_max) if T_max is not None else float(r.T),
            "P_out": float(r.thermo.P),
            "A_surf_stage": float(A_surf_stage),
            "V_stage": float(V_stage),
        }
        if profile is not None:
            out["profile"] = profile
        return out

    def objective_CH4(self, params) -> float:
        cat_area_per_vol, diameter_cm, porosity = params
        try:
            res = self.simulate(cat_area_per_vol, diameter_cm, porosity, return_profile=False)
            return res["CH4"]
        except Exception:
            return 1e3

    def objective_eps_constraint_Vcat(self, params, Vcat_max: float) -> float:
        cat_area_per_vol, diameter_cm, porosity = params

        # geometrisches V_cat (robust!)
        A_cs = np.pi / 4.0 * (diameter_cm * cm) ** 2
        V_bed = A_cs * self.length_m
        vcat = (1.0 - porosity) * V_bed

        try:
            res = self.simulate(cat_area_per_vol, diameter_cm, porosity,
                                return_profile=False)
            ch4 = float(res["CH4"])
        except Exception:
            return 100.0  # klar schlecht

        if vcat <= Vcat_max:
            return ch4

        rel_violation = (vcat - Vcat_max) / (Vcat_max + 1e-30)
        penalty = 50.0 * rel_violation ** 2
        return ch4 + penalty
