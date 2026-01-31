# bench_cstr_cascade.py
# Benchmark: run many simulations and measure runtime + (optional) profiling

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from Simulation.Kaskade_Klasse import CSTRCascadeModel


import time
import statistics as stats
from contextlib import contextmanager

"""
---- Ergebnis ----+
100/1000 done; last T_out=2692.9 K
200/1000 done; last T_out=2692.9 K
300/1000 done; last T_out=2692.9 K
400/1000 done; last T_out=2692.9 K
500/1000 done; last T_out=2692.9 K
600/1000 done; last T_out=2692.9 K
700/1000 done; last T_out=2692.9 K
800/1000 done; last T_out=2692.9 K
900/1000 done; last T_out=2692.9 K
1000/1000 done; last T_out=2692.9 K

--- Benchmark results ---
Runs: 1000 (warmup: 10)
Total time: 184.326 s
Mean per run: 184.32 ms
Median per run: 181.58 ms
P90 per run: 194.19 ms
Min/Max per run: 174.25 / 466.89 ms

Process finished with exit code 0

"""

@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"{label}: {dt:.3f} s")


def build_model():
    # ---- USER INPUTS ----
    yaml_file = "methane_pox_on_pt.yaml"
    gas_name = "gas"
    surface_name = "Pt_surf"

    tc_C = 900.0
    p_Pa = 5e5
    length_m = 0.20
    mdot_kg_s = 0.01
    n_cstr = 50

    gas_comp = "CH4:1, O2:0.6, AR:0.1"
    energy_enabled = True

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
        track_species=("CH4", "O2", "H2", "CO"),
        track_coverages=False,
    )
    return model


def run_one(model):
    # keep it small for benchmarking: NO profile arrays unless needed
    return model.simulate(
        cat_area_per_vol_per_cm=200.0,
        diameter_cm=2.0,
        porosity=0.40,
        return_profile=False,   # important: faster + less memory
        T_amb_C=300,
        U_W_m2K=100
    )


def bench(n_runs=1000, warmup=10):
    model = build_model()

    # --- warmup (fills caches, triggers JIT-like init in libs, etc.) ---
    for _ in range(warmup):
        run_one(model)

    # --- timed loop ---
    times = []
    t_start = time.perf_counter()
    for i in range(n_runs):
        t0 = time.perf_counter()
        res = run_one(model)
        times.append(time.perf_counter() - t0)

        # optional: light progress
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{n_runs} done; last T_out={res['T_out']:.1f} K")

    total = time.perf_counter() - t_start

    print("\n--- Benchmark results ---")
    print(f"Runs: {n_runs} (warmup: {warmup})")
    print(f"Total time: {total:.3f} s")
    print(f"Mean per run: {stats.mean(times)*1000:.2f} ms")
    print(f"Median per run: {stats.median(times)*1000:.2f} ms")
    print(f"P90 per run: {sorted(times)[int(0.90*len(times))-1]*1000:.2f} ms")
    print(f"Min/Max per run: {min(times)*1000:.2f} / {max(times)*1000:.2f} ms")


if __name__ == "__main__":
    bench(n_runs=1000, warmup=10)
