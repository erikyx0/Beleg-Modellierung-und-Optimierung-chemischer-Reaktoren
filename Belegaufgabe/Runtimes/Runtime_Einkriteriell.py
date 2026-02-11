import time
import statistics as stats
import sys
from pathlib import Path
import os 

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.chdir(os.path.dirname(__file__))

from Simulation.optimize_kaskade_einkriteriell import main


def bench(n_runs=10):

    times = []

    print("\nStarting optimization benchmark...\n")

    for i in range(n_runs):

        print(f"Run {i+1}/{n_runs}")

        t0 = time.perf_counter()

        main()     # <-- führt deine komplette Optimierung aus

        dt = time.perf_counter() - t0
        times.append(dt)

        print(f"Runtime: {dt:.2f} s\n")

    print("====== RESULTS ======")
    print(f"Mean:   {stats.mean(times):.2f} s")
    print(f"Median: {stats.median(times):.2f} s")
    print(f"Min:    {min(times):.2f} s")
    print(f"Max:    {max(times):.2f} s")


if __name__ == "__main__":
    bench(10)



"""
--- Ergebnis --- 
Einkriterielle Optimierung ist ca. 70s/run 
"""