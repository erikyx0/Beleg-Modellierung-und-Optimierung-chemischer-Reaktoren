import pandas as pd
import matplotlib.pyplot as plt

colors = [
    "#4C72B0",  # muted blue
    "#DD8452",  # soft orange
    "#55A868",  # muted green
    "#C44E52",  # muted red
    "#8172B2",  # soft purple
    "#937860",  # warm brown
    "#64B5CD",  # calm cyan
    "#CCB974",  # soft olive
]


df_pareto = pd.read_csv("pareto_CH4_vs_Vcat.csv")
df_all = pd.read_csv("all_evaluated_points.csv")

print(df_pareto.head())
pareto_ch4_out = df_pareto["CH4_out"]
all_ch4_out = df_all["CH4_out"]

pareto_Vcat_out = df_pareto["Vcat_m3"]
all_Vcat_out = df_all["Vcat_m3"]


plt.scatter(all_ch4_out, all_Vcat_out, label = "Alle Punkte", color = "blue", marker = "x")
plt.scatter(pareto_ch4_out, pareto_Vcat_out, label = "Pareto-Front", color = "red", marker="x")
plt.grid()
plt.legend()
plt.show()