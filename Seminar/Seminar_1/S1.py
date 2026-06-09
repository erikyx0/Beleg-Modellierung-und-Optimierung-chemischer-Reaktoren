"""
In der Einführungsaufgabe geht es darum, eine stationäre 1D-Wärmeleitung zu bestimmen.

------------------------------
Aufgabenstellung:
Eine Verbundwand besteht aus drei Materialien mit
eindimensionalem, gleichmäßigem Wärmeübergang, wie in
der Abbildung dargestellt. Schreiben Sie ein Python Programm, das folgende Aufgaben erfüllt:
1. Berechnung Gesamtwärmewiderstand (RTot).
2. Berechnung Wärmestromdichte
3. Berechnung innere Oberflächentemperatur .
4. Temperaturen T1 und T2
5. Graphische Darstellung der Temperaturverteilung (nur
Wärmeleitungsteil)
6. Speichern Sie die Ergebnisse in einer Textdatei.
7. Variieren Sie die Werte der Wärmeleitfähigkeit und
beobachten Sie die Auswirkungen auf die
Temperaturverteilung
"""


import matplotlib.pyplot as plt

class Wand:
    def __init__(self, lambda_i, dicke, T_inf, T_O, alpha_i):
        self.r_tot = None
        self.arr_aufbau = [(lambda_i,dicke)]
        self.T_inf = T_inf
        self.T_O = T_O
        self.alpha_i = alpha_i
    def add_layer(self,lambda_i, dicke):
        self.arr_aufbau.append((lambda_i,dicke))
    def calc_dicke(self):
        dicke = 0
        for layer in self.arr_aufbau:
            dicke += layer[1]
        return dicke

    def calc_R_tot(self):
        R_tot = 0.0

        for L, lam in self.arr_aufbau:
            R_tot += lam/L

        R_tot += 1 / self.alpha_i

        self.r_tot = R_tot
        return self.r_tot

    def calc_q(self):
        self.q = (self.T_inf - self.T_O) / self.r_tot

    def calc_temperatures(self):
        temps = []
        T = self.T_inf
        q = self.q

        # Konvektion
        R_alpha = 1 / self.alpha_i
        T -= q * R_alpha
        temps.append(T)

        # Schichten
        for L, lam in self.arr_aufbau:
            R = lam/L
            T -= q * R
            temps.append(T)

        return temps

    def add_multiple_layers(self,arr_aufbau):
        """
        fügt mehrere Schichten zur Wand gleichzeitig hinzu. Format der input liste ist [(lambda_i, dicke), ...]
        :param arr_aufbau:
        :return:
        """
        for i in arr_aufbau:
            self.add_layer(i[0],i[1])

    def plot_temperatures(self, savename=None):
        arr_aufbau = self.arr_aufbau
        arr_positions = []
        pos = 0


        arr_positions.append(0)
        for lam, L in arr_aufbau:
            pos += L
            arr_positions.append(pos)

        #------------------------------ PLOT ------------------------------
        plt.plot(arr_positions,self.calc_temperatures())
        for i in arr_positions:
            plt.axvline(i,color='k', linestyle=":")
        plt.axvline(0, color='k', linestyle=":")
        plt.grid()
        plt.xlabel("Distanz in m")
        plt.ylabel("Temperatur in °C")
        plt.title("Temperaturverteilung in einer geschichteten Wand")
        if savename==None:
            plt.show()
        else:
            plt.savefig(f"{savename}.png")
        plt.close()

    def save_results(self, filename="ergebnisse.txt"):
        # sicherstellen, dass alles berechnet ist
        if self.r_tot is None:
            self.calc_R_tot()
        if not hasattr(self, "q"):
            self.calc_q()

        temps = self.calc_temperatures()

        with open(filename, "w", encoding="utf-8") as f:
            f.write("=== Ergebnisse Wärmeleitung ===\n\n")

            f.write("Allgemeine Parameter:\n")
            f.write(f"T_inf: {self.T_inf} °C\n")
            f.write(f"T_O: {self.T_O} °C\n")
            f.write(f"alpha_i: {self.alpha_i} W/(m²K)\n\n")

            f.write("Schichtaufbau (lambda, dicke):\n")
            for i, (lam, L) in enumerate(self.arr_aufbau, start=1):
                f.write(f"Schicht {i}: lambda={lam} W/(mK), dicke={L} m\n")
            f.write("\n")

            f.write("Ergebnisse:\n")
            f.write(f"R_tot: {self.r_tot} (m²K)/W\n")
            f.write(f"Wärmestromdichte q: {self.q} W/m²\n\n")

            f.write("Temperaturen:\n")
            f.write(f"Innenoberfläche T_S,i: {temps[0]} °C\n")

            for i, T in enumerate(temps[1:], start=1):
                f.write(f"T{i}: {T} °C\n")

    def change_layer(self,index_number, layer_data):
        self.arr_aufbau[index_number] = layer_data

# Lösen der Aufgabe
#%% 0. Initialangaben zur Lösung
wand = Wand(20,0.06, T_inf=225, T_O=20, alpha_i = 25)
wand.add_layer(0.04,0.03)
wand.add_layer(0.6,0.04)
#print(wand.r_tot)

#%% 1. Berechnung Gesamtwärmewiederstand
wand.calc_R_tot()
print("=== 1. Berechnung Gesamtwärmewiederstand ===")
print(f"R_tot: {wand.r_tot}")

#%% 2. Berechnung Wärmestromdichte
print("=== 2. Berechnung Wärmestromdichte === ")
wand.calc_q()
print(f"Wärmestromdichte: {wand.q}")

#%% 3. Berechnung innere Oberflächentemperatur (T_S,i) & 4. Berechnung Temperaturen T1 und T2
print("=== 3. Berechnung innere Oberflächentemperatur (T_S,i) & 4. Berechnung Temperaturen T1 und T2 ===")
wand.calc_temperatures()
print(f"{wand.calc_temperatures()}")

#%% 5. Graphische Darstellung Temperaturverteilung
print("=== 5. Graphische Darstellung Temperaturverteilung ===")
wand.plot_temperatures(savename="Temperaturverteilung")
print("siehe Plot")

#%% 6. Speichern in einer Textdatei
print("=== 6. Speichern in einer Textdatei ===")
wand.save_results("Ergebnisse.txt")

#%% 7. Variieren der Wärmeleitfähigkeit
wand.change_layer(1,[0.35,0.03])
wand.plot_temperatures()