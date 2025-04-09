import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

#Speichert die ID und erstellt Graphen



# Pfad zur Datei
dateipfad = r"C:\Users\Dahmen_P\PycharmProjects\Masterarbeit\exposure_dict.json"

# JSON-Datei öffnen und Inhalt laden
with open(dateipfad, "r", encoding="utf-8") as f:
    daten = json.load(f)

# Jetzt kannst du mit 'daten' arbeiten, z. B. anzeigen:
print(daten)

# Anzeigeoptionen setzen
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', None)

# Umwandeln in DataFrame (jede ID = Zeile)
df = pd.DataFrame.from_dict(daten, orient='index')
df.columns = [f'Messung_{i+1}' for i in range(df.shape[1])]
df.reset_index(inplace=True)
df.rename(columns={'index': 'Parameter_ID'}, inplace=True)
print(df)

# Belichtungszeiten (ms)
belichtungszeiten = [33, 66, 99, 198, 396, 792, 1584, 3168, 4782, 6336, 12672, 25344]

# Einträge prüfen
for k, v in daten.items():
    if len(v) != len(belichtungszeiten):
        print(f"{k} → {len(v)} Werte statt {len(belichtungszeiten)}")

# Fehlerhafte Positionen manuell markieren
fehlerhafte_positionen = {
    "325": [0],
    "342": [6],
}

# Zielintensität definieren
ziel_I = 45000

# Exponentieller Fit

def exp_func(t, a, b):
    return a * (1 - np.exp(-b * t))

# Funktion: Beste Belichtungszeit + Fit finden

def berechne_t_best(t, I, ziel_I=45000):
    modelle = {}

    # 1. Linearer Fit
    try:
        coeffs_lin = np.polyfit(t, I, deg=1)
        lin_func = np.poly1d(coeffs_lin)
        I_pred_lin = lin_func(t)
        mse_lin = mean_squared_error(I, I_pred_lin)
        modelle["linear"] = (lin_func, mse_lin)
    except:
        pass

    # 2. Exponentieller Fit
    try:
        popt, _ = curve_fit(exp_func, t, I, p0=[max(I), 0.001], maxfev=10000)
        exp_fitted = lambda x: exp_func(x, *popt)
        I_pred_exp = exp_fitted(t)
        mse_exp = mean_squared_error(I, I_pred_exp)
        modelle["exp"] = (exp_fitted, mse_exp)

    except:
            return None, None, None

    bester_name = min(modelle, key=lambda k: modelle[k][1])
    bester_func = modelle[bester_name][0]

    t_fine = np.linspace(min(t), max(t), 1000)
    I_fine = bester_func(t_fine)
    idx_best = np.argmin(np.abs(I_fine - ziel_I))
    t_best = t_fine[idx_best]

    return t_best, bester_func, bester_name

# Plot-Konfiguration
kurven_pro_plot = 2
anzahl_geplottet = 0
plot_nummer = 1
plt.figure(figsize=(10, 6))

# Trainingsdatenliste initialisieren
id_and_time = []

# Durch alle Parameter
for param_id, intensitäten in daten.items():
    if not intensitäten:
        print(f"⚠️  Kein Eintrag bei ID {param_id}, wird übersprungen.")
        continue

    kürzeste_länge = min(len(intensitäten), len(belichtungszeiten))
    t = np.array(belichtungszeiten[:kürzeste_länge])
    I = np.array(intensitäten[:kürzeste_länge])

    maske = (~np.isnan(I)) & (I != 65535.0)
    if param_id in fehlerhafte_positionen:
        for idx in fehlerhafte_positionen[param_id]:
            if idx < len(maske):
                maske[idx] = False

    t_clean = t[maske]
    I_clean = I[maske]

    if len(t_clean) == 0:
        continue

    if t_clean[0] > 0 and I_clean[0] > ziel_I:
        t_clean = np.insert(t_clean, 0, 0)
        I_clean = np.insert(I_clean, 0, 0)

    if len(I_clean) == 0:
        print(f"⚠️  Nur ungültige Daten bei ID {param_id}, wird übersprungen.")
        continue

    print(f"\n✅ ID {param_id}")
    print("Belichtungszeiten:", t_clean)
    print("Intensitäten:     ", I_clean)

    t_best, fit_func, fit_typ = berechne_t_best(t_clean, I_clean, ziel_I)

    if t_best is not None:
        print(f"🧠 Geschätzte optimale Belichtungszeit: {t_best:.0f} ms ({fit_typ})")
        id_and_time.append([param_id, round(t_best, 2)])


        if fit_func is not None:
            t_fine = np.linspace(min(t_clean), max(t_clean), 300)
            I_fine = fit_func(t_fine)
            plt.plot(t_fine, I_fine, '--', label=f'{fit_typ} Fit ID {param_id}')
    else:
        print("⚠️  Keine sinnvolle Belichtungszeit schätzbar.")


    plt.plot(t_clean, I_clean, label=f'ID {param_id}', marker='o')
    anzahl_geplottet += 1

    if anzahl_geplottet % kurven_pro_plot == 0:
        plt.xlabel("Belichtungszeit [ms]")
        plt.ylabel("Intensität [Counts]")
        plt.title(f"Intensität vs. Belichtungszeit (Plot {plot_nummer})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)
        plt.figure(figsize=(10, 6))
        plot_nummer += 1

# 🧾 Als DataFrame speichern
df_id_time = pd.DataFrame(id_and_time, columns=["Parameter_ID", "Belichtungszeit_ms"])

# 💾 Exportieren als CSV
df_id_time.to_csv("belichtungszeit_nach_ID.csv", index=False)
print(df_id_time.to_string())
print("✅ CSV gespeichert: belichtungszeit_nach_ID.csv")


if anzahl_geplottet % kurven_pro_plot != 0:
        plt.xlabel("Belichtungszeit [ms]")
        plt.ylabel("Intensität [Counts]")
        plt.title(f"Intensität vs. Belichtungszeit (Plot {plot_nummer})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
plt.show()

