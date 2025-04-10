import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.model_selection import train_test_split
import os
import re
# CSVs einlesen
df_belichtung = pd.read_csv(r"C:\Users\Dahmen_P\PycharmProjects\Masterarbeit\.venv\belichtungszeit_nach_ID.csv")
df_param = pd.read_csv(r"C:\Users\Dahmen_P\PycharmProjects\Masterarbeit\.venv\Rahmenparameter.csv")
df_ergänzteparameter = pd.read_csv(r"C:\Users\Dahmen_P\PycharmProjects\Masterarbeit\.venv\Test speicher\trainingsdaten_inkl_neuer_vorschlaege.csv")

# CSV-Dateien korrekt einlesen
df_combined_results = pd.read_csv(
    r"C:\Users\Dahmen_P\PycharmProjects\Masterarbeit\.venv\Parameter für PlasmaOS\combined_results.csv", sep=",", names=["test_point_id", "intensity", "Exposure [ms]"], header=0)

df_neue_parameter = pd.read_csv(
    r"C:\Users\Dahmen_P\PycharmProjects\Masterarbeit\.venv\Parameter für PlasmaOS\neue_Parameter_im_Richtigen_Format.csv", sep=";")

# Entferne die alten Namensspalten
df_neue_parameter= df_neue_parameter.drop(columns=["Test Point Name", "Name"], errors="ignore")

# Neue test_point_id von 462 bis 511 (für 50 Einträge)
df_neue_parameter["test_point_id"] = list(range(462, 462 + len(df_neue_parameter)))

df_neue_parameter.head()

# Entferne die Spalte aus neue_parameter vor dem Merge
df_neue_parameter = df_neue_parameter.drop(columns=["Exposure [ms]"], errors="ignore")

# Dann normal mergen
df_merged2 = pd.merge(df_combined_results, df_neue_parameter, on="test_point_id", how="outer")

# Temporär alle Spalten anzeigen
#with pd.option_context('display.max_columns', None):
    #print(df_merged2.head())


# Mergen über die gemeinsame Spalte "Parameter_ID"
df_merged = pd.merge(df_belichtung, df_param, on="Parameter_ID", how="inner")
#print(df_merged.head(100))

df_merged_renamed = df_merged.rename(columns={
    "Parameter_ID": "test_point_id",
    "Belichtungszeit_ms": "Exposure [ms]",
    "P_MW": "High Power [W]",  # oder "Low Power [W]" – je nach Bedeutung
    "t_on": "Pulse-On [ms]",
    "t_off": "Pulse-Off [ms]",
    "p": "Pressure [Pa]",
    "O2": "O2 [sccm]",
    "HMDSO": "HMDSO [sccm]"
})

# Falls df_merged2 vorher eingelesen wurde:
df_kombiniert = pd.concat([df_merged2, df_merged_renamed], ignore_index=True)

with pd.option_context('display.max_columns', None):
    print(df_kombiniert)

# ❌ Falls du 'Parameter_ID' nicht fürs Training brauchst, entfernen:
df_kombiniert = df_kombiniert.drop(columns=["Low Power [W]","Parameter_ID","intensity","test_point_id","Coat Time [s]" , "Purge Time [s]","HMDSN [sccm]" ,  "C2H2 [sccm]","Ar [sccm]"], errors='ignore')
with pd.option_context('display.max_columns',None, 'display.max_rows', None):
    print(df_kombiniert)
# ❗ Nur die für das Modell relevanten Spalten umbenennen
df_kombiniert = df_kombiniert.rename(columns={
    "High Power [W]": "P_MW",
    "Pulse-On [ms]": "t_on",
    "Pulse-Off [ms]": "t_off",
    "Pressure [Pa]": "p",
    "O2 [sccm]": "O2",
    "HMDSO [sccm]": "HMDSO",
    "Exposure [ms]": "Exposure_ms"
})



# Optional: Versuche, erweiterte Daten einzulesen (falls vorhanden)
##ateipfad_gesamt = "trainingsdaten_inkl_neuer_vorschlaege.csv"
#if os.path.exists(dateipfad_gesamt):
   # print("📂 Erweiterte Daten werden geladen...")
    #df_neu = pd.read_csv(dateipfad_gesamt)
    #df_merged = pd.concat([df_merged, df_neu], ignore_index=True).drop_duplicates()
#print(df_merged)

# ========================================
# ✍️ Benutzerdefinierte Eingabe (parallel)
# ========================================
#print("\n🆕 Möchtest du eigene Parameterkombinationen manuell eingeben?")
#manuell = input("➡️  Eingabe starten? (j/n): ")

#if manuell.lower() in ["j", "ja", "y", "yes"]:

    # Funktion zur Eingabe & Umwandlung
   # def eingabe_liste(name):
      #  werte = input(f"{name} (durch Komma getrennt, z. B. 500,600): ")
       # return [int(x.strip()) for x in werte.split(",")]

   # print("\n🔢 Gib die Werte ein – alle Listen sollten gleich lang sein:")

#    P_MW   = eingabe_liste("P_MW")
#    t_on   = eingabe_liste("t_on")
 #   t_off  = eingabe_liste("t_off")
  #  p      = eingabe_liste("p")
   # O2     = eingabe_liste("O2")
    #HMDSO  = eingabe_liste("HMDSO")

#    n_eintraege = len(P_MW)
#    if not all(len(lst) == n_eintraege for lst in [t_on, t_off, p, O2, HMDSO]):
 #       print("❌ Fehler: Alle Eingabelisten müssen gleich viele Werte haben.")
#        exit()

    # Als DataFrame zusammenbauen
#    df_manuell = pd.DataFrame({
 #       "P_MW": P_MW,
#        "t_on": t_on,
 #       "t_off": t_off,
 #       "p": p,
 #       "O2": O2,
  #      "HMDSO": HMDSO
 #   })

 #   # Belichtungszeiten erfragen
 #   bel = []
 #   for i in range(n_eintraege):
 #       print(f"\n📌 Parameterkombination {i+1}:\n{df_manuell.iloc[i].to_string()}")
  #      while True:
  #          wert = input("➡️  Gemessene Belichtungszeit in ms: ")
  #          if wert.isdigit():
 #               bel.append(int(wert))
 #               break
 #           else:
 #               print("❌ Ungültige Eingabe. Bitte eine ganze Zahl eingeben.")

 #   df_manuell["Belichtungszeit_ms"] = bel

    # Anhängen an bestehenden Datensatz
 #   gemeinsame_spalten = df_merged.columns.intersection(df_manuell.columns)
 #   df_merged = pd.concat([df_merged[gemeinsame_spalten], df_manuell[gemeinsame_spalten]], ignore_index=True)

  #  print("\n✅ Eigene Parameterkombinationen wurden hinzugefügt.")




# ========================
# 🔁 Haupt-Loop starten
# ========================
# Spaltennamen in X „cleanen“ für LightGBM-Kompatibilität


while True:

    # Ziel definieren
    ziel = "Exposure_ms"
    X = df_kombiniert.drop(columns=[ziel])
    y = df_kombiniert[ziel]
    X.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", col) for col in X.columns]
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # FLAML Training
    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task="regression",
        time_budget=50400,
        metric="mse",
        log_file_name="flaml.log",
    )

    print("\n✅ Modell trainiert.")
    print("📈 Bestes Modell:", automl.best_estimator)
    print("📊 R2 auf Testset:", automl.score(X_test, y_test))

    # 🔍 Feature Importance anzeigen, falls verfügbar



    # Neue zufällige Parameterwerte mit Schrittweiten erzeugen
    def generate_unique_samples(df_existing, n_samples):
        param_cols = ["P_MW", "t_on", "t_off", "p", "O2", "HMDSO"]
        steps = {"P_MW": 100, "t_on": 2, "t_off": 20, "p": 10, "O2": 60, "HMDSO": 4}
        min_vals = {"P_MW": 500, "t_on": 3, "t_off": 30, "p": 5, "O2": 100, "HMDSO": 2}
        max_vals = {"P_MW": 900, "t_on": 9, "t_off": 90, "p": 45, "O2": 400, "HMDSO": 20}

        existing_keys = set(df_existing[param_cols].astype(str).agg("_".join, axis=1))

        unique_rows = []
        while len(unique_rows) < n_samples:
            sample = {col: np.random.choice(np.arange(min_vals[col], max_vals[col] + 1, steps[col]))
                      for col in param_cols}
            key = "_".join(map(str, sample.values()))
            if key not in existing_keys:
                existing_keys.add(key)
                unique_rows.append(sample)

        return pd.DataFrame(unique_rows)


    X_new = generate_unique_samples(df_kombiniert, 50)


    y_pred_new = automl.predict(X_new)
    y_pred_new = np.clip(y_pred_new, a_min=0, a_max=None)
    y_pred_new = np.round(y_pred_new).astype(int)

    df_vorhersage = X_new.copy()
    df_vorhersage["Vorhergesagte_Belichtungszeit_ms"] = y_pred_new

    print(df_vorhersage)
    # 📦 Formatieren für PlasmaOS CSV-Export
    df_export = pd.DataFrame({
        "Test Point Name": [f"V_{i + 1}" for i in range(len(df_vorhersage))],
        "Name": [f"V_{i + 1}" for i in range(len(df_vorhersage))],
        "High Power [W]": df_vorhersage["P_MW"],
        "Low Power [W]": [0] * len(df_vorhersage),
        "Pulse-On [ms]": df_vorhersage["t_on"],
        "Pulse-Off [ms]": df_vorhersage["t_off"],
        "Coat Time [s]": [30] * len(df_vorhersage),
        "Purge Time [s]": [10] * len(df_vorhersage),
        "Pressure [Pa]": df_vorhersage["p"],
        "O2 [sccm]": df_vorhersage["O2"],
        "HMDSO [sccm]": df_vorhersage["HMDSO"],
        "HMDSN [sccm]": [""] * len(df_vorhersage),  # leere Zellen, nicht NaN!
        "C2H2 [sccm]": [""] * len(df_vorhersage),
        "Ar [sccm]": [""] * len(df_vorhersage),
        "Exposure [ms]": df_vorhersage["Vorhergesagte_Belichtungszeit_ms"]
    }, columns=[
        "Test Point Name", "Name", "High Power [W]", "Low Power [W]",
        "Pulse-On [ms]", "Pulse-Off [ms]", "Coat Time [s]", "Purge Time [s]",
        "Pressure [Pa]", "O2 [sccm]", "HMDSO [sccm]",
        "HMDSN [sccm]", "C2H2 [sccm]", "Ar [sccm]",
        "Exposure [ms]"
    ])

    # 💾 Speichern in neues korrektes Format (keine Indexspalte!)
    df_export.to_csv(r"C:\Users\Dahmen_P\PycharmProjects\Masterarbeit\.venv\Parameter für PlasmaOS\neue_Parameter_im_Richtigen_Format.csv", sep=";", index=False)

    print("\n✅ Neue Parameter wurden korrekt im PlasmaOS-kompatiblen Format gespeichert.")




    # Manuelle Eingabe
    echte_zeiten = []
    print("\n🔢 Bitte gemessene Belichtungszeiten eingeben:")
    for i in range(len(df_vorhersage)):
        print(f"\n📌 Parameterkombination {i+1}:\n{df_vorhersage.iloc[i][:-1].to_string()}")
        while True:
            wert = input("➡️  Gemessene Belichtungszeit in ms: ")
            if wert.isdigit():
                echte_zeiten.append(int(wert))
                break
            else:
                print("❌ Ungültige Eingabe. Bitte eine ganze Zahl eingeben.")

    df_vorhersage["Belichtungszeit_ms"] = echte_zeiten
    df_vorhersage = df_vorhersage.drop(columns=["Vorhergesagte_Belichtungszeit_ms"], errors="ignore")

    df_vorhersage.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", col) for col in df_vorhersage.columns]
    # Zum Trainings-DataFrame hinzufügen
    gemeinsame_spalten = df_kombiniert.columns.intersection(df_vorhersage.columns)
    df_kombiniert = pd.concat([df_kombiniert[gemeinsame_spalten], df_vorhersage[gemeinsame_spalten]], ignore_index=True)

    print("\n✅ Neue Parameterkombinationen wurden übernommen.")

    # Fortsetzen?
    weiter = input("\n🔁 Neue Runde starten? (j = ja, n = nein): ")
    if weiter.lower() in ["n", "no", "exit"]:
        break

# ========================
# 💾 Am Ende speichern
# ========================
speichern = input("\n💾 Möchtest du den finalen Datensatz speichern? (j/n): ")
if speichern.lower() in ["j", "ja", "y", "yes"]:
    df_merged.to_csv(dateipfad_gesamt, index=False)
    print(f"✅ Datei gespeichert unter: {dateipfad_gesamt}")
    print(df_merged)
else:
    print("⚠️ Nicht gespeichert. Änderungen sind nur im RAM verfügbar.")

