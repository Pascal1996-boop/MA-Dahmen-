import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# CSVs einlesen
df_belichtung = pd.read_csv(r"C:\Users\Dahmen_P\PycharmProjects\Masterarbeit\.venv\belichtungszeit_nach_ID.csv")
df_param = pd.read_csv(r"C:\Users\Dahmen_P\PycharmProjects\Masterarbeit\.venv\Rahmenparameter.csv")
df_ergänzteparameter = pd.read_csv(
    r"C:\Users\Dahmen_P\PycharmProjects\Masterarbeit\.venv\Test speicher\trainingsdaten_inkl_neuer_vorschlaege.csv")
# Mergen über die gemeinsame Spalte "Parameter_ID"
df_merged = pd.merge(df_belichtung, df_param, on="Parameter_ID", how="inner")

# Ergebnis anzeigen
print(df_merged.head(100))

# ❌ Falls du 'Parameter_ID' nicht fürs Training brauchst, entfernen:
df_merged = df_merged.drop(columns=["Parameter_ID"], errors='ignore')

# Optional: Versuche, erweiterte Daten einzulesen (falls vorhanden)
dateipfad_gesamt = "trainingsdaten_inkl_neuer_vorschlaege.csv"
if os.path.exists(dateipfad_gesamt):
    print("📂 Erweiterte Daten werden geladen...")
    df_neu = pd.read_csv(dateipfad_gesamt)
    df_merged = pd.concat([df_merged, df_neu], ignore_index=True).drop_duplicates()
print(df_merged)

# ========================================
# ✍️ Benutzerdefinierte Eingabe (parallel)
# ========================================
print("\n🆕 Möchtest du eigene Parameterkombinationen manuell eingeben?")
manuell = input("➡️  Eingabe starten? (j/n): ")

if manuell.lower() in ["j", "ja", "y", "yes"]:

    # Funktion zur Eingabe & Umwandlung
    def eingabe_liste(name):
        werte = input(f"{name} (durch Komma getrennt, z. B. 500,600): ")
        return [int(x.strip()) for x in werte.split(",")]


    print("\n🔢 Gib die Werte ein – alle Listen sollten gleich lang sein:")

    P_MW = eingabe_liste("P_MW")
    t_on = eingabe_liste("t_on")
    t_off = eingabe_liste("t_off")
    p = eingabe_liste("p")
    O2 = eingabe_liste("O2")
    HMDSO = eingabe_liste("HMDSO")

    n_eintraege = len(P_MW)
    if not all(len(lst) == n_eintraege for lst in [t_on, t_off, p, O2, HMDSO]):
        print("❌ Fehler: Alle Eingabelisten müssen gleich viele Werte haben.")
        exit()

    # Als DataFrame zusammenbauen
    df_manuell = pd.DataFrame({
        "P_MW": P_MW,
        "t_on": t_on,
        "t_off": t_off,
        "p": p,
        "O2": O2,
        "HMDSO": HMDSO
    })

    # Belichtungszeiten erfragen
    bel = []
    for i in range(n_eintraege):
        print(f"\n📌 Parameterkombination {i + 1}:\n{df_manuell.iloc[i].to_string()}")
        while True:
            wert = input("➡️  Gemessene Belichtungszeit in ms: ")
            if wert.isdigit():
                bel.append(int(wert))
                break
            else:
                print("❌ Ungültige Eingabe. Bitte eine ganze Zahl eingeben.")

    df_manuell["Belichtungszeit_ms"] = bel

    # Anhängen an bestehenden Datensatz
    gemeinsame_spalten = df_merged.columns.intersection(df_manuell.columns)
    df_merged = pd.concat([df_merged[gemeinsame_spalten], df_manuell[gemeinsame_spalten]], ignore_index=True)

    print("\n✅ Eigene Parameterkombinationen wurden hinzugefügt.")

# ========================
# 🔁 Haupt-Loop starten
# ========================

while True:
    # Ziel definieren
    ziel = "Belichtungszeit_ms"
    X = df_merged.drop(columns=[ziel])
    y = df_merged[ziel]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # AutoGluon Training vorbereiten - Trainingsdaten zusammenführen
    train_data = X_train.copy()
    train_data[ziel] = y_train

    # Test-Daten zusammenführen für Evaluation
    test_data = X_test.copy()
    test_data[ziel] = y_test

    # Modellpfad definieren
    model_path = 'autogluon_models'

    # AutoGluon Training
    print("\n⏳ AutoGluon Modell wird trainiert (300 Sekunden)...")
    ag_predictor = TabularPredictor(
        label=ziel,
        path=model_path,
        eval_metric='mean_squared_error'
    ).fit(
        train_data=train_data,
        time_limit=300,  # 5 Minuten Trainingszeit
        presets='medium_quality'  # Qualitätseinstellung: 'best_quality', 'high_quality', 'medium_quality'
    )

    print("\n✅ Modell trainiert.")

    # Leaderboard anzeigen - welche Modelle wurden trainiert und wie gut sind sie?
    print("\n📊 Modell-Leaderboard:")
    leaderboard = ag_predictor.leaderboard(test_data)
    print(leaderboard)

    # Modellleistung auf Testdaten bewerten
    test_score = ag_predictor.evaluate(test_data)
    print(f"\n📈 R² auf Testset: {1 - test_score['mean_squared_error'] / np.var(y_test):.4f}")
    print(f"📉 MSE auf Testset: {test_score['mean_squared_error']:.4f}")

    # Feature Importance anzeigen
    feature_importance = ag_predictor.feature_importance(train_data)
    print("\n🔍 Feature Importance:")
    print(feature_importance)

    # Feature Importance visualisieren
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='barh', x='feature', y='importance', legend=False)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("📊 Feature Importance Grafik gespeichert als 'feature_importance.png'")

    # Vorhersagen vs. tatsächliche Werte visualisieren
    y_pred = ag_predictor.predict(test_data)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title('Vorhersage vs. Tatsächliche Werte')
    plt.xlabel('Tatsächliche Belichtungszeit (ms)')
    plt.ylabel('Vorhergesagte Belichtungszeit (ms)')
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png')
    print("📊 Vorhersagegrafik gespeichert als 'predictions_vs_actual.png'")


    # Neue zufällige Parameterwerte mit Schrittweiten erzeugen
    def generate_random_values(min_val, max_val, step, size):
        possible_vals = np.arange(min_val, max_val + 1, step)
        return np.random.choice(possible_vals, size=size)


    # Anzahl neuer Kombinationen
    n_samples = 50
    min_vals = {
        "P_MW": 500,
        "t_on": 3,
        "t_off": 30,
        "p": 5,
        "O2": 100,
        "HMDSO": 2
    }
    max_vals = {
        "P_MW": 900,
        "t_on": 9,
        "t_off": 90,
        "p": 45,
        "O2": 400,
        "HMDSO": 20
    }

    X_new = pd.DataFrame({
        "P_MW": generate_random_values(min_vals["P_MW"], max_vals["P_MW"], 100, n_samples),
        "t_on": generate_random_values(min_vals["t_on"], max_vals["t_on"], 2, n_samples),
        "t_off": generate_random_values(min_vals["t_off"], max_vals["t_off"], 20, n_samples),
        "p": generate_random_values(min_vals["p"], max_vals["p"], 10, n_samples),
        "O2": generate_random_values(min_vals["O2"], max_vals["O2"], 60, n_samples),
        "HMDSO": generate_random_values(min_vals["HMDSO"], max_vals["HMDSO"], 4, n_samples),
    })

    # AutoGluon Vorhersagen für neue Parameterkombinationen
    y_pred_new = ag_predictor.predict(X_new)
    y_pred_new = np.clip(y_pred_new, a_min=0, a_max=None)  # Keine negativen Belichtungszeiten
    y_pred_new = np.round(y_pred_new).astype(int)  # Auf ganze Millisekunden runden

    df_vorhersage = X_new.copy()
    df_vorhersage["Vorhergesagte_Belichtungszeit_ms"] = y_pred_new

    print("\n🔮 Neue Parameterkombinationen mit Vorhersagen:")
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
    export_path = r"C:\Users\Dahmen_P\PycharmProjects\Masterarbeit\.venv\Parameter für PlasmaOS\neue_Parameter_im_Richtigen_Format.csv"
    df_export.to_csv(export_path, sep=";", index=False)

    print(f"\n✅ Neue Parameter wurden korrekt im PlasmaOS-kompatiblen Format gespeichert unter:\n{export_path}")

    # Manuelle Eingabe der tatsächlichen Messwerte
    echte_zeiten = []
    print("\n🔢 Bitte gemessene Belichtungszeiten eingeben:")
    for i in range(len(df_vorhersage)):
        print(f"\n📌 Parameterkombination {i + 1}:\n{df_vorhersage.iloc[i][:-1].to_string()}")
        print(f"Vorhersage: {df_vorhersage.iloc[i]['Vorhergesagte_Belichtungszeit_ms']} ms")
        while True:
            wert = input("➡️  Gemessene Belichtungszeit in ms: ")
            if wert.isdigit():
                echte_zeiten.append(int(wert))
                break
            else:
                print("❌ Ungültige Eingabe. Bitte eine ganze Zahl eingeben.")

    df_vorhersage["Belichtungszeit_ms"] = echte_zeiten
    df_vorhersage = df_vorhersage.drop(columns=["Vorhergesagte_Belichtungszeit_ms"], errors="ignore")

    # Zum Trainings-DataFrame hinzufügen
    gemeinsame_spalten = df_merged.columns.intersection(df_vorhersage.columns)
    df_merged = pd.concat([df_merged[gemeinsame_spalten], df_vorhersage[gemeinsame_spalten]], ignore_index=True)

    print("\n✅ Neue Parameterkombinationen wurden übernommen.")

    # Vorhersagefehler für diese Runde analysieren
    fehler = np.abs(np.array(echte_zeiten) - y_pred_new)
    print("\n📊 Analyse der Vorhersagefehler dieser Runde:")
    print(f"Durchschnittlicher absoluter Fehler: {fehler.mean():.2f} ms")
    print(f"Maximaler Fehler: {fehler.max():.2f} ms")
    print(f"Minimaler Fehler: {fehler.min():.2f} ms")

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