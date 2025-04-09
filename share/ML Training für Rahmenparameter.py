import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.model_selection import train_test_split

# CSVs einlesen
df_belichtung = pd.read_csv(r"C:\Users\Dahmen_P\PycharmProjects\Masterarbeit\.venv\belichtungszeit_nach_ID.csv")
df_param = pd.read_csv(r"C:\Users\Dahmen_P\PycharmProjects\Masterarbeit\.venv\Rahmenparameter.csv")

# Mergen über die gemeinsame Spalte "Parameter_ID"
df_merged = pd.merge(df_belichtung, df_param, on="Parameter_ID", how="inner")

# Ergebnis anzeigen
print(df_merged.head(100))

# ❌ Falls du 'Parameter_ID' nicht fürs Training brauchst, entfernen:
df_merged = df_merged.drop(columns=["Parameter_ID"], errors='ignore')

# 🎯 Zielspalte (hier z. B. die berechnete Belichtungszeit)
ziel = "Belichtungszeit_ms"
X = df_merged.drop(columns=[ziel])
y = df_merged[ziel]

# 🔀 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 🚀 FLAML starten
automl = AutoML()
automl.fit(
    X_train=X_train,
    y_train=y_train,
    task="regression",
    time_budget=300,
    metric="mse",
    log_file_name="flaml.log",
)

# 📈 Ergebnisse
print("Bestes Modell:", automl.best_estimator)
print("R2 auf Testset:", automl.score(X_test, y_test))

# 🔮 Vorhersage
y_pred = automl.predict(X_test)
y_pred = np.clip(y_pred, a_min=0, a_max=None)
y_pred = np.round(y_pred).astype(int)

# ========================
# 🔮 Neue Vorhersagen mit zufälligen Parametern innerhalb des bekannten Bereichs
# ========================

# Min/Max je Spalte berechnen aus Trainingsdaten
min_vals = X.min()
max_vals = X.max()

# Anzahl neuer Kombinationen
n_samples = 10

# Neue zufällige Parameterwerte erzeugen
X_new = pd.DataFrame({
    col: np.random.uniform(min_vals[col], max_vals[col], size=n_samples).round(0).astype(int)
    for col in X.columns
})

# Vorhersage durchführen
y_pred_new = automl.predict(X_new)
y_pred_new = np.clip(y_pred_new, a_min=0, a_max=None)
y_pred_new = np.round(y_pred_new).astype(int)

# Ausgabe als sortierte Tabelle
df_vorhersage = X_new.copy()
df_vorhersage["Vorhergesagte_Belichtungszeit_ms"] = y_pred_new
df_vorhersage = df_vorhersage.sort_values("Vorhergesagte_Belichtungszeit_ms")

# Anzeige
print("\n📊 Neue Parameterkombinationen & Vorhersagen:")
print(df_vorhersage.to_string(index=False))

# Nutzer gibt echte Belichtungszeiten ein
echte_zeiten = []
print("\n🔢 Bitte gemessene Belichtungszeiten für die neuen Parameter eingeben:")

for i in range(len(df_vorhersage)):
    print(f"\nParameterkombination {i+1}:\n{df_vorhersage.iloc[i][:-1].to_string()}")

    while True:
        wert = input("➡️  Gemessene Belichtungszeit in ms: ")
        if wert.isdigit():
            echte_zeiten.append(int(wert))
            break
        else:
            print("❌ Ungültige Eingabe. Bitte eine ganze Zahl eingeben (z. B. 420).")

# Neue Spalte ergänzen
df_vorhersage["Belichtungszeit_ms"] = echte_zeiten

# Speichern
df_vorhersage.to_csv("neue_parametervorschlaege.csv", index=False)
print("\n✅ Neue Parameterkombinationen mit gemessenen Belichtungszeiten wurden gespeichert.")

