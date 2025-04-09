import json
import pandas as pd

# Pfad zur Datei
parameters = r"C:\Users\Dahmen_P\PycharmProjects\Masterarbeit\parameters.json"

# JSON laden
with open(parameters, "r", encoding="utf-8") as f:
    rohdaten = json.load(f)


# Anzeigeoptionen setzen
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', None)
# Liste bauen: aus jedem Key den Inhalt extrahieren
parameter_liste = []
for param_id_str, einträge in rohdaten.items():
    if isinstance(einträge, list) and len(einträge) > 0:
        eintrag = einträge[0]  # Es gibt pro ID immer genau einen Dict in der Liste
        eintrag["Parameter_ID"] = int(param_id_str)
        parameter_liste.append(eintrag)

# 🧾 In DataFrame umwandeln
df_param = pd.DataFrame(parameter_liste)

# 🧠 Anzeigeoptionen
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)  # 🔥 Hier: Zeige alle Zeilen

# 🧹 Unerwünschte Spalten löschen
df_param = df_param.drop(columns=["id", "test_point_id", "HMDSN", "C2H2", "Ar","step", "name", "t_coat", "t_purge"])

# ✅ Bereinigte Tabelle anzeigen
print("✅ Bereinigte Parameterdaten:")
print(df_param.to_string(index=False))

df_param.to_csv("Rahmenparameter.csv", index=False)
print(df_param.to_string())
print("✅ CSV gespeichert: Rahmenparameter.csv")