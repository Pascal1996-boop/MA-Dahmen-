import pandas as pd

df_vorschlaege = pd.read_csv(r"C:\Users\Dahmen_P\PycharmProjects\Masterarbeit\.venv\neue_parametervorschläge.csv", sep=";")

print(df_vorschlaege.head())
850,850,800,750,850,500,550,650,900,550
t_on (durch Komma getrennt, z. B. 500,600): 8,7,9,9,4,6,4,6,6,4
t_off (durch Komma getrennt, z. B. 500,600): 60,67,67,73,76,48,52,44,73,75
p (durch Komma getrennt, z. B. 500,600): 44,30,33,28,23,29,16,12,24,26
O2 (durch Komma getrennt, z. B. 500,600): 206,368,374,185,392,255,163,154,225,347
HMDSO (durch Komma getrennt, z. B. 500,600): 11,8,17,20,9,3,2,9,10,16