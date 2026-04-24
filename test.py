import pandas as pd
import requests
import os

# Charge les données de test (non encodées)
path = r"H:\Documents\ING3 IA\UE  Entreprises\Big data\MLOPS\Projet\mlops_cours-main\purchase-predict\data\03_primary\primary.csv"
dataset = pd.read_csv(path)
dataset = dataset.drop(["user_session", "user_id", "purchased"], axis=1)

# Requête vers l'API
result = requests.post(
    "http://127.0.0.1:5000/predict",
    json=dataset.sample(n=5).to_json()
).json()
print(result)