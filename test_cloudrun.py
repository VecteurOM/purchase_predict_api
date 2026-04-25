import pandas as pd
import requests

path = r"H:\Documents\ING3 IA\UE  Entreprises\Big data\MLOPS\Projet\mlops_cours-main\purchase-predict\data\03_primary\primary.csv"
dataset = pd.read_csv(path)
dataset = dataset.drop(["user_session", "user_id", "purchased"], axis=1)

URL = "https://purchase-predict-api-511275009279.europe-west1.run.app/predict"

response = requests.post(URL, json=dataset.sample(n=5).to_json())
print("Status:", response.status_code)
print("Résultat:", response.json())