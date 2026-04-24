import os
from dotenv import load_dotenv

load_dotenv()

for env_var in ["ENV", "MLFLOW_SERVER", "MLFLOW_REGISTRY_NAME"]:
    if not os.getenv(env_var):
        raise Exception(f"Variable d'environnement {env_var} manquante.")