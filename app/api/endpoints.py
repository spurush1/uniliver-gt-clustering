
from fastapi import APIRouter, UploadFile, File
import pandas as pd
from app.services.preprocess import prepare_features
from app.services.traditional import run_kmeans, run_dbscan, run_agglomerative
from app.models.game_theory import GameTheoryClusterer

router = APIRouter()

@router.post("/cluster/")
async def cluster(file: UploadFile = File(...), method: str = "kmeans"):
    df = pd.read_csv(file.file)
    X = prepare_features(df)

    if method == "kmeans":
        labels = run_kmeans(X)
    elif method == "dbscan":
        labels = run_dbscan(X)
    elif method == "agglo":
        labels = run_agglomerative(X)
    elif method == "gametheory":
        model = GameTheoryClusterer(X)
        labels = model.fit()
    else:
        return {"error": "Unknown method"}

    df["cluster"] = labels
    return df.to_dict(orient="records")
