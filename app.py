# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import weaviate
from sentence_transformers import SentenceTransformer

WEAVIATE_URL = "http://localhost:8081"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

print("Loading local embedding model:", EMBED_MODEL_NAME)
model = SentenceTransformer(EMBED_MODEL_NAME)

print("Connecting to Weaviate:", WEAVIATE_URL)
client = weaviate.Client(WEAVIATE_URL)

app = FastAPI(title="RAG-based Crop Recommendation API")


class SoilQuery(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    temperature: float
    humidity: float
    question: str = "Which crop is suitable for this soil?"


def embed_text(text: str):
    vec = model.encode([text])[0].tolist()
    return vec


def search_weaviate(vector, k: int = TOP_K):
    result = (
        client.query
        .get(
            "CropRow",
            [
                "text",
                "recommended_crop",
                "nitrogen",
                "phosphorus",
                "potassium",
                "temperature",
                "humidity",
                "ph_value",
                "chemical",
                "threshold",
                "disease",
                "affected_crops",
            ],
        )
        .with_near_vector({"vector": vector})
        .with_limit(k)
        .do()
    )
    return result.get("data", {}).get("Get", {}).get("CropRow", [])


@app.get("/")
def root():
    return {
        "message": "RAG-based Crop Recommendation API running",
        "docs": "/docs",
        "recommend_endpoint": "/recommend",
    }


@app.post("/recommend")
def recommend(query: SoilQuery):
    query_text = (
        f"Soil: N={query.nitrogen} ppm, P={query.phosphorus} ppm, "
        f"K={query.potassium} ppm, pH={query.ph}, Temp={query.temperature}C, "
        f"Humidity={query.humidity}%. Question: {query.question}"
    )

    vector = embed_text(query_text)
    rows = search_weaviate(vector, TOP_K)

    if not rows:
        return {
            "status": "error",
            "message": "No matching rows found in the vector database.",
            "query_summary": query_text,
        }

    # Build cleaned match list
    matches = []
    for r in rows:
        matches.append(
            {
                "recommended_crop": r.get("recommended_crop"),
                "nitrogen": r.get("nitrogen"),
                "phosphorus": r.get("phosphorus"),
                "potassium": r.get("potassium"),
                "temperature": r.get("temperature"),
                "humidity": r.get("humidity"),
                "ph_value": r.get("ph_value"),
                "disease": r.get("disease"),
                "affected_crops": r.get("affected_crops"),
                "chemical": r.get("chemical"),
                "threshold": r.get("threshold"),
                "source_text": r.get("text"),
            }
        )

    # Best crop = the top match
    best_crop = matches[0]["recommended_crop"]

    # Other unique crops from the remaining matches
    other_crops = []
    for m in matches[1:]:
        c = m["recommended_crop"]
        if c and c != best_crop and c not in other_crops:
            other_crops.append(c)

    return {
        "status": "success",
        "query_summary": query_text,
        "best_crop": best_crop,
        "other_candidates": other_crops,
        "matches": matches,
    }
