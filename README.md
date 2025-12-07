# RAG-based Crop Recommendation System 🌾

This project is a **Retrieval-Augmented Generation (RAG)** style assistant for crop recommendation.  
It combines:

- A **tabular agronomy dataset** (soil metrics, diseases, chemicals, thresholds)
- **Local text embeddings** using `sentence-transformers`
- A **vector database** (Weaviate) for semantic search
- A **FastAPI backend** exposing a `/recommend` endpoint

Given soil metrics (N, P, K, pH, temperature, humidity) and a question, the system retrieves the most similar rows from the dataset and suggests suitable crops.

---

## 🔧 Tech Stack

- **Language:** Python 3.x  
- **Framework:** FastAPI  
- **Vector DB:** Weaviate (Docker)  
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)  
- **Server:** Uvicorn  
- **Environment:** Virtualenv (`.venv`)

---

## 📂 Project Structure

```text
RAG_based/
├── app.py                          # FastAPI app (recommendation API)
├── ingest.py                       # Script to ingest CSV into Weaviate
├── docker-compose.yml              # Weaviate service (Docker)
├── Updated_Crop_Recommendation_with_Disease_Info.csv
├── requirements.txt                # Python dependencies
└── .venv/                          # Virtual environment (ignored in Git)
