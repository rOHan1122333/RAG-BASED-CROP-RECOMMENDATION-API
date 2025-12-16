import pandas as pd
from sentence_transformers import SentenceTransformer
import weaviate

# ---- CONFIG ----
CSV_PATH = "Updated_Crop_Recommendation_with_Disease_Info.csv"
WEAVIATE_URL = "http://localhost:8081"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading local embedding model:", EMBED_MODEL_NAME)
model = SentenceTransformer(EMBED_MODEL_NAME)

print("Connecting to Weaviate:", WEAVIATE_URL)
client = weaviate.Client(WEAVIATE_URL)   # v3 client style

# ---- SCHEMA ----
schema = {
    "class": "CropRow",
    "vectorizer": "none",
    "properties": [
        {"name": "text", "dataType": ["text"]},
        {"name": "nitrogen", "dataType": ["number"]},
        {"name": "phosphorus", "dataType": ["number"]},
        {"name": "potassium", "dataType": ["number"]},
        {"name": "temperature", "dataType": ["number"]},
        {"name": "humidity", "dataType": ["number"]},
        {"name": "ph_value", "dataType": ["number"]},
        {"name": "recommended_crop", "dataType": ["text"]},
        {"name": "chemical", "dataType": ["text"]},
        {"name": "threshold", "dataType": ["text"]},
        {"name": "disease", "dataType": ["text"]},
        {"name": "affected_crops", "dataType": ["text"]},
    ]
}

# Try to create class (ignore if exists)
try:
    client.schema.create_class(schema)
    print("Schema created.")
except Exception as e:
    print("Schema may already exist:", e)

# ---- LOAD CSV ----
print("Loading CSV:", CSV_PATH)
# If your file is tab-separated, keep sep="\t"
df = pd.read_csv(CSV_PATH, sep="\t")

def row_to_text(row):
    return (
        f"Soil: N={row['Nitrogen']} ppm, P={row['Phosphorus']} ppm, "
        f"K={row['Potassium']} ppm, pH={row['pH_Value']}, Temp={row['Temperature']}C, "
        f"Humidity={row['Humidity']}%. "
        f"Recommended Crop: {row['Recommended_Crop']}. "
        f"Disease: {row['Disease']} (Affected: {row['Affected Crops']}). "
        f"Chemical Component: {row['Chemical/Component']} (Threshold: {row['Threshold']})."
    )

print("Creating embeddings...")
texts = [row_to_text(r) for _, r in df.iterrows()]
embeddings = model.encode(texts, show_progress_bar=True)

print("Configuring batch...")
client.batch.configure(batch_size=20)

print("Uploading to Weaviate...")
with client.batch as batch:
    for i, (idx, row) in enumerate(df.iterrows()):
        props = {
            "text": texts[i],
            "nitrogen": float(row["Nitrogen"]),
            "phosphorus": float(row["Phosphorus"]),
            "potassium": float(row["Potassium"]),
            "temperature": float(row["Temperature"]),
            "humidity": float(row["Humidity"]),
            "ph_value": float(row["pH_Value"]),
            "recommended_crop": row["Recommended_Crop"],
            "chemical": row["Chemical/Component"],
            "threshold": row["Threshold"],
            "disease": row["Disease"],
            "affected_crops": row["Affected Crops"],
        }

        # v3-style method name:
        batch.add_data_object(
            data_object=props,
            class_name="CropRow",
            vector=embeddings[i]
        )

print("Ingestion done.")
