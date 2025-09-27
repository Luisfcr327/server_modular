# ml/recommender.py
from __future__ import annotations
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, request, jsonify
from typing import Dict

app = Flask(__name__)

CSV_PATH = r"C:\Users\fcrlu\Downloads/Clasificacion_plantas.csv"

# Mapeo nombre mostrado -> slug usado en assets/config
PLANT_SLUGS = {
    "Cactus": "cactus",
    "Aloe Vera": "aloe_vera",
    "Aloe": "aloe_vera",
    "Calathea": "calathea",
    "Coleus": "coleus",
    "Espatifilo": "espatifilo",
    "Espatifilo (Peace Lily)": "espatifilo",
    "Gardenia": "gardenia",
    "Orquídea": "orquidea",
    "Orquidias": "orquidea",
    "Palma": "palma",
    "Potos": "potos",
    "Rosa": "rosa",
    "Sansevieria": "sansevieria",
    "Bonsai": "bonsai",
    "Bonsái": "bonsai",
}

# Columnas esperadas (sin 'Planta')
FEATURE_COLUMNS = [
    "Tiempo de cuidado",
    "Espacio",
    "Experiencia previa",
    "Tamaño preferido",
    "Luz disponible (valor)",
    "Mascotas",
    "Presupuesto de cuidados",
    "Objetivo principal",
    "Clima de la zona",
    "Disponibilidad de riego",
]

def _load_and_normalize(path: str = CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}")

    df = pd.read_csv(path, encoding="latin1")

    # Mapear valores a numéricos (idéntico a formulario.py)
    df["Tiempo de cuidado"] = df["Tiempo de cuidado"].map({
        "Poco": 0.0, "Medio": 0.5, "Mucho": 1.0
    })
    df["Espacio"] = df["Espacio"].map({
        "Ambos": 0.0, "Interior": 0.5, "Patio": 1.0
    })
    df["Experiencia previa"] = df["Experiencia previa"].map({
        "Principiante": 0.0, "Intermedio": 0.33,
        "Avanzado": 0.66, "Experto": 1.0
    })
    df["Tamaño preferido"] = df["Tamaño preferido"].map({
        "Pequeña": 0.0, "Mediana": 0.5, "Grande": 1.0
    })
    df["Luz disponible (valor)"] = df["Luz disponible (valor)"].map({
        "Directa": 0.0, "Indirecta": 1.0
    })
    df["Mascotas"] = df["Mascotas"].map({"No": 0, "Si": 1})
    df["Presupuesto de cuidados"] = df["Presupuesto de cuidados"].map({
        "Bajo": 0.0, "Medio": 0.5, "Alto": 1.0
    })
    df["Objetivo principal"] = df["Objetivo principal"].map({
        "Decoración": 0.0, "Aroma": 0.33,
        "Purificación del aire": 0.66, "Fácil cuidado": 1.0
    })
    df["Clima de la zona"] = df["Clima de la zona"].map({
        "Cálido/seco": 0.0, "Templado": 0.5, "Húmedo": 1.0
    })
    df["Disponibilidad de riego"] = df["Disponibilidad de riego"].map({
        "Poco": 0.0, "Medio": 0.5, "Mucho": 1.0
    })

    return df

def recommend(answers: dict[str, str]) -> tuple[str, str]:
    df = _load_and_normalize(CSV_PATH)

    # Entrenamiento
    X = df.drop("Planta", axis=1)
    y = df["Planta"]

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    # Normalizar respuestas igual que el dataset
    resp_df = pd.DataFrame([answers])

    mappings = {
        "Tiempo de cuidado": {"Poco": 0.0, "Medio": 0.5, "Mucho": 1.0},
        "Espacio": {"Ambos": 0.0, "Interior": 0.5, "Patio": 1.0},
        "Experiencia previa": {
            "Principiante": 0.0, "Intermedio": 0.33,
            "Avanzado": 0.66, "Experto": 1.0
        },
        "Tamaño preferido": {"Pequeña": 0.0, "Mediana": 0.5, "Grande": 1.0},
        "Luz disponible (valor)": {"Directa": 0.0, "Indirecta": 1.0},
        "Mascotas": {"No": 0, "Si": 1, "Sí": 1},  # agregado "Sí"
        "Presupuesto de cuidados": {"Bajo": 0.0, "Medio": 0.5, "Alto": 1.0},
        "Objetivo principal": {
            "Decoración": 0.0, "Aroma": 0.33,
            "Purificación": 0.66, "Purificación del aire": 0.66,
            "Fácil cuidado": 1.0
        },
        "Clima de la zona": {"Cálido/seco": 0.0, "Templado": 0.5, "Húmedo": 1.0},
        "Disponibilidad de riego": {"Poco": 0.0, "Medio": 0.5, "Mucho": 1.0},
    }

    for col, mapping in mappings.items():
        if col in resp_df:
            resp_df[col] = resp_df[col].map(mapping).fillna(0.0)

    # Predicción (aseguramos columnas en el mismo orden)
    resp_df = resp_df.reindex(columns=FEATURE_COLUMNS).fillna(0.0)

    pred = knn.predict(resp_df)[0]

    slug = PLANT_SLUGS.get(
        pred,
        pred.lower().replace(" ", "_")
        .replace("á","a").replace("é","e")
        .replace("í","i").replace("ó","o")
        .replace("ú","u").replace("ñ","n")
    )

    return pred, slug

@app.route("/recommend", methods=["POST"])
def recommend_route():
    data: Dict[str, str] = request.json
    try:
        plant_name, slug = recommend(data)
        return jsonify({"plant_name": plant_name, "slug": slug})
    except Exception as e:
        import traceback
        traceback.print_exc()  # Muestra el error completo en la consola
        return jsonify({"error": str(e)}), 500
    
@app.route("/health", methods=["GET"])
def health() -> Dict:
    return {"status": "ok"}
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=12345, debug=True)





