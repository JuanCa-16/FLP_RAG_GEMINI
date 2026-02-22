# src/scripts/cargar_materiales.py

import json
from sqlalchemy.orm import Session
from src.database.database import SessionLocal
from src.models.material_estudio import MaterialEstudio


JSONL_PATH = "src/embeddings/corpus_con_ejemplos_embeddings.jsonl"  # cambia si es necesario
BATCH_SIZE = 100  # tamaño de lote


def procesar_linea(linea_json):
    """
    Convierte una línea del jsonl en un objeto MaterialEstudio
    """
    data = json.loads(linea_json)

    metadata = data.get("metadata", {})

    # Intentar obtener URL o LINK
    url = metadata.get("URL") or metadata.get("LINK_VIDEO") or ""


    return MaterialEstudio(
        fuente=metadata.get("FUENTE"),
        nombre_documento=metadata.get("NOMBRE_DOCUMENTO"),
        url=url, 
        tematica=metadata.get("TEMATICA"),
        competencia=metadata.get("COMPETENCIA"),
        resultado_aprendizaje=metadata.get("RESULTADO_APRENDIZAJE"),
        nivel_dificultad=metadata.get("NIVEL_DIFICULTAD"),
        material_embedding=data.get("embeddings"),
    )


def cargar_materiales():
    db: Session = SessionLocal()
    contador = 0
    batch = []

    try:
        with open(JSONL_PATH, "r", encoding="utf-8") as file:
            for linea in file:

                material = procesar_linea(linea)
                batch.append(material)

                if len(batch) >= BATCH_SIZE:
                    db.add_all(batch)
                    db.commit()
                    contador += len(batch)
                    print(f"{contador} registros insertados...")
                    batch = []

            # Insertar lo que quede
            if batch:
                db.add_all(batch)
                db.commit()
                contador += len(batch)

        print(f"\nCarga finalizada. Total insertados: {contador}")

    except Exception as e:
        db.rollback()
        print("Error durante la carga:", e)

    finally:
        db.close()


if __name__ == "__main__":
    cargar_materiales()