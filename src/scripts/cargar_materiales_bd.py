# src/scripts/cargar_materiales.py

import json
from sqlalchemy.orm import Session
from src.database.database import SessionLocal

import src.models

from src.models.material_estudio import MaterialEstudio
from src.models.documento import Documento

JSONL_PATH = "src/embeddings/corpus_con_ejemplos_embeddings.jsonl"
BATCH_SIZE = 100


def obtener_o_crear_documento(db: Session, metadata: dict):
    """
    Busca si el documento ya existe.
    Si existe, devuelve su ID.
    Si no existe, lo crea y devuelve su ID.
    """

    nombre_documento = metadata.get("NOMBRE_DOCUMENTO")
    fuente = metadata.get("FUENTE")

    documento_existente = db.query(Documento).filter(
        Documento.nombre_documento == nombre_documento,
        Documento.fuente == fuente
    ).first()

    if documento_existente:
        return documento_existente.id

    nuevo_documento = Documento(
        fuente=fuente,
        nombre_documento=nombre_documento,
        url=metadata.get("URL") or metadata.get("LINK_VIDEO") or "",
        tematica=metadata.get("TEMATICA"),
        competencia=metadata.get("COMPETENCIA"),
        resultado_aprendizaje=metadata.get("RESULTADO_APRENDIZAJE"),
        nivel_dificultad=metadata.get("NIVEL_DIFICULTAD"),
    )

    db.add(nuevo_documento)
    db.flush()  # Obtiene el ID sin commit

    return nuevo_documento.id


def procesar_linea(db: Session, linea_json: str):
    """
    Procesa una línea del JSONL:
    - Obtiene o crea Documento
    - Crea MaterialEstudio usando el ID del JSON
    """

    data = json.loads(linea_json)
    metadata = data.get("metadata", {})

    # 🔹 ID obligatorio desde el JSON
    material_id = data.get("id")

    if material_id is None:
        raise ValueError("El JSON no contiene campo 'id'")

    material_id = int(material_id)

    # 🔹 Verificar si el material ya existe
    material_existente = db.query(MaterialEstudio).filter(
        MaterialEstudio.id == material_id
    ).first()

    if material_existente:
        print(f"⚠️ Material {material_id} ya existe - Omitiendo...")
        return None

    # 🔹 Obtener o crear documento
    documento_id = obtener_o_crear_documento(db, metadata)

    # 🔹 Crear material con ID explícito
    material = MaterialEstudio(
        id=material_id,
        documento_id=documento_id,
        material_embedding=data.get("embeddings"),
    )

    return material


def cargar_materiales():
    db: Session = SessionLocal()
    contador = 0
    batch = []

    try:
        with open(JSONL_PATH, "r", encoding="utf-8") as file:
            for linea in file:

                material = procesar_linea(db, linea)

                if material is None:
                    continue

                batch.append(material)

                if len(batch) >= BATCH_SIZE:
                    db.add_all(batch)
                    db.commit()
                    contador += len(batch)
                    print(f"{contador} materiales insertados...")
                    batch = []

            # Insertar lo restante
            if batch:
                db.add_all(batch)
                db.commit()
                contador += len(batch)

        print(f"\nCarga finalizada. Total materiales insertados: {contador}")

    except Exception as e:
        db.rollback()
        print("Error durante la carga:", e)

    finally:
        db.close()


if __name__ == "__main__":
    cargar_materiales()