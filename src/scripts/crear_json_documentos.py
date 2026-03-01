from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

import json
import os
from pathlib import Path
from src.models.documento import Documento
from dotenv import load_dotenv


load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def cargar_mapping_tematicas():
    """
    Carga el archivo tematicas.json para obtener el mapeo de números a temáticas.
    """
    ruta_json = Path(__file__).parent.parent.parent  / 'tematicas.json'
    
    with open(ruta_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Crear un diccionario invertido: {nombre_tematica: numero}
    mapping = {}
    for numero, info in data.items():
        tematica = info['TEMATICA']
        mapping[tematica] = numero
    
    return mapping


def generar_json():
    """
    Genera el JSON agrupando documentos por temática.
    """
    
    # Cargar el mapping de temáticas
    mapping_tematicas = cargar_mapping_tematicas()
    
    # Crear engine y sesión
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Obtener todas las temáticas únicas
        tematicas = session.query(
            Documento.tematica
        ).distinct().filter(
            Documento.tematica.isnot(None)
        ).all()
        
        resultado = {}
        
        for (tematica,) in tematicas:
            # Buscar el número correcto en el mapping
            numero = mapping_tematicas.get(tematica)
            
            if numero is None:
                print(f"⚠️  Advertencia: Temática '{tematica}' no encontrada en tematicas.json")
                continue
            
            # Obtener el primer documento de esta temática para info general
            doc_base = session.query(Documento).filter(
                Documento.tematica == tematica
            ).first()
            
            # Obtener videos
            videos = session.query(Documento).filter(
                Documento.tematica == tematica,
                Documento.fuente == 'VIDEO'
            ).all()
            
            # Obtener PDFs
            pdfs = session.query(Documento).filter(
                Documento.tematica == tematica,
                Documento.fuente == 'PDF'
            ).all()
            
            # Obtener GIT
            gits = session.query(Documento).filter(
                Documento.tematica == tematica,
                Documento.fuente == 'GIT'
            ).all()
            
            # Construir objeto de temática usando el número correcto
            resultado[numero] = {
                'tematica': doc_base.tematica,
                'videos': [
                    {
                        'id': str(v.id),
                        'nombre': v.nombre_documento or '',
                        'url': v.url or ''
                    }
                    for v in videos
                ],
                'pdfs': [
                    {
                        'id': str(p.id),
                        'nombre': p.nombre_documento or '',
                        'url': p.url or ''
                    }
                    for p in pdfs
                ],
                'git': [
                    {
                        'id': str(g.id),
                        'nombre': g.nombre_documento or '',
                        'url': g.url or ''
                    }
                    for g in gits
                ]
            }
        
        # Ordenar el resultado por las llaves numéricas (0, 1, 2, ...)
        resultado_ordenado = dict(sorted(resultado.items(), key=lambda x: int(x[0])))
        
        return resultado_ordenado
        
    finally:
        session.close()


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    print("Generando JSON desde la base de datos...")
    
    try:
        datos = generar_json()
        
        # Imprimir en consola
        print(json.dumps(datos, indent=2, ensure_ascii=False))
        
        # Guardar en archivo
        with open('json_documentos.json', 'w', encoding='utf-8') as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ JSON generado exitosamente")
        print(f"✓ Archivo guardado: json_documentos.json")
        print(f"✓ Total de temáticas: {len(datos)}")
        
    except Exception as e:
        print(f"✗ Error: {e}")