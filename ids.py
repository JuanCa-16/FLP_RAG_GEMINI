import json
import os

# Ruta con el prefijo 'r' para evitar errores de Windows
archivo_path = r'C:\Users\juanc\Desktop\TG\FLP_RAG_GEMINI\src\embeddings\new_corpus.jsonl'
archivo_temp = archivo_path + ".tmp"

def modificar_jsonl_existente(ruta, ruta_temporal):
    try:
        with open(ruta, 'r', encoding='utf-8') as entrada, \
             open(ruta_temporal, 'w', encoding='utf-8') as salida:
            
            for idx, linea in enumerate(entrada, start=1):
                linea = linea.strip()
                if not linea: continue # Salta líneas vacías
                
                datos = json.loads(linea)
                
                # Insertar el ID al inicio
                nuevo_objeto = {"id": idx}
                nuevo_objeto.update(datos)
                
                salida.write(json.dumps(nuevo_objeto, ensure_ascii=False) + '\n')
        
        # Paso final: Borrar el original y renombrar el temporal
        os.remove(ruta)
        os.rename(ruta_temporal, ruta)
        print(f"✅ Archivo modificado con éxito: {ruta}")

    except Exception as e:
        print(f"❌ Error durante el proceso: {e}")
        if os.path.exists(ruta_temporal):
            os.remove(ruta_temporal)

# modificar_jsonl_existente(archivo_path, archivo_temp)

def convertir_a_diccionario_por_id(archivo_entrada, archivo_salida):
    data = {}

    with open(archivo_entrada, "r", encoding="utf-8") as f:
        contenido = f.read().strip()

        # Detectar si es JSONL o JSON normal
        if contenido.startswith("["):
            # JSON tipo lista
            objetos = json.loads(contenido)
            for obj in objetos:
                data[str(obj["id"])] = obj
        else:
            # JSONL (una línea por JSON)
            for line in contenido.splitlines():
                obj = json.loads(line)
                data[str(obj["id"])] = obj

    with open(archivo_salida, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

convertir_a_diccionario_por_id(archivo_path, "datos_indexados.json")


def actualizar_embeddings_con_metadata(archivo_sin_embeddings, archivo_con_embeddings, archivo_salida=None):
    """
    Actualiza el archivo con embeddings reemplazando 'tipo' y 'metadata' 
    con los valores del archivo sin embeddings, emparejando por ID.
    
    Args:
        archivo_sin_embeddings: Ruta al archivo JSONL sin embeddings (corpus original)
        archivo_con_embeddings: Ruta al archivo JSONL con embeddings
        archivo_salida: Ruta del archivo de salida. Si es None, sobrescribe archivo_con_embeddings
    
    Returns:
        int: Número de registros actualizados
    """
    print("=" * 70)
    print("🔄 ACTUALIZACIÓN DE TIPO Y METADATA EN EMBEDDINGS")
    print("=" * 70)
    
    # Si no se especifica archivo de salida, usar el mismo que el de embeddings
    if archivo_salida is None:
        archivo_salida = archivo_con_embeddings
        usar_temporal = True
        archivo_temporal = archivo_con_embeddings + ".tmp"
    else:
        usar_temporal = False
    
    try:
        # 1. Cargar datos sin embeddings en un diccionario indexado por ID
        print(f"\n📂 Cargando archivo sin embeddings: {os.path.basename(archivo_sin_embeddings)}")
        datos_originales = {}
        
        with open(archivo_sin_embeddings, 'r', encoding='utf-8') as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                
                obj = json.loads(linea)
                id_registro = obj.get("id")
                if id_registro is not None:
                    datos_originales[id_registro] = {
                        "tipo": obj.get("tipo"),
                        "metadata": obj.get("metadata", {})
                    }
        
        print(f"   ✅ {len(datos_originales)} registros cargados")
        
        # 2. Procesar archivo con embeddings
        print(f"\n📂 Procesando archivo con embeddings: {os.path.basename(archivo_con_embeddings)}")
        
        archivo_escritura = archivo_temporal if usar_temporal else archivo_salida
        
        registros_actualizados = 0
        registros_sin_match = 0
        total_procesados = 0
        
        with open(archivo_con_embeddings, 'r', encoding='utf-8') as entrada, \
             open(archivo_escritura, 'w', encoding='utf-8') as salida:
            
            for linea in entrada:
                linea = linea.strip()
                if not linea:
                    continue
                
                obj_embedding = json.loads(linea)
                id_registro = obj_embedding.get("id")
                total_procesados += 1
                
                # 3. Buscar tipo y metadata original
                if id_registro in datos_originales:
                    # Reemplazar tipo y metadata
                    obj_embedding["tipo"] = datos_originales[id_registro]["tipo"]
                    obj_embedding["metadata"] = datos_originales[id_registro]["metadata"]
                    registros_actualizados += 1
                else:
                    registros_sin_match += 1
                    print(f"   ⚠️  ID {id_registro} no encontrado en archivo original")
                
                # Escribir objeto actualizado
                salida.write(json.dumps(obj_embedding, ensure_ascii=False) + '\n')
        
        # 4. Si usamos archivo temporal, reemplazar el original
        if usar_temporal:
            os.remove(archivo_con_embeddings)
            os.rename(archivo_temporal, archivo_con_embeddings)
        
        # 5. Resumen
        print("\n" + "=" * 70)
        print("📊 RESUMEN DE ACTUALIZACIÓN")
        print("=" * 70)
        print(f"   📝 Total de registros procesados: {total_procesados}")
        print(f"   ✅ Registros actualizados: {registros_actualizados}")
        print(f"   ⚠️  Registros sin match: {registros_sin_match}")
        print(f"   📁 Archivo de salida: {os.path.basename(archivo_salida)}")
        print("=" * 70)
        print("✅ ACTUALIZACIÓN COMPLETADA\n")
        
        return registros_actualizados
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Archivo no encontrado - {e}")
        if usar_temporal and os.path.exists(archivo_temporal):
            os.remove(archivo_temporal)
        return 0
    
    except json.JSONDecodeError as e:
        print(f"\n❌ Error al parsear JSON: {e}")
        if usar_temporal and os.path.exists(archivo_temporal):
            os.remove(archivo_temporal)
        return 0
    
    except Exception as e:
        print(f"\n❌ Error durante la actualización: {e}")
        if usar_temporal and os.path.exists(archivo_temporal):
            os.remove(archivo_temporal)
        return 0
 
 
# ============================================================================
# EJEMPLO DE USO
# ============================================================================
 
# if __name__ == "__main__":
    # Rutas de archivos
    CORPUS_SIN_EMBEDDINGS = r'C:\Users\juanc\Desktop\TG\FLP_RAG_GEMINI\src\embeddings\new_corpus.jsonl'
    CORPUS_CON_EMBEDDINGS = r'C:\Users\juanc\Desktop\TG\FLP_RAG_GEMINI\src\embeddings\new_corpus_embeddings.jsonl'
    
    # Opción 1: Sobrescribir el archivo con embeddings (recomendado)
    actualizar_embeddings_con_metadata(
        archivo_sin_embeddings=CORPUS_SIN_EMBEDDINGS,
        archivo_con_embeddings=CORPUS_CON_EMBEDDINGS
    )