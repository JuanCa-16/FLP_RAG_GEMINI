import json
import os

# Ruta con el prefijo 'r' para evitar errores de Windows
archivo_path = r'C:\Users\juanc\Desktop\TG\FLP_RAG_GEMINI\src\embeddings\corpus_con_ejemplos.jsonl'
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