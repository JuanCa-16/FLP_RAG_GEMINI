import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import time
import re

URL_PRINCIPAL = "https://cardel.github.io/notasUniversidad/2025-II/FLP/Contenido/"
CARPETA_RAIZ = "GIT_FLP"

def limpiar_nombre_archivo(nombre):
    # Elimina caracteres no permitidos en nombres de archivos
    return re.sub(r'[\\/*?:"<>|]', "", nombre).strip()

def extraer_notas_completas():
    if not os.path.exists(CARPETA_RAIZ):
        os.makedirs(CARPETA_RAIZ)

    try:
        print(f"--- Iniciando extracción desde: {URL_PRINCIPAL} ---")
        res_principal = requests.get(URL_PRINCIPAL)
        soup_principal = BeautifulSoup(res_principal.text, 'html.parser')
        
        # 1. Obtener enlaces de las 13 clases (Padres)
        main_content = soup_principal.find('article', class_='md-content__inner')
        enlaces_padre = main_content.find_all('a', href=True)
        
        for link_p in enlaces_padre:
            nombre_padre = link_p.get_text(strip=True)
            href_p = link_p['href']
            
            if "Clase" in nombre_padre and "../" in href_p:
                url_padre = urljoin(URL_PRINCIPAL, href_p)
                nombre_carpeta_clase = limpiar_nombre_archivo(nombre_padre)
                ruta_clase = os.path.join(CARPETA_RAIZ, nombre_carpeta_clase)
                
                if not os.path.exists(ruta_clase):
                    os.makedirs(ruta_clase)
                
                # 2. Entrar a la clase padre para buscar subtemas (Hijos)
                print(f"\n📂 Procesando {nombre_padre}...")
                res_padre = requests.get(url_padre)
                soup_padre = BeautifulSoup(res_padre.text, 'html.parser')
                contenido_padre = soup_padre.find('article', class_='md-content__inner')
                
                if not contenido_padre: continue
                enlaces_hijos = contenido_padre.find_all('a', href=True)
                
                for link_h in enlaces_hijos:
                    nombre_hijo = link_h.get_text(strip=True)
                    href_h = link_h['href']
                    
                    if not href_h.startswith('#') and nombre_hijo != "":
                        url_hijo = urljoin(url_padre, href_h)
                        
                        # Evitar volver a la página de contenidos o al mismo padre
                        if "Contenido" in nombre_hijo or url_hijo == url_padre:
                            continue

                        try:
                            # 3. Entrar al contenido final y extraer texto limpio
                            res_final = requests.get(url_hijo)
                            soup_final = BeautifulSoup(res_final.text, 'html.parser')
                            
                            # MkDocs guarda el contenido real en <article>
                            articulo = soup_final.find('article')
                            
                            if articulo:
                                # Eliminamos posibles elementos internos que no queremos (como links de anclaje ¶)
                                for anchor in articulo.find_all('a', class_='headerlink'):
                                    anchor.decompose()
                                
                                texto_final = articulo.get_text(separator='\n', strip=True)
                                
                                # Guardar en archivo TXT
                                nombre_txt = limpiar_nombre_archivo(nombre_hijo) + ".txt"
                                with open(os.path.join(ruta_clase, nombre_txt), "w", encoding="utf-8") as f:
                                    f.write(f"ORIGEN: {url_hijo}\n")
                                    f.write("="*50 + "\n\n")
                                    f.write(texto_final)
                                
                                print(f"   ✅ Guardado: {nombre_txt}")
                            
                            time.sleep(0.2) # Pausa breve para no ser bloqueados
                            
                        except Exception as e:
                            print(f"   ❌ Error en subtema {nombre_hijo}: {e}")

        print("\n--- ¡Proceso completado con éxito! ---")

    except Exception as e:
        print(f"Error general: {e}")

extraer_notas_completas()