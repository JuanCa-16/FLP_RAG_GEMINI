import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

URL_PRINCIPAL = "https://cardel.github.io/notasUniversidad/2025-II/FLP/Contenido/"

def listar_enlaces_hijos():
    try:
        print(f"Buscando clases padre en: {URL_PRINCIPAL}\n")
        res_principal = requests.get(URL_PRINCIPAL)
        soup_principal = BeautifulSoup(res_principal.text, 'html.parser')
        
        # 1. Encontrar los enlaces de las 13 clases en la página principal
        main_content = soup_principal.find('article', class_='md-content__inner')
        enlaces_padre = main_content.find_all('a', href=True)
        
        links_padre_validos = []
        for link in enlaces_padre:
            texto = link.get_text(strip=True)
            href = link['href']
            # Filtramos que sea una de las clases y que no sea un link externo
            if "Clase" in texto and "../" in href:
                url_padre = urljoin(URL_PRINCIPAL, href)
                links_padre_validos.append((texto, url_padre))

        print(f"Se encontraron {len(links_padre_validos)} clases padre. Entrando a buscar enlaces hijos...\n")
        print(f"{'CLASE PADRE':<25} | {'TEMA (HIJO)':<40} | {'URL FINAL'}")
        print("-" * 110)

        # 2. Visitar cada clase padre para buscar sus hijos
        total_hijos = 0
        for nombre_padre, url_padre in links_padre_validos:
            try:
                res_hijo = requests.get(url_padre)
                soup_hijo = BeautifulSoup(res_hijo.text, 'html.parser')
                
                # Buscamos enlaces dentro del contenido de esa clase específica
                # Usualmente en MkDocs están dentro de la lista de contenidos o el article
                contenido_hijo = soup_hijo.find('article', class_='md-content__inner')
                if not contenido_hijo: continue
                
                links_hijos = contenido_hijo.find_all('a', href=True)
                
                for l_hijo in links_hijos:
                    texto_hijo = l_hijo.get_text(strip=True)
                    href_hijo = l_hijo['href']
                    
                    # Evitamos enlaces de anclaje (que empiezan con #) y el enlace a la misma página
                    if not href_hijo.startswith('#') and texto_hijo != "":
                        url_final = urljoin(url_padre, href_hijo)
                        
                        # Evitamos que el hijo sea el mismo padre o el índice superior
                        if url_final != url_padre and "Contenido" not in texto_hijo:
                            print(f"{nombre_padre:<25} | {texto_hijo:<40} | {url_final}")
                            total_hijos += 1
                            
            except Exception as e:
                print(f"Error al entrar a {nombre_padre}: {e}")

        print("-" * 110)
        print(f"Búsqueda finalizada. Total de temas (hijos) encontrados: {total_hijos}")

    except Exception as e:
        print(f"Error general: {e}")

listar_enlaces_hijos()