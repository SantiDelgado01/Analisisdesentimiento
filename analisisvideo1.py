import pandas as pd
from googleapiclient.discovery import build
import re
import string
import emoji
import os
import sys
from pysentimiento import create_analyzer


try:
 
    print("Configuración de NLTK completa.")
except Exception as e:
    print(f"Error al configurar NLTK: {e}")
    sys.exit()



def preprocesar_texto_mejorado(texto):
   
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    
    texto = texto.lower()
    texto = texto.replace('&quot;', ' ')
    texto = texto.replace('&#39;', ' ')
    texto = re.sub(r'<.*?>', '', texto)
    
    
    return texto.strip()



print("⏳ Cargando modelo de Inteligencia Artificial (pysentimiento/BERT). Esto demora la primera vez...")
analyzer_model = create_analyzer(task="sentiment", lang="es")
print("✅ Modelo cargado y listo.")



API_KEY = "AIzaSyDOqghZ2EDYVGJfAgYVNHFKM-rHoxDTI3o" 
VIDEO_ID = "oI1eamjjTAo" 

youtube = build('youtube', 'v3', developerKey=API_KEY)



def obtener_comentarios_youtube(video_id, max_comentarios=1000):
    comentarios = []
    next_page_token = None
    
    print(f"Iniciando extracción masiva. Objetivo: {max_comentarios} comentarios.")
    
    while len(comentarios) < max_comentarios:
        print(f"  Recolectando página... Total actual: {len(comentarios)}")
        
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token 
        )
        
        try:
            response = request.execute()
        except Exception as e:
            print(f"🛑 Error crítico al conectar con la API (Página {len(comentarios)//100 + 1}): {e}")
            break 
            
        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']
            texto_original = comment['textDisplay']
            
        
            texto_procesado = preprocesar_texto_mejorado(texto_original)
            
            comentarios.append({
                'autor': comment['authorDisplayName'],
                'texto_original': texto_original,
                'texto_procesado': texto_procesado 
            })
            
            if len(comentarios) >= max_comentarios:
                break
        
       
        next_page_token = response.get('nextPageToken')
        
    
        if not next_page_token:
            print("  Se alcanzó el final de los comentarios disponibles en el video.")
            break
            
    print(f"✅ Extracción finalizada. Comentarios recolectados: {len(comentarios)}")
    return pd.DataFrame(comentarios)



def clasificar_comentario(texto_procesado):
    try:
        resultado = analyzer_model.predict(texto_procesado)
        sentimiento = resultado.output
        
        if sentimiento == 'POS':
            return "POSITIVO"
        elif sentimiento == 'NEG':
            return "NEGATIVO"
        else:
            return "NEUTRO"
    except:
        return "NEUTRO"


USAR_API = False
LIMITE_COMENTARIOS = 15000
NOMBRE_CSV_LOCAL = f"comentarios_video_{VIDEO_ID}.csv"
OCULTAR_NEGATIVOS = True


df_comentarios = pd.DataFrame()


if USAR_API:
    print(f"\n--- MODO: EXTRACCIÓN MASIVA (Límite: {LIMITE_COMENTARIOS}) ---")
    df_comentarios = obtener_comentarios_youtube(VIDEO_ID, max_comentarios=LIMITE_COMENTARIOS)
    
    if not df_comentarios.empty:
        try:
            df_comentarios.to_csv(NOMBRE_CSV_LOCAL, index=False, encoding='utf-8')
            print(f"💾 Datos extraídos guardados con éxito en {NOMBRE_CSV_LOCAL}.")
            
        except Exception as e:
            print(f"Advertencia: No se pudo guardar el archivo CSV. {e}")

else:
    print("\n--- MODO: CLASIFICACIÓN LOCAL (Sin API) ---")
    if os.path.exists(NOMBRE_CSV_LOCAL):
        try:
            df_comentarios = pd.read_csv(NOMBRE_CSV_LOCAL)
            df_comentarios['texto_procesado'] = df_comentarios['texto_original'].apply(preprocesar_texto_mejorado)
            print(f"💾 Datos cargados con éxito desde {NOMBRE_CSV_LOCAL}. Total: {len(df_comentarios)}.")
        except Exception as e:
            print(f"🛑 Error al cargar el archivo CSV local: {e}")
            sys.exit()
    else:
        print(f"🛑 Error: Archivo local '{NOMBRE_CSV_LOCAL}' no encontrado. ¡Cambia USAR_API a True para extraer!")
        sys.exit()



if not df_comentarios.empty:
    

    print(f"\n🧠 Aplicando análisis de sentimiento (pysentimiento) a {len(df_comentarios)} comentarios...")
    sentimientos_clasificados = []
    total_comentarios = len(df_comentarios)

    for i, texto in enumerate(df_comentarios['texto_procesado']):
    
        sentimiento = clasificar_comentario(texto)
        sentimientos_clasificados.append(sentimiento)

        
        if (i + 1) % 500 == 0:
            print(f"   Procesados {i + 1}/{total_comentarios} comentarios...")

            
    df_comentarios['sentimiento'] = sentimientos_clasificados
    print("✅ Análisis completado.")


    
    df_analisis = df_comentarios.copy()
    if OCULTAR_NEGATIVOS:
      
        pass 

    print("\n--- RESUMEN DEL SENTIMIENTO EN YOUTUBE ---")
    conteo_sentimientos = df_comentarios['sentimiento'].value_counts(normalize=True) * 100

    print(f"Total de comentarios analizados: {len(df_comentarios)}")
    print("\nDistribución Porcentual:")
    

    positivo = conteo_sentimientos.get('POSITIVO', 0)
    neutro = conteo_sentimientos.get('NEUTRO', 0)
    negativo = conteo_sentimientos.get('NEGATIVO',0 )
    
    print(f"  🟢 Positivos: {positivo:.1f}%")
    print(f"  🟡 Neutros:   {neutro:.1f}%")
    print(f"  🔴 Negativos: {negativo:.1f}%")
        
    if positivo > negativo:
        print("\nConclusión: El video está generando una respuesta mayormente positiva.")
    else:
        print("\nConclusión: La reacción es mixta o inclinada hacia la negatividad/neutralidad.")



    df_positivos = df_comentarios[df_comentarios['sentimiento'] == 'POSITIVO'].sample(n=min(10, len(df_comentarios)), random_state=42)

    print("\n⭐⭐ TOP 10 COMENTARIOS POSITIVOS (Ejemplos Aleatorios) ⭐⭐")
    
    print(df_positivos[['autor', 'texto_original']].head(10).to_markdown(index=False))

    df_negativos_ejemplos = df_comentarios[df_comentarios['sentimiento'] == 'NEGATIVO'].sample(n=min(10, len(df_comentarios)), random_state=42)

    print("\n🔴 TOP 10 COMENTARIOS NEGATIVOS (Ejemplos Aleatorios) 🔴")
    print(df_negativos_ejemplos[['autor', 'texto_original']].head(10).to_markdown(index=False))