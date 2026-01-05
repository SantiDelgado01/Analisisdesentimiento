import pandas as pd
from googleapiclient.discovery import build
import re
import string
import emoji
import os
import sys
from pysentimiento import create_analyzer
import seaborn as sns # type: ignore # Para gráficos bonitos
import matplotlib.pyplot as plt # type: ignore # Para control de gráficos


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




API_KEY = "apikey" 
youtube = build('youtube', 'v3', developerKey=API_KEY)
VIDEOS_A_COMPARAR = [
    {
        'titulo': 'Musica pauloLondra',
        'id': '2yGZPCjtGJ8',
        'csv': 'comentarios_video_2yGZPCjtGJ8.csv'
    },
    {
        'titulo': 'Noticia Canal26',
        'id': 'Fclb2EUJxbQ', 
        'csv': 'comentarios_video_Fclb2EUJxbQ.csv'
    },
    {
        'titulo': 'Comedia HablandoHuevadas',
        'id': 'oI1eamjjTAo', 
        'csv': 'comentarios_video_oI1eamjjTAo.csv'
    }
]



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





def ejecutar_analisis_para_video(video_id, titulo_video, nombre_csv_local, usar_api=False):
    """
    Ejecuta el pipeline completo de extracción/carga y análisis de sentimiento
    para un único video, y devuelve los porcentajes de sentimiento.
    """
    
    df_comentarios = pd.DataFrame()
    
    
    if usar_api:
        print(f"\n--- INICIANDO ANÁLISIS para: {titulo_video} (MODO API) ---")
        df_comentarios = obtener_comentarios_youtube(video_id, max_comentarios=15000)
        
        if not df_comentarios.empty:
            df_comentarios.to_csv(nombre_csv_local, index=False, encoding='utf-8')
            print(f"💾 Datos extraídos guardados en {nombre_csv_local}.")
    else:
        
        print(f"\n--- INICIANDO ANÁLISIS para: {titulo_video} (MODO LOCAL) ---")
        if os.path.exists(nombre_csv_local):
            try:
                df_comentarios = pd.read_csv(nombre_csv_local)
                df_comentarios['texto_procesado'] = df_comentarios['texto_original'].apply(preprocesar_texto_mejorado)
                print(f"💾 Datos cargados con éxito desde {nombre_csv_local}. Total: {len(df_comentarios)}.")
            except Exception as e:
                print(f"🛑 Error al cargar CSV: {e}")
                return None
        else:
            print(f"🛑 Error: Archivo local '{nombre_csv_local}' no encontrado. Ejecuta con USAR_API=True primero.")
            return None

    if df_comentarios.empty:
        return None

    
    total_comentarios = len(df_comentarios)
    print(f"\n🧠 Aplicando análisis de sentimiento (pysentimiento) a {total_comentarios} comentarios...")
    
    sentimientos_clasificados = []
    for i, texto in enumerate(df_comentarios['texto_procesado']):
        sentimiento = clasificar_comentario(texto)
        sentimientos_clasificados.append(sentimiento)
        if (i + 1) % 500 == 0:
            print(f"   Procesados {i + 1}/{total_comentarios} comentarios...")
            
    df_comentarios['sentimiento'] = sentimientos_clasificados
    print("✅ Análisis completado.")
    
   
    conteo_sentimientos = df_comentarios['sentimiento'].value_counts(normalize=True) * 100
    
    positivo = conteo_sentimientos.get('POSITIVO', 0)
    neutro = conteo_sentimientos.get('NEUTRO', 0)
    negativo = conteo_sentimientos.get('NEGATIVO', 0)
    
    
    print("\n--- RESUMEN RÁPIDO ---")
    print(f"Video: {titulo_video}")
    print(f"  🟢 Positivos: {positivo:.1f}%")
    print(f"  🟡 Neutros:   {neutro:.1f}%")
    print(f"  🔴 Negativos: {negativo:.1f}%")

    
    return {
        'Video': titulo_video,
        'Positivo': positivo,
        'Neutro': neutro,
        'Negativo': negativo
    }
    

USAR_API_EXTRACCION = False 
resultados_finales = []


for video in VIDEOS_A_COMPARAR:
    
    
    resultado = ejecutar_analisis_para_video(
        video_id=video['id'], 
        titulo_video=video['titulo'], 
        nombre_csv_local=video['csv'],
        usar_api=USAR_API_EXTRACCION
    )
    
    if resultado:
        resultados_finales.append(resultado)


if len(resultados_finales) >= 2:
    
    print("\n\n--- GENERANDO GRÁFICO COMPARATIVO ---")
    

    df_comparativo = pd.DataFrame(resultados_finales)
    
   
    df_long = df_comparativo.melt(
        id_vars='Video', 
        var_name='Sentimiento', 
        value_name='Porcentaje'
    )
    
   
    orden_sentimiento = ['Positivo', 'Neutro', 'Negativo']
    df_long['Sentimiento'] = pd.Categorical(df_long['Sentimiento'], categories=orden_sentimiento, ordered=True)
    
    colores = {'Positivo': '#4CAF50', 'Neutro': '#FFC107', 'Negativo': '#F44336'} # Verde, Amarillo, Rojo

    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    g = sns.barplot(
        x='Sentimiento', 
        y='Porcentaje', 
        hue='Video',
        data=df_long, 
        palette=[colores[s] for s in orden_sentimiento] 
    )


    for container in g.containers:
        g.bar_label(container, fmt='%.1f%%', fontsize=9)
    
   
    plt.title('Comparación de Sentimiento de Audiencias en YouTube (Múltiples Videos)', fontsize=16)
    plt.xlabel('Categoría de Sentimiento', fontsize=12)
    plt.ylabel('Porcentaje (%)', fontsize=12)
    plt.legend(title='Video Analizado', loc='upper right')
    plt.ylim(0, 100) 
    
   
    nombre_archivo_grafico = 'comparacion_sentimiento_final.png'
    plt.savefig(nombre_archivo_grafico)
    plt.show()
    
    print(f"✅ Gráfico '{nombre_archivo_grafico}' generado con éxito.")

else:
    print("🛑 No hay suficientes datos para generar el gráfico comparativo.")