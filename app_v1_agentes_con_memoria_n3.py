# app.py

import streamlit as st
import pandas as pd
import re
from sqlalchemy import text
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import create_sql_query_chain

# ============================================
# 0) Configuración de la Página y Título
# ============================================
st.set_page_config(page_title="IANA para Ventus", page_icon="logo_ventus.png", layout="wide") 

# Creamos columnas para alinear el logo y el título
col1, col2 = st.columns([1, 4]) 

with col1:
    # Asegúrate de tener un archivo "logo_ventus.png" en la misma carpeta
    st.image("logo_ventus.png", width=120) 

with col2:
    st.title("IANA: Tu Asistente IA para Análisis de Datos")
    st.markdown("Soy la red de agentes IA de **VENTUS**. Hazme una pregunta sobre los datos del proyecto IGUANA.")

# ============================================
# 1) Conexión a la Base de Datos y LLMs (con caché para eficiencia)
# ============================================

@st.cache_resource
def get_database_connection():
    """Establece y cachea la conexión a la base de datos."""
    with st.spinner("🔌 Conectando a la base de datos de Ventus..."):
        try:
            db_user = st.secrets["db_credentials"]["user"]
            db_pass = st.secrets["db_credentials"]["password"]
            db_host = st.secrets["db_credentials"]["host"]
            db_name = st.secrets["db_credentials"]["database"]
            uri = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}/{db_name}"

            # << CAMBIO CLAVE AQUÍ >>
            # Añadimos argumentos al motor de SQLAlchemy para manejar timeouts de conexión.
            # Esto evita el error "MySQL server has gone away".
            engine_args = {
                "pool_recycle": 3600,  # Recicla conexiones cada 3600 seg (1 hora)
                "pool_pre_ping": True  # Verifica si la conexión está viva antes de usarla
            }
            
            db = SQLDatabase.from_uri(
                uri, 
                include_tables=["ventus"], 
                engine_args=engine_args  # <-- Pasamos los nuevos argumentos aquí
            ) 
            
            st.success("✅ Conexión a la base de datos establecida.")
            return db
        except Exception as e:
            st.error(f"Error al conectar a la base de datos: {e}")
            return None

@st.cache_resource
def get_llms():
    """Inicializa y cachea los modelos de lenguaje."""
    with st.spinner("🧠 Inicializando la red de agentes IANA..."):
        try:
            api_key = st.secrets["google_api_key"]
            llm_sql = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.1, google_api_key=api_key)
            llm_analista = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
            llm_orq = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.0, google_api_key=api_key)
            st.success("✅ Agentes de IANA listos.")
            return llm_sql, llm_analista, llm_orq
        except Exception as e:
            st.error(f"Error al inicializar los LLMs. Asegúrate de que tu API key es correcta. Error: {e}")
            return None, None, None

db = get_database_connection()
llm_sql, llm_analista, llm_orq = get_llms()

@st.cache_resource
def get_sql_agent(_llm, _db):
    """Crea y cachea el agente SQL."""
    if not _llm or not _db:
        return None
    with st.spinner("🛠️ Configurando agente SQL de IANA..."):
        toolkit = SQLDatabaseToolkit(db=_db, llm=_llm)
        agent = create_sql_agent(
            llm=_llm, 
            toolkit=toolkit, 
            verbose=False,
            top_k=1000)
        st.success("✅ Agente SQL configurado.")
        return agent

agente_sql = get_sql_agent(llm_sql, db)

# ============================================
# NUEVA FUNCIÓN AUXILIAR DE MEMORIA
# ============================================
def get_history_text(chat_history: list, n_turns=3) -> str:
    """Extrae el texto de las últimas N vueltas del historial de chat."""
    
    if not chat_history or len(chat_history) <= 1:
        return ""

    history_text = []
    # Itera hacia atrás, saltando el último mensaje (la consulta actual del usuario)
    # Buscamos n_turns * 2 mensajes (pares de usuario/asistente)
    relevant_history = chat_history[-(n_turns * 2 + 1) : -1]

    for msg in relevant_history:
        content = msg.get("content", {})
        text_content = ""
        
        # El contenido puede ser un dict (para IANA o usuario procesado) o str (para usuario inicial)
        if isinstance(content, dict):
            text_content = content.get("texto", "") 
        elif isinstance(content, str):
            text_content = content
        
        if text_content:
            role = "Usuario" if msg["role"] == "user" else "IANA"
            history_text.append(f"{role}: {text_content}")

    if not history_text:
        return ""
    
    # Devuelve el contexto formateado
    return "\n--- Contexto de Conversación Anterior (Últimas 3 vueltas) ---\n" + "\n".join(history_text) + "\n--- Fin del Contexto ---\n"


# ============================================
# 2) Funciones de Agentes (Lógica Principal)
# ============================================
def markdown_table_to_df(texto: str) -> pd.DataFrame:
    """Convierte una tabla en formato Markdown a un DataFrame de pandas."""
    lineas = [l.strip() for l in texto.splitlines() if l.strip().startswith('|')]
    if not lineas: return pd.DataFrame()
    lineas = [l for l in lineas if not re.match(r'^\|\s*-', l)]
    filas = [[c.strip() for c in l.strip('|').split('|')] for l in lineas]
    if len(filas) < 2: return pd.DataFrame()
    header, data = filas[0], filas[1:]
    df = pd.DataFrame(data, columns=header)
    for c in df.columns:
        # Limpiamos los datos que vienen del Markdown
        s = df[c].astype(str).str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
        try: df[c] = pd.to_numeric(s)
        except Exception: df[c] = s
    return df

def _df_preview(df: pd.DataFrame, n: int = 20) -> str:
    """Crea un preview en texto de un DataFrame."""
    if df is None or df.empty: return ""
    try: return df.head(n).to_markdown(index=False)
    except Exception: return df.head(n).to_string(index=False)


def ejecutar_sql_real(pregunta_usuario: str):
    st.info("🤖 Entendido. El agente de datos de IANA está traduciendo tu pregunta a SQL...")
    
    prompt_con_instrucciones = f"""
    Tu tarea es generar una consulta SQL **únicamente contra la tabla 'ventus'** para responder la pregunta del usuario.
    Debes usar el contexto de la conversación anterior para resolver pronombres o preguntas de seguimiento (ej. "ese proveedor", "esos productos", "cuántos fueron").

    {hist_text}
    
    REGLA 1: La única tabla que debes usar es "ventus".
    REGLA 2: Los campos de costo y cantidad (Total_COP, Total_USD, Cantidad, etc.) YA SON numéricos (DECIMAL). 
             Puedes usarlos directamente con SUM(), AVG(), etc. NO necesitas usar CAST.

    Columnas disponibles en la tabla "ventus":
    - `_item` (Texto, ID)
    - `Numerador` (Texto, ID)
    - `Codigo_de_proyecto` (Texto)
    - `Rubro_anterior` (Texto)
    - `Rubro_CF` (Texto)
    - `Descripcion` (Texto)
    - `Fecha_aprobacion` (DATE) <- Usa esta columna para agrupar por día, mes o año.
    - `Numero_de_documento` (Texto, ID)
    - `Proveedor` (Texto, Categoría para agrupar)
    - `Moneda` (Texto, Ej: 'COP', 'USD')
    - `Descripcion_Linea` (Texto)
    - `Subtotal_COP` (DECIMAL)
    - `IVA_COP` (DECIMAL)
    - `Total_COP` (DECIMAL) <- Métrica clave de costo.
    - `SubTotal_USD` (DECIMAL)
    - `IVA_USD` (DECIMAL)
    - `Total_USD` (DECIMAL) <- Métrica clave de costo.
    - `Comentarios` (Texto)
    - `Condicion_de_pago` (Texto)
    - `Condiciones_Comerciales` (Texto)
    - `Comprador` (Texto, Categoría para agrupar)
    - `Cantidad` (DECIMAL)
    - `Cluster` (Texto, Categoría)
    - `Producto` (Texto, Aquí está el campo más importante donde se indentifica el trabajo realizado en el proyecto, ej: Transporte de materiales, Bultos de Cemento, Bolsas de basura, etc...)
    - `Grupo_Producto` (Texto, Categoría para agrupar)
    - `Familia` (Texto, Categoría para agrupar)
    - `Tipo` (Texto, Categoría para agrupar)

    REGLA 3 (Agrupación): Presta mucha atención a palabras como 'diariamente', 'mensual', etc.
    REGLA 4 (LIMIT): Nunca, bajo ninguna circunstancia, agregues un 'LIMIT' al final de la consulta.

    Pregunta original del usuario (actual): "{pregunta_usuario}"
    """    
    try:
        query_chain = create_sql_query_chain(llm_sql, db)
        sql_query = query_chain.invoke({"question": prompt_con_instrucciones})
        
        # Limpieza estándar de ```sql
        sql_query = re.sub(r"^\s*```sql\s*|\s*```\s*$", "", sql_query, flags=re.IGNORECASE).strip()

        # Forzamos la eliminación del LIMIT.
        sql_query_limpia = re.sub(r'LIMIT\s+\d+\s*;?$', '', sql_query, flags=re.IGNORECASE | re.DOTALL).strip()

        # Mostramos la consulta que REALMENTE vamos a ejecutar (la limpia)
        st.code(sql_query_limpia, language='sql')
        
        with st.spinner("⏳ Ejecutando la consulta en la base de datos..."):
            with db._engine.connect() as conn:
                # Ejecutamos la consulta LIMPIA, sin el LIMIT
                df = pd.read_sql(text(sql_query_limpia), conn)
                
        st.success("✅ ¡Consulta ejecutada!")
        return {"sql": sql_query_limpia, "df": df} # Devolvemos los datos limpios
    
    except Exception as e:
        st.warning(f"❌ Error en la consulta directa. Intentando un método alternativo... Error: {e}")
        return {"sql": None, "df": None, "error": str(e)}


def ejecutar_sql_en_lenguaje_natural(pregunta_usuario: str, hist_text: str):
    st.info("🤔 La consulta directa falló. Activando el agente SQL experto de IANA como plan B.")
    
    prompt_sql = (
        "Tu tarea es responder la pregunta del usuario consultando la base de datos (tabla 'ventus'). "
        "Debes usar el contexto de la conversación anterior para resolver pronombres o preguntas de seguimiento (ej. 'esos', 'ese proveedor')."
        f"{hist_text}"
        "Debes devolver ÚNICAMENTE una tabla de datos en formato Markdown. "
        "REGLA CRÍTICA 1: Devuelve SIEMPRE TODAS las filas de datos que encuentres. NUNCA resumas, trunques ni expliques los resultados. "
        "REGLA CRÍTICA 2 (MUY IMPORTANTE): El SQL que generes internamente NO DEBE CONTENER 'LIMIT'. Debes devolver todos los resultados."
        "REGLA NUMÉRICA: Columnas como Total_COP y Cantidad ya son numéricas (DECIMAL). "
        "Responde siempre en español. "
        "\nPregunta actual del usuario: "
        f"{pregunta_usuario}"
    )
    try:
        with st.spinner("💬 Pidiendo al agente SQL que responda en lenguaje natural..."):
            # Pasamos el prompt completo al agente
            res = agente_sql.invoke(prompt_sql) 
            texto = res["output"] if isinstance(res, dict) and "output" in res else str(res)
        st.info("📝 Recibí una respuesta en texto. Intentando convertirla en una tabla de datos...")
        df_md = markdown_table_to_df(texto)
        return {"texto": texto, "df": df_md}
    except Exception as e:
        st.error(f"❌ El agente SQL experto también encontró un problema: {e}")
        return {"texto": f"[SQL_ERROR] {e}", "df": pd.DataFrame()}


def analizar_con_datos(pregunta_usuario: str, datos_texto: str, df: pd.DataFrame | None):
    st.info("\n🧠 Ahora, el analista experto de IANA está examinando los datos...")
    
    # << ESTE ES EL PROMPT DE ANÁLISIS COMPLETO (reemplaza tu versión anterior) >>
    # Este prompt obliga a la IA a seguir tus instrucciones específicas 
    # de Pareto, concentración, promedios y el formato ejecutivo con emojis.
    # También aumenté el preview de 20 a 30 filas para darle más contexto.
    
    prompt_analisis = f"""
    Eres IANA, analista de datos senior en Ventus. Tu tarea es realizar un análisis ejecutivo rápido sobre los datos proporcionados.
    
    Pregunta Original del Usuario: {pregunta_usuario}
    
    Datos para tu análisis (preview de las primeras 30 filas):
    {_df_preview(df, 30)}

    ---
    INSTRUCCIONES DE ANÁLISIS OBLIGATORIAS:
    Sigue estos pasos para analizar la tabla de resultados (costos, métricas, etc.):
    1. Calcular totales (ej. SUM(Total_COP) o SUM(Cantidad)) y porcentajes clave (ej. participación de los 5 items más grandes, distribución por 'Tipo' o 'Proveedor', % acumulado si aplica).
    2. Detectar concentración (Principio de Pareto): ¿Pocos registros (ej. 20% de los proveedores/productos) explican una gran parte del total (ej. 80% del costo)?
    3. Identificar patrones temporales (basado en 'Fecha_aprobacion'): ¿Hay días o periodos (ej. fin de mes) con concentración inusual de gastos o actividad?
    4. Analizar dispersión: Calcula el "ticket promedio" (Costo Total / Cantidad Total, o Costo Total / # de Registros). Compara los valores más grandes contra los más pequeños.

    ---
    FORMATO DE ENTREGA OBLIGATORIO:
    Entrega el resultado EXACTAMENTE en estos 2 bloques. Usa frases cortas en bullets. Sé muy breve, directo y diciente para un gerente.

    📌 Resumen Ejecutivo:
    - (Aquí van los hallazgos principales y patrones detectados, con números clave. Ej: "El 80% del costo se concentra en solo 3 proveedores.")
    - (Bullet point 2 del hallazgo principal.)
    - (Bullet point 3 del hallazgo principal. Si no hay más, no agregues bullets vacíos.)

    🔍 Números de referencia:
    - (Bullet point con el Total General. Ej: "Costo Total (COP): $XX.XXX.XXX")
    - (Bullet point con el Promedio. Ej: "Costo Promedio por Transacción: $XX.XXX")
    - (Bullet point con el ratio de concentración. Ej: "Top 5 Productos representan: XX% del total.")

    ⚠ Importante: No describas lo obvio de la tabla (como "la tabla muestra proveedores"). Ve directo a los números y al insight.
    """
    
    with st.spinner("💡 Generando análisis y recomendaciones avanzadas..."):
        analisis = llm_analista.invoke(prompt_analisis).content
    st.success("💡 ¡Análisis completado!")
    return analisis


def responder_conversacion(pregunta_usuario: str):
    """Activa el modo conversacional de IANA."""
    st.info("💬 Activando modo de conversación...")
    
    # << CAMBIO CLAVE: Personalidad mejorada con manejo de bromas >>
    prompt_personalidad = f"""
    Tu nombre es IANA, una asistente de IA de Ventus.
    Tu personalidad es amable, servicial, profesional y con un toque de empatía humana.
    Tu objetivo principal es ayudar a analizar los datos de 'Ventus'.
    
    REGLA DE CONVERSACIÓN: Eres eficiente, pero no eres un robot sin personalidad. Si el usuario hace un comentario casual, un saludo o una broma ligera (como "no tienes sentido del humor"), responde amablemente siguiendo la corriente por un momento, antes de redirigirlo a tus capacidades de análisis de datos.
    
    EJEMPLO DE CÓMO MANEJAR UNA BROMA:
    Usuario: No tienes sentido del humor.
    Respuesta Amable: ¡Jaja, buen punto! Digamos que mi fuerte son los números y el análisis de datos. Mi sentido del humor todavía está en versión beta. ¿Pero sabes qué es mejor que un chiste? Encontrar datos útiles en tus proyectos. ¿Te ayudo con eso?

    Tus capacidades principales (ejemplos): "Puedo sumar el Total COP por Proveedor", "puedo contar cuántos registros hay por Familia", "puedo analizar los costos del proyecto", etc.
    NO intentes generar código SQL. Solo responde de forma conversacional.
    Responde siempre en español.

    {hist_text}

    Pregunta del usuario: "{pregunta_usuario}"
    """
    respuesta = llm_analista.invoke(prompt_personalidad).content
    return {"texto": respuesta, "df": None, "analisis": None}


# --- Orquestador Principal ---

def clasificar_intencion(pregunta: str) -> str:
    prompt_orq = f"""
    Devuelve UNA sola palabra exacta según la intención del usuario:
    - `consulta`: si pide extraer, filtrar o contar datos específicos. (Ej: 'cuánto sumó el proveedor X?', 'dame el total_cop por tipo')
    - `analista`: si pide interpretar, resumir o recomendar acciones sobre datos. (Ej: 'analiza las tendencias de costo')
    - `conversacional`: si es un saludo, una pregunta general sobre tus capacidades (Ej: '¿qué puedes hacer?' o '¿cómo me puedes ayudar?'), o no está relacionada con datos específicos.
    Mensaje: {pregunta}
    """
    clasificacion = llm_orq.invoke(prompt_orq).content.strip().lower().replace('"', '').replace("'", "")
    return clasificacion


def obtener_datos_sql(pregunta_usuario: str, hist_text: str) -> dict:
    # Pasamos el historial a ambos planes
    res_real = ejecutar_sql_real(pregunta_usuario, hist_text)
    
    if res_real.get("df") is not None and not res_real["df"].empty:
        return {"sql": res_real["sql"], "df": res_real["df"], "texto": None}
    
    # Si Plan A falla, Plan B también recibe el historial
    res_nat = ejecutar_sql_en_lenguaje_natural(pregunta_usuario, hist_text)
    return {"sql": None, "df": res_nat["df"], "texto": res_nat["texto"]}


def orquestador(pregunta_usuario: str, chat_history: list):
    with st.expander("⚙️ Ver Proceso de IANA", expanded=False):
        st.info(f"🚀 Recibido: '{pregunta_usuario}'")
        
        # << CAMBIO DE MEMORIA >>
        # Generamos el texto del historial UNA VEZ para pasarlo a todos los agentes.
        hist_text = get_history_text(chat_history, n_turns=3) # Usamos 3 niveles como sugeriste
        
        with st.spinner("🔍 IANA está analizando tu pregunta..."):
            clasificacion = clasificar_intencion(pregunta_usuario)
        st.success(f"✅ ¡Intención detectada! Tarea: {clasificacion.upper()}.")

        if clasificacion == "conversacional":
            # Pasamos el historial al agente conversacional
            return responder_conversacion(pregunta_usuario, hist_text)

        # --- INICIO DE LA LÓGICA DE MEMORIA DE DATAFRAME (Esto sigue igual) ---
        if clasificacion == "analista":
            palabras_clave_contexto = [
                "esto", "esos", "esa", "información", 
                "datos", "tabla", "anterior", "acabas de dar"
            ]
            es_pregunta_de_contexto = any(palabra in pregunta_usuario.lower() for palabra in palabras_clave_contexto)

            if es_pregunta_de_contexto and len(chat_history) > 1:
                mensaje_anterior = chat_history[-2]
                
                if mensaje_anterior["role"] == "assistant" and "df" in mensaje_anterior["content"]:
                    df_contexto = mensaje_anterior["content"]["df"]
                    
                    if df_contexto is not None and not df_contexto.empty:
                        st.info("💡 Usando datos de la conversación anterior para el análisis...")
                        # El analista también recibe el contexto de texto Y el DF anterior
                        analisis = analizar_con_datos(pregunta_usuario, hist_text, df_contexto)
                        return {"tipo": "analista", "df": df_contexto, "texto": None, "analisis": analisis}
        # --- FIN DE LA LÓGICA DE MEMORIA DE DATAFRAME ---

        # Si no es una pregunta de contexto, sigue el flujo normal
        # Pasamos el historial a los agentes SQL
        res_datos = obtener_datos_sql(pregunta_usuario, hist_text)
        resultado = {"tipo": clasificacion, **res_datos, "analisis": None}
        
        if clasificacion == "analista":
            if res_datos.get("df") is not None and not res_datos["df"].empty:
                # El analista recibe el contexto de texto y el NUEVO DF
                analisis = analizar_con_datos(pregunta_usuario, hist_text, res_datos["df"])
                resultado["analisis"] = analisis
            else:
                resultado["texto"] = "Para poder realizar un análisis, primero necesito datos. Por favor, haz una pregunta más específica para obtener la información que quieres analizar."
                resultado["df"] = None
    return resultado


# ============================================
# 3) Interfaz de Chat de Streamlit
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": {"texto": "¡Hola! Soy IANA, tu asistente de IA de Ventus. Estoy lista para analizar los datos de tus proyectos. ¿Qué te gustaría saber?"}}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Aseguramos que el contenido sea un diccionario antes de acceder
        content = message.get("content", {})
        if isinstance(content, dict):
            if "texto" in content and content["texto"]: st.markdown(content["texto"])
            if "df" in content and content["df"] is not None: st.dataframe(content["df"])
            if "analisis" in content and content["analisis"]: st.markdown(content["analisis"])
        elif isinstance(content, str): # Fallback por si acaso
             st.markdown(content)


if prompt := st.chat_input("Pregunta por costos, proveedores, familia..."):
    if not all([db, llm_sql, llm_analista, llm_orq, agente_sql]):
        st.error("La aplicación no está completamente inicializada. Revisa los errores de conexión o de API key.")
    else:
        st.session_state.messages.append({"role": "user", "content": {"texto": prompt}})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Pasamos el historial de mensajes al orquestador para la lógica de memoria
            res = orquestador(prompt, st.session_state.messages)
            
            st.markdown(f"### IANA responde a: '{prompt}'")
            if res.get("df") is not None and not res["df"].empty:
                st.dataframe(res["df"])
            
            if res.get("texto"):
                 st.markdown(res["texto"])
            
            if res.get("analisis"):
                st.markdown("---")
                st.markdown("### 🧠 Análisis de IANA para Ventus") 
                st.markdown(res["analisis"])
                

            st.session_state.messages.append({"role": "assistant", "content": res})

