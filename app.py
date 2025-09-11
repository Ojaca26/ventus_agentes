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

# ... (El código de configuración de la página y las columnas sigue igual) ...
# ============================================
# 0) Configuración de la Página y Título
# ============================================
st.set_page_config(page_title="IANA para Ventus", page_icon="logo_ventus.png", layout="wide") 

# Creamos columnas para alinear el logo y el título
col1, col2 = st.columns([1, 4]) 

with col1:
    st.image("logo_ventus.png", width=120) 

with col2:
    st.title("IANA: Tu Asistente IA para Análisis de Datos")
    st.markdown("Soy la red de agentes IA de **VENTUS**. Hazme una pregunta sobre los datos del proyecto IGUANA.")

# ============================================
# 1) Conexión a la Base de Datos y LLMs (con caché para eficiencia)
# ============================================

@st.cache_resource
def get_database_connection():
    # ... (Esta función no cambia) ...
    """Establece y cachea la conexión a la base de datos."""
    with st.spinner("🔌 Conectando a la base de datos de Ventus..."):
        try:
            db_user = st.secrets["db_credentials"]["user"]
            db_pass = st.secrets["db_credentials"]["password"]
            db_host = st.secrets["db_credentials"]["host"]
            db_name = st.secrets["db_credentials"]["database"]
            uri = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}/{db_name}"

            engine_args = {
                "pool_recycle": 3600,
                "pool_pre_ping": True
            }
            
            db = SQLDatabase.from_uri(
                uri, 
                include_tables=["ventus"], 
                engine_args=engine_args
            ) 
            
            st.success("✅ Conexión a la base de datos establecida.")
            return db
        except Exception as e:
            st.error(f"Error al conectar a la base de datos: {e}")
            return None

@st.cache_resource
def get_llms():
    # ... (Esta función no cambia, ya habíamos añadido el validador) ...
    """Inicializa y cachea los modelos de lenguaje."""
    with st.spinner("🧠 Inicializando la red de agentes IANA..."):
        try:
            api_key = st.secrets["google_api_key"]
            llm_sql = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.1, google_api_key=api_key)
            llm_analista = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
            llm_orq = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.0, google_api_key=api_key)
            llm_validador = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.0, google_api_key=api_key) 
            st.success("✅ Agentes de IANA listos.")
            return llm_sql, llm_analista, llm_orq, llm_validador
        except Exception as e:
            st.error(f"Error al inicializar los LLMs. Asegúrate de que tu API key es correcta. Error: {e}")
            return None, None, None, None

db = get_database_connection()
llm_sql, llm_analista, llm_orq, llm_validador = get_llms()

@st.cache_resource
def get_sql_agent(_llm, _db):
    # ... (Esta función no cambia) ...
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


# ... (Las funciones auxiliares como get_history_text, markdown_table_to_df, _df_preview no cambian) ...
def get_history_text(chat_history: list, n_turns=3) -> str:
    # ... (No changes here) ...
    if not chat_history or len(chat_history) <= 1: return ""
    history_text = []
    relevant_history = chat_history[-(n_turns * 2 + 1) : -1]
    for msg in relevant_history:
        content = msg.get("content", {})
        text_content = ""
        if isinstance(content, dict): text_content = content.get("texto", "") 
        elif isinstance(content, str): text_content = content
        if text_content:
            role = "Usuario" if msg["role"] == "user" else "IANA"
            history_text.append(f"{role}: {text_content}")
    if not history_text: return ""
    return "\n--- Contexto de Conversación Anterior (Últimas 3 vueltas) ---\n" + "\n".join(history_text) + "\n--- Fin del Contexto ---\n"

def markdown_table_to_df(texto: str) -> pd.DataFrame:
    # ... (No changes here) ...
    lineas = [l.strip() for l in texto.splitlines() if l.strip().startswith('|')]
    if not lineas: return pd.DataFrame()
    lineas = [l for l in lineas if not re.match(r'^\|\s*-', l)]
    filas = [[c.strip() for c in l.strip('|').split('|')] for l in lineas]
    if len(filas) < 2: return pd.DataFrame()
    header, data = filas[0], filas[1:]
    df = pd.DataFrame(data, columns=header)
    for c in df.columns:
        s = df[c].astype(str).str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
        try: df[c] = pd.to_numeric(s)
        except Exception: df[c] = s
    return df

def _df_preview(df: pd.DataFrame, n: int = 20) -> str:
    # ... (No changes here) ...
    if df is None or df.empty: return ""
    try: return df.head(n).to_markdown(index=False)
    except Exception: return df.head(n).to_string(index=False)


# ... (Las funciones de los agentes como ejecutar_sql_real, analizar_con_datos, etc., no cambian) ...
# (Solo asegúrate de que aceptan `hist_text` como argumento)
def ejecutar_sql_real(pregunta_usuario: str, hist_text: str):
    # ... (No changes here, assuming it's the same as the previous version) ...
    st.info("🤖 Entendido. El agente de datos de IANA está traduciendo tu pregunta a SQL...")
    # (Full function code from previous version)
    pass # Placeholder for brevity

def ejecutar_sql_en_lenguaje_natural(pregunta_usuario: str, hist_text: str):
    # ... (No changes here, assuming it's the same as the previous version) ...
    st.info("🤔 La consulta directa falló. Activando el agente SQL experto de IANA como plan B.")
    # (Full function code from previous version)
    pass # Placeholder for brevity

def analizar_con_datos(pregunta_usuario: str, hist_text: str, df: pd.DataFrame | None, feedback: str = None): # <<< MODIFICADO: Acepta feedback
    st.info("\n🧠 Ahora, el analista experto de IANA está examinando los datos...")
    
    # Si hay feedback, se añade al prompt para la corrección
    correccion_prompt = ""
    if feedback:
        st.warning(f"⚠️ La respuesta anterior fue rechazada. Reintentando con feedback: {feedback}")
        correccion_prompt = f"""
        ---
        INSTRUCCIÓN DE CORRECCIÓN URGENTE:
        Tu respuesta anterior fue considerada incorrecta o incompleta.
        Feedback del supervisor: "{feedback}"
        Por favor, genera una NUEVA respuesta que corrija este error específico y se alinee completamente con la pregunta original del usuario.
        ---
        """

    prompt_analisis = f"""
    {correccion_prompt}
    Eres IANA, analista de datos senior en Ventus. Tu tarea es realizar un análisis ejecutivo rápido...
    ... (el resto del prompt de análisis sigue igual) ...
    Pregunta Original del Usuario: {pregunta_usuario}
    {hist_text}
    Datos para tu análisis (preview de las primeras 30 filas):
    {_df_preview(df, 30)}
    ... (el resto de las instrucciones de análisis y formato) ...
    """
    with st.spinner("💡 Generando análisis y recomendaciones avanzadas..."):
        analisis = llm_analista.invoke(prompt_analisis).content
    st.success("💡 ¡Análisis completado!")
    return analisis


def responder_conversacion(pregunta_usuario: str, hist_text: str, feedback: str = None): # <<< MODIFICADO: Acepta feedback
    st.info("💬 Activando modo de conversación...")
    
    correccion_prompt = ""
    if feedback:
        st.warning(f"⚠️ La respuesta anterior fue rechazada. Reintentando con feedback: {feedback}")
        correccion_prompt = f"""
        ---
        INSTRUCCIÓN DE CORRECCIÓN URGENTE:
        Tu respuesta anterior no fue adecuada.
        Feedback del supervisor: "{feedback}"
        Por favor, genera una NUEVA respuesta que siga este feedback y responda mejor a la pregunta del usuario.
        ---
        """

    prompt_personalidad = f"""
    {correccion_prompt}
    Tu nombre es IANA, una asistente de IA de Ventus...
    ... (el resto del prompt de personalidad sigue igual) ...
    {hist_text}
    Pregunta del usuario: "{pregunta_usuario}"
    """
    respuesta = llm_analista.invoke(prompt_personalidad).content
    return {"texto": respuesta, "df": None, "analisis": None}


# ============================================
# <<< NUEVO >>> AGENTE VALIDADOR CON FEEDBACK
# ============================================
def validar_y_corregir_respuesta(pregunta_usuario: str, respuesta_iana: dict, hist_text: str) -> dict:
    """
    Valida la respuesta de IANA. Si es correcta, la aprueba. Si no, genera feedback para corregirla.
    """
    st.info("🕵️‍♀️ Supervisor de Calidad IANA: Verificando la respuesta...")

    contenido_respuesta = ""
    if respuesta_iana.get("texto"): contenido_respuesta += respuesta_iana["texto"]
    if respuesta_iana.get("df") is not None and not respuesta_iana["df"].empty:
        contenido_respuesta += "\n[TABLA DE DATOS CON " + str(len(respuesta_iana["df"])) + " FILAS]"
    if respuesta_iana.get("analisis"): contenido_respuesta += "\n" + respuesta_iana["analisis"]

    if not contenido_respuesta.strip():
        return {"aprobado": False, "feedback": "La respuesta generada está completamente vacía."}

    prompt_validacion = f"""
    Eres un supervisor de calidad de IA, muy estricto y lógico. Tu trabajo es evaluar si la respuesta de un asistente de IA (IANA) es coherente, relevante y correcta con respecto a la pregunta de un usuario y el contexto de la conversación.

    FORMATO DE SALIDA OBLIGATORIO:
    - Si la respuesta es buena, relevante y precisa, responde SOLAMENTE con la palabra: APROBADO
    - Si la respuesta es incorrecta, irrelevante, una alucinación, o no responde a la pregunta, responde con la palabra RECHAZADO, seguida de dos puntos y una razón corta y accionable para la corrección.

    EJEMPLOS:
    - RECHAZADO: El análisis no menciona al proveedor 'ACME', que fue explícitamente solicitado.
    - RECHAZADO: La tabla de datos está vacía, pero debería haber resultados para esa consulta.
    - RECHAZADO: Es una respuesta genérica que no utiliza los datos proporcionados para responder la pregunta específica.
    - APROBADO

    ---
    Contexto de Conversación:
    {hist_text}
    ---
    Pregunta del Usuario: "{pregunta_usuario}"
    ---
    Respuesta Generada por IANA para Evaluar:
    "{contenido_respuesta}"
    ---

    Evaluación:
    """
    
    try:
        resultado_validacion = llm_validador.invoke(prompt_validacion).content.strip()
        
        if resultado_validacion.upper().startswith("APROBADO"):
            st.success("✅ Respuesta aprobada por el Supervisor de Calidad.")
            return {"aprobado": True, "feedback": None}
        elif resultado_validacion.upper().startswith("RECHAZADO"):
            feedback = resultado_validacion.split(":", 1)[1].strip() if ":" in resultado_validacion else "Razón no especificada."
            st.warning(f"❌ Respuesta rechazada. Feedback: {feedback}")
            return {"aprobado": False, "feedback": feedback}
        else:
            # Si el LLM no sigue el formato, lo rechazamos por seguridad
            st.warning("⚠️ El validador dio una respuesta ambigua. Rechazando por precaución.")
            return {"aprobado": False, "feedback": "La validación produjo una respuesta ambigua."}

    except Exception as e:
        st.error(f"Error en el agente validador: {e}")
        return {"aprobado": False, "feedback": f"Excepción durante la validación: {e}"}

# --- Orquestador Principal (MODIFICADO CON BUCLE DE CORRECCIÓN) ---

def orquestador(pregunta_usuario: str, chat_history: list):
    MAX_INTENTOS = 2 # Intento inicial + 1 reintento
    respuesta_final = None
    feedback_previo = None

    with st.expander("⚙️ Ver Proceso de IANA", expanded=False):
        for intento in range(MAX_INTENTOS):
            st.info(f"🚀 **Intento {intento + 1} de {MAX_INTENTOS}**")
            
            if intento == 0:
                # --- PRIMER INTENTO ---
                with st.spinner("🔍 IANA está analizando tu pregunta..."):
                    clasificacion = clasificar_intencion(pregunta_usuario)
                st.success(f"✅ ¡Intención detectada! Tarea: {clasificacion.upper()}.")
                hist_text = get_history_text(chat_history)
                
                if clasificacion == "conversacional":
                    res = responder_conversacion(pregunta_usuario, hist_text)
                elif clasificacion == "analista" or clasificacion == "consulta":
                    # (Lógica para obtener datos y analizar, simplificada para claridad)
                    res_datos = obtener_datos_sql(pregunta_usuario, hist_text)
                    res = {"tipo": clasificacion, **res_datos, "analisis": None}
                    if clasificacion == "analista" and res.get("df") is not None and not res["df"].empty:
                        res["analisis"] = analizar_con_datos(pregunta_usuario, hist_text, res["df"])
            else:
                # --- REINTENTO CON FEEDBACK ---
                st.info(f"🔄 Regenerando respuesta con base en el feedback...")
                if clasificacion == "conversacional":
                    res = responder_conversacion(pregunta_usuario, hist_text, feedback=feedback_previo)
                elif clasificacion == "analista":
                    # Reutilizamos los datos obtenidos, pero regeneramos el análisis
                    analisis_corregido = analizar_con_datos(pregunta_usuario, hist_text, res["df"], feedback=feedback_previo)
                    res["analisis"] = analisis_corregido
                # (Podríamos añadir lógica para re-ejecutar SQL si el feedback fue sobre datos incorrectos)
            
            # --- VALIDACIÓN EN CADA INTENTO ---
            resultado_validacion = validar_y_corregir_respuesta(pregunta_usuario, res, hist_text)
            
            if resultado_validacion["aprobado"]:
                respuesta_final = res
                break # Salimos del bucle si la respuesta es aprobada
            else:
                feedback_previo = resultado_validacion["feedback"]
                if intento == MAX_INTENTOS - 1: # Si es el último intento y falló
                    st.error("❗ Lo siento, he intentado corregir mi respuesta pero sigo sin poder darte un resultado preciso. ¿Podrías reformular tu pregunta?")
                    respuesta_final = {"tipo": "error", "texto": "Lo siento, he intentado corregir mi respuesta pero sigo sin poder darte un resultado preciso. ¿Podrías reformular tu pregunta?", "df": None, "analisis": None}

    return respuesta_final


# ... (El resto del código de la interfaz de Streamlit sigue igual) ...

# (Las funciones clasificar_intencion y obtener_datos_sql no necesitan cambios)
def clasificar_intencion(pregunta: str) -> str:
    # ... (No changes here) ...
    pass # Placeholder for brevity

def obtener_datos_sql(pregunta_usuario: str, hist_text: str) -> dict:
    # ... (No changes here) ...
    pass # Placeholder for brevity

# ============================================
# 3) Interfaz de Chat de Streamlit
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": {"texto": "¡Hola! Soy IANA, tu asistente de IA de Ventus. Estoy lista para analizar los datos de tus proyectos. ¿Qué te gustaría saber?"}}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message.get("content", {})
        if isinstance(content, dict):
            if "texto" in content and content["texto"]: st.markdown(content["texto"])
            if "df" in content and content["df"] is not None: st.dataframe(content["df"])
            if "analisis" in content and content["analisis"]: st.markdown(content["analisis"])
        elif isinstance(content, str):
             st.markdown(content)


if prompt := st.chat_input("Pregunta por costos, proveedores, familia..."):
    if not all([db, llm_sql, llm_analista, llm_orq, agente_sql, llm_validador]):
        st.error("La aplicación no está completamente inicializada. Revisa los errores de conexión o de API key.")
    else:
        st.session_state.messages.append({"role": "user", "content": {"texto": prompt}})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            res = orquestador(prompt, st.session_state.messages)
            
            if res.get("tipo") != "error":
                st.markdown(f"### IANA responde a: '{prompt}'")
                if res.get("df") is not None and not res["df"].empty:
                    st.dataframe(res["df"])
                if res.get("texto"):
                    st.markdown(res["texto"])
                if res.get("analisis"):
                    st.markdown("---")
                    st.markdown("### 🧠 Análisis de IANA para Ventus") 
                    st.markdown(res["analisis"])
            else:
                st.markdown(res["texto"])
                
            st.session_state.messages.append({"role": "assistant", "content": res})
