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

col1, col2 = st.columns([1, 4]) 

with col1:
    st.image("logo_ventus.png", width=120) 

with col2:
    st.title("IANA: Tu Asistente IA para Análisis de Datos")
    st.markdown("Soy la red de agentes IA de **VENTUS**. Hazme una pregunta sobre los datos del proyecto IGUANA.")

# ============================================
# 1) Conexión a la Base de Datos y LLMs
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
# Funciones Auxiliares
# ============================================
def get_history_text(chat_history: list, n_turns=3) -> str:
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
    return "\n--- Contexto de Conversación Anterior ---\n" + "\n".join(history_text) + "\n--- Fin del Contexto ---\n"

def markdown_table_to_df(texto: str) -> pd.DataFrame:
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
    if df is None or df.empty: return ""
    try: return df.head(n).to_markdown(index=False)
    except Exception: return df.head(n).to_string(index=False)

def interpretar_resultado_sql(res: dict) -> dict:
    df = res.get("df")
    if df is not None and not df.empty and res.get("texto") is None:
        if df.shape == (1, 1):
            valor = df.iloc[0, 0]
            nombre_columna = df.columns[0]
            res["texto"] = f"La respuesta para '{nombre_columna}' es: **{valor}**"
            st.info("💡 Resultado numérico interpretado para una respuesta directa.")
    return res

# ============================================
# Funciones de Agentes
# ============================================
def ejecutar_sql_real(pregunta_usuario: str, hist_text: str):
    st.info("🤖 El agente de datos está traduciendo tu pregunta a SQL...")
    # Limpiamos una posible mala generación del LLM
    prompt_con_instrucciones = f"""
    Tu tarea es generar una consulta SQL limpia para responder la pregunta del usuario.
    {hist_text}
    Pregunta del usuario: "{pregunta_usuario}"
    """
    try:
        # Usamos un LLM específico para generar solo la consulta
        query_chain = create_sql_query_chain(llm_sql, db)
        sql_query = query_chain.invoke({"question": prompt_con_instrucciones})
        
        # Limpieza robusta
        sql_query_limpia = re.sub(r"^\s*```sql\s*|\s*SQLQuery:\s*|\s*```\s*$", "", sql_query, flags=re.IGNORECASE).strip()
        sql_query_limpia = re.sub(r'LIMIT\s+\d+\s*;?$', '', sql_query_limpia, flags=re.IGNORECASE | re.DOTALL).strip()

        st.code(sql_query_limpia, language='sql')
        with st.spinner("⏳ Ejecutando consulta..."):
            with db._engine.connect() as conn:
                df = pd.read_sql(text(sql_query_limpia), conn)
        st.success("✅ ¡Consulta ejecutada!")
        return {"sql": sql_query_limpia, "df": df}
    except Exception as e:
        st.warning(f"❌ Error en la consulta directa. Intentando método alternativo... Error: {e}")
        return {"sql": None, "df": None, "error": str(e)}

def ejecutar_sql_en_lenguaje_natural(pregunta_usuario: str, hist_text: str):
    st.info("🤔 Activando el agente SQL experto como plan B.")
    prompt_sql = (
        "Tu tarea es responder la pregunta del usuario consultando la tabla 'ventus'. "
        f"{hist_text}"
        "Devuelve ÚNICAMENTE una tabla de datos en formato Markdown. "
        "NUNCA resumas ni expliques los resultados. "
        "El SQL que generes internamente NO DEBE CONTENER 'LIMIT'. "
        f"Pregunta del usuario: {pregunta_usuario}"
    )
    try:
        with st.spinner("💬 Pidiendo al agente SQL que responda..."):
            res = agente_sql.invoke(prompt_sql)
            texto = res["output"] if isinstance(res, dict) and "output" in res else str(res)
        st.info("📝 Intentando convertir la respuesta en una tabla de datos...")
        df_md = markdown_table_to_df(texto)
        return {"texto": texto, "df": df_md}
    except Exception as e:
        st.error(f"❌ El agente SQL experto también encontró un problema: {e}")
        return {"texto": f"[SQL_ERROR] {e}", "df": pd.DataFrame()}

def analizar_con_datos(pregunta_usuario: str, hist_text: str, df: pd.DataFrame | None, feedback: str = None):
    st.info("\n🧠 El analista experto está examinando los datos...")
    correccion_prompt = ""
    if feedback:
        st.warning(f"⚠️ Reintentando con feedback: {feedback}")
        correccion_prompt = f'INSTRUCCIÓN DE CORRECCIÓN: Tu respuesta anterior fue incorrecta. Feedback: "{feedback}". Genera una NUEVA respuesta corrigiendo este error.'

    prompt_analisis = f"""
    {correccion_prompt}
    Eres IANA, analista de datos senior. Tu tarea es realizar un análisis ejecutivo rápido sobre los datos proporcionados.
    Pregunta Original: {pregunta_usuario}
    {hist_text}
    Datos:
    {_df_preview(df, 30)}
    ---
    FORMATO DE ENTREGA OBLIGATORIO:
    📌 Resumen Ejecutivo:
    - (Hallazgos principales con números clave.)
    🔍 Números de referencia:
    - (Total General, Promedio, etc.)
    """
    with st.spinner("💡 Generando análisis avanzado..."):
        analisis = llm_analista.invoke(prompt_analisis).content
    st.success("💡 ¡Análisis completado!")
    return analisis

def responder_conversacion(pregunta_usuario: str, hist_text: str, feedback: str = None):
    st.info("💬 Activando modo de conversación...")
    correccion_prompt = ""
    if feedback:
        st.warning(f"⚠️ Reintentando con feedback: {feedback}")
        correccion_prompt = f'INSTRUCCIÓN DE CORRECCIÓN: Tu respuesta anterior no fue adecuada. Feedback: "{feedback}". Genera una NUEVA respuesta.'

    prompt_personalidad = f"""
    {correccion_prompt}
    Tu nombre es IANA, una IA amable y profesional de Ventus. Ayuda a analizar datos.
    Si el usuario hace un comentario casual, responde amablemente y redirígelo a tus capacidades.
    {hist_text}
    Pregunta: "{pregunta_usuario}"
    """
    respuesta = llm_analista.invoke(prompt_personalidad).content
    return {"texto": respuesta, "df": None, "analisis": None}

# ============================================
# Lógica Principal y Orquestador
# ============================================
def validar_y_corregir_respuesta(pregunta_usuario: str, respuesta_iana: dict, hist_text: str) -> dict:
    st.info("🕵️‍♀️ Supervisor de Calidad: Verificando la respuesta...")
    contenido_respuesta = ""
    if respuesta_iana.get("texto"): contenido_respuesta += respuesta_iana["texto"]
    if respuesta_iana.get("df") is not None and not respuesta_iana["df"].empty:
        contenido_respuesta += "\n[TABLA DE DATOS CON " + str(len(respuesta_iana["df"])) + " FILAS]"
    if respuesta_iana.get("analisis"): contenido_respuesta += "\n" + respuesta_iana["analisis"]
    if not contenido_respuesta.strip():
        return {"aprobado": False, "feedback": "La respuesta generada está vacía."}

    prompt_validacion = f"""
    Eres un supervisor de calidad de IA estricto. Evalúa si la respuesta es coherente y relevante.
    FORMATO OBLIGATORIO:
    - Si es buena, responde: APROBADO
    - Si es incorrecta, responde: RECHAZADO: [razón corta y accionable]
    ---
    Contexto: {hist_text}
    Pregunta: "{pregunta_usuario}"
    Respuesta a Evaluar: "{contenido_respuesta}"
    ---
    Evaluación:
    """
    try:
        resultado_validacion = llm_validador.invoke(prompt_validacion).content.strip()
        if resultado_validacion.upper().startswith("APROBADO"):
            st.success("✅ Respuesta aprobada por el Supervisor.")
            return {"aprobado": True, "feedback": None}
        elif resultado_validacion.upper().startswith("RECHAZADO"):
            feedback = resultado_validacion.split(":", 1)[1].strip() if ":" in resultado_validacion else "Razón no especificada."
            st.warning(f"❌ Respuesta rechazada. Feedback: {feedback}")
            return {"aprobado": False, "feedback": feedback}
        else:
            return {"aprobado": False, "feedback": "Respuesta ambigua del validador."}
    except Exception as e:
        return {"aprobado": False, "feedback": f"Excepción durante la validación: {e}"}

def clasificar_intencion(pregunta: str) -> str:
    # <<< PROMPT MEJORADO PARA MAYOR PRECISIÓN >>>
    prompt_orq = f"""
    Tu tarea es clasificar la intención del usuario en UNA de tres categorías. Responde con UNA SOLA PALABRA.

    1.  `consulta`: Si el usuario pide datos crudos.
        Ejemplos: 'dime cuántos...', 'lista todos los...', 'muéstrame el total de...', 'cuáles son los proveedores'.
    
    2.  `analista`: Si el usuario pide una interpretación, resumen, comparación o insight sobre los datos.
        Ejemplos: 'analiza los costos', 'compara los proveedores', 'cuál es la tendencia', 'dame un resumen de gastos', 'por qué subió el costo'.

    3.  `conversacional`: Si es un saludo, una pregunta sobre tus capacidades o no está relacionada con datos.
        Ejemplos: 'hola', 'qué puedes hacer', 'gracias'.

    Pregunta del usuario: "{pregunta}"
    """
    try:
        opciones_validas = ["consulta", "analista", "conversacional"]
        respuesta_llm = llm_orq.invoke(prompt_orq).content.strip().lower().replace('"', '').replace("'", "")
        
        if respuesta_llm in opciones_validas:
            return respuesta_llm
        else:
            st.warning("⚠️ Intención no clara. Se asume 'conversacional'.")
            return "conversacional"
    except Exception as e:
        st.error(f"Error al clasificar intención: {e}. Se usará 'conversacional'.")
        return "conversacional"

def obtener_datos_sql(pregunta_usuario: str, hist_text: str) -> dict:
    res_real = ejecutar_sql_real(pregunta_usuario, hist_text)
    if res_real.get("df") is not None and not res_real["df"].empty:
        return {"sql": res_real["sql"], "df": res_real["df"], "texto": None}
    res_nat = ejecutar_sql_en_lenguaje_natural(pregunta_usuario, hist_text)
    return {"sql": None, "df": res_nat["df"], "texto": res_nat["texto"]}

def orquestador(pregunta_usuario: str, chat_history: list):
    MAX_INTENTOS = 2
    respuesta_final = None
    feedback_previo = None

    with st.expander("⚙️ Ver Proceso de IANA", expanded=False):
        hist_text = get_history_text(chat_history)
        clasificacion = clasificar_intencion(pregunta_usuario)
        st.success(f"✅ ¡Intención detectada! Tarea: {clasificacion.upper()}.")
        
        res = {"tipo": clasificacion, "df": None, "texto": None, "analisis": None}

        for intento in range(MAX_INTENTOS):
            st.info(f"🚀 **Intento {intento + 1} de {MAX_INTENTOS}**")
            
            if intento == 0:
                if clasificacion == "conversacional":
                    res = responder_conversacion(pregunta_usuario, hist_text)
                else: 
                    res_datos = obtener_datos_sql(pregunta_usuario, hist_text)
                    res.update(res_datos)
                    res = interpretar_resultado_sql(res)
                    if clasificacion == "analista" and res.get("df") is not None and not res["df"].empty:
                        res["analisis"] = analizar_con_datos(pregunta_usuario, hist_text, res["df"])
            else: 
                st.info(f"🔄 Regenerando respuesta con base en el feedback: '{feedback_previo}'")
                if clasificacion == "conversacional":
                    res = responder_conversacion(pregunta_usuario, hist_text, feedback=feedback_previo)
                elif clasificacion == "consulta":
                    res_datos = obtener_datos_sql(pregunta_usuario, hist_text)
                    res.update(res_datos)
                    res = interpretar_resultado_sql(res)
                elif clasificacion == "analista":
                    res["analisis"] = analizar_con_datos(pregunta_usuario, hist_text, res.get("df"), feedback=feedback_previo)

            resultado_validacion = validar_y_corregir_respuesta(pregunta_usuario, res, hist_text)
            
            if resultado_validacion["aprobado"]:
                respuesta_final = res
                break 
            else:
                feedback_previo = resultado_validacion["feedback"]
                if intento == MAX_INTENTOS - 1:
                    respuesta_final = {"tipo": "error", "texto": "Lo siento, mi respuesta no fue satisfactoria incluso después de una corrección. Por favor, intenta reformular tu pregunta."}

    return respuesta_final

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
            if "df" in content and content["df"] is not None and not content["df"].empty: st.dataframe(content["df"])
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
            
            st.session_state.messages.append({"role": "assistant", "content": res})

            if res.get("tipo") != "error":
                if res.get("texto"):
                    st.markdown(res["texto"])
                if res.get("df") is not None and not res["df"].empty:
                    st.dataframe(res["df"])
                if res.get("analisis"):
                    st.markdown("---")
                    st.markdown("### 🧠 Análisis de IANA") 
                    st.markdown(res["analisis"])
            else:
                st.error(res.get("texto", "Ocurrió un error inesperado."))
