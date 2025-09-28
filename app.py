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
    st.image("logo_.png", width=120)

with col2:
    st.title("IANA: Tu Asistente IA para Análisis de Datos")
    st.markdown("Soy la red de agentes IA de ****. Hazme una pregunta sobre los datos del proyecto IGUANA.")

# ============================================
# 1) Conexión a la Base de Datos y LLMs
# ============================================

@st.cache_resource
def get_database_connection():
    with st.spinner("🔌 Conectando a la base de datos de Ventus..."):
        try:
            db_user = st.secrets["db_credentials"]["user"]
            db_pass = st.secrets["db_credentials"]["password"]
            db_host = st.secrets["db_credentials"]["host"]
            db_name = st.secrets["db_credentials"]["database"]
            uri = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}/{db_name}"
            engine_args = {"pool_recycle": 3600, "pool_pre_ping": True}
            db = SQLDatabase.from_uri(uri, include_tables=["ventus_bi"], engine_args=engine_args)
            st.success("✅ Conexión a la base de datos establecida.")
            return db
        except Exception as e:
            st.error(f"Error al conectar a la base de datos: {e}")
            return None

@st.cache_resource
def get_llms():
    with st.spinner("🧠 Inicializando la red de agentes IANA..."):
        try:
            api_key = st.secrets["google_api_key"]
            
            model_name = "gemini-1.5-pro"
            
            llm_sql = ChatGoogleGenerativeAI(model=model_name, temperature=0.1, google_api_key=api_key)
            llm_analista = ChatGoogleGenerativeAI(model=model_name, temperature=0.1, google_api_key=api_key)
            llm_orq = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, google_api_key=api_key)
            llm_validador = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, google_api_key=api_key)
            
            st.success("✅ Agentes de IANA listos.")
            return llm_sql, llm_analista, llm_orq, llm_validador
        except Exception as e:
            st.error(f"Error al inicializar los LLMs. Asegúrate de que tu API key es correcta. Error: {e}")
            return None, None, None, None

db = get_database_connection()
llm_sql, llm_analista, llm_orq, llm_validador = get_llms()

@st.cache_resource
def get_sql_agent(_llm, _db):
    if not _llm or not _db: return None
    with st.spinner("🛠️ Configurando agente SQL de IANA..."):
        toolkit = SQLDatabaseToolkit(db=_db, llm=_llm)
        agent = create_sql_agent(llm=_llm, toolkit=toolkit, verbose=False, top_k=1000)
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

def _df_preview(df: pd.DataFrame, n: int = 5) -> str:
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

    # =============================================================================
    # NUEVA SECCIÓN: CONTEXTO DE CÁLCULOS PARA EL AGENTE
    # Aquí le "enseñamos" al agente las fórmulas de negocio.
    # Puedes añadir todas las que necesites.
    # =============================================================================
    contexto_calculos = """
    ---
    <<< DEFINICIONES DE CÁLCULOS Y LÓGICA DE NEGOCIO >>>
    1.  **Monto del IVA en COP**: Si el usuario pregunta por el monto del IVA, este se calcula como la diferencia entre el total y el subtotal: `Total_COP - Subtotal_COP`. La tabla también tiene una columna `IVA_COP` que debería ser lo mismo.
    2.  **Porcentaje de IVA**: Para calcular la proporción o el porcentaje que representa el IVA, usa la fórmula: `((Total_COP - Subtotal_COP) / Subtotal_COP) * 100`.
    3.  **Precio Unitario en COP**: Si el usuario pregunta por el costo o precio por unidad, calcula `Total_COP / Cantidad`. Asegúrate de no dividir por cero.
    4.  **Precio Unitario en USD**: Para el precio por unidad en dólares, calcula `Total_USD / Cantidad`.

    <<< EJEMPLO DE USO >>>
    - Pregunta del usuario: "cuál es el porcentaje de iva promedio para el proveedor 'ACME'?"
    - SQL Esperado: SELECT AVG(((Total_COP - Subtotal_COP) / Subtotal_COP) * 100) FROM ventus_bi WHERE Proveedor LIKE '%ACME%';
    ---
    """

    # <-- CAMBIO: Se añade el 'contexto_calculos' al prompt principal.
    prompt_con_instrucciones = f"""
    Tu tarea es generar una consulta SQL limpia para responder la pregunta del usuario, aplicando la lógica de negocio si es necesario.
    {contexto_calculos}
    ---
    <<< REGLA DE ORO PARA BÚSQUEDA DE PRODUCTOS >>>
    1. La columna `Producto` contiene descripciones largas.
    2. Si el usuario pregunta por un producto o servicio específico (ej: 'transporte', 'guantes'), SIEMPRE debes usar `LIKE '%%'` para buscar dentro de la columna `Producto`.
    3. EJEMPLO: Si el usuario pregunta "cuántos transportes...", debes generar `WHERE Producto LIKE '%transporte%'`.
    ---
    {hist_text}
    Pregunta del usuario: "{pregunta_usuario}"
    """
    try:
        query_chain = create_sql_query_chain(llm_sql, db)
        sql_query_bruta = query_chain.invoke({"question": prompt_con_instrucciones})
        select_pos = sql_query_bruta.upper().rfind("SELECT")
        sql_query_limpia = sql_query_bruta[select_pos:] if select_pos != -1 else sql_query_bruta
        sql_query_limpia = re.sub(r"^\s*```sql\s*|\s*```\s*$", "", sql_query_limpia, flags=re.IGNORECASE).strip()
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
    # <-- CAMBIO MENOR: Asegurarse que el plan B también apunte a la tabla correcta.
    prompt_sql = (f"Tu tarea es responder la pregunta del usuario consultando la tabla 'ventus_bi'.\n{hist_text}\nDevuelve ÚNICAMENTE una tabla en formato Markdown. NUNCA resumas. El SQL interno NO DEBE CONTENER 'LIMIT'.\nPregunta: {pregunta_usuario}")
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
    
    prompt_analisis = f"""{correccion_prompt}
    Eres IANA, un analista de datos senior EXTREMADAMENTE PRECISO y riguroso.
    ---
    <<< REGLAS CRÍTICAS DE PRECISIÓN >>>
    1.  **NO ALUCINAR**: NUNCA inventes números, totales, porcentajes o nombres de productos/categorías que no estén EXPRESAMENTE en la tabla de 'Datos'. Tu respuesta debe ser 100% verificable con los datos proporcionados.
    2.  **MANEJO DE DATOS INCOMPLETOS (SPARSE DATA)**: Es normal que los datos no contengan entradas para todos los meses o categorías. Tu tarea es reportar sobre los datos que SÍ existen. Es un hallazgo importante señalar los vacíos. EJEMPLO: "No se registraron datos para el mes de Marzo". NUNCA inventes datos para rellenar vacíos.
    3.  **VERIFICAR CÁLCULOS**: Antes de escribir un número, verifica dos veces el cálculo (SUMA, CONTEO, PROMEDIO) directamente de la tabla de 'Datos'.
    4.  **CITAR DATOS**: Basa CADA afirmación que hagas en los datos visibles en la tabla. No hagas suposiciones.
    ---
    Pregunta Original: {pregunta_usuario}\n{hist_text}
    Datos para tu análisis (usa SÓLO estos datos):
    {_df_preview(df, 50)} 
    ---
    FORMATO OBLIGATORIO:
    📌 Resumen Ejecutivo:\n- (Hallazgos principales basados ESTRICTAMENTE en los datos.)
    🔍 Números de referencia:\n- (Cifras clave calculadas DIRECTAMENTE de los datos.)"""
    with st.spinner("💡 Generando análisis avanzado..."):
        analisis = llm_analista.invoke(prompt_analisis).content
    st.success("💡 ¡Análisis completado!")
    return analisis

def responder_conversacion(pregunta_usuario: str, hist_text: str):
    st.info("💬 Activando modo de conversación...")
    prompt_personalidad = f"""
    Tu nombre es IANA, una IA amable de Ventus. Ayuda a analizar datos.
    Si el usuario hace un comentario casual, responde amablemente y redirígelo a tus capacidades.
    {hist_text}\nPregunta: "{pregunta_usuario}" """
    respuesta = llm_analista.invoke(prompt_personalidad).content
    return {"texto": respuesta, "df": None, "analisis": None}

# ============================================
# Lógica Principal y Orquestador
# ============================================
def validar_y_corregir_respuesta_analista(pregunta_usuario: str, res_analisis: dict, hist_text: str) -> dict:
    MAX_INTENTOS = 2
    for intento in range(MAX_INTENTOS):
        st.info(f"🕵️‍♀️ Supervisor de Calidad: Verificando análisis (Intento {intento + 1})...")
        
        contenido_respuesta = res_analisis.get("analisis", "")
        if not contenido_respuesta.strip():
            return {"tipo": "error", "texto": "El análisis generado estaba vacío."}

        df_preview = _df_preview(res_analisis.get("df"), 50)

        prompt_validacion = f"""
        Eres un supervisor de calidad estricto. Tu tarea es validar si un 'Análisis' es coherente y se basa ESTRICTAMENTE en los 'Datos de Soporte' proporcionados.
        FORMATO OBLIGATORIO:
        - Si el análisis se basa 100% en los datos, responde: APROBADO
        - Si el análisis alucina, inventa datos o no es relevante, responde: RECHAZADO: [razón corta y accionable]
        ---
        Pregunta Original del Usuario: "{pregunta_usuario}"
        ---
        Datos de Soporte (la tabla que la IA usó para el análisis):
        {df_preview}
        ---
        Análisis a Evaluar:
        "{contenido_respuesta}"
        ---
        Evaluación: ¿El análisis se basa fielmente en los Datos de Soporte?
        """
        try:
            resultado_validacion = llm_validador.invoke(prompt_validacion).content.strip()
            if resultado_validacion.upper().startswith("APROBADO"):
                st.success("✅ Análisis aprobado por el Supervisor.")
                return res_analisis 
            elif resultado_validacion.upper().startswith("RECHAZADO"):
                feedback_previo = resultado_validacion.split(":", 1)[1].strip() if ":" in resultado_validacion else "Razón no especificada."
                st.warning(f"❌ Análisis rechazado. Feedback: {feedback_previo}")
                if intento < MAX_INTENTOS - 1:
                    st.info("🔄 Regenerando análisis con feedback...")
                    res_analisis["analisis"] = analizar_con_datos(pregunta_usuario, hist_text, res_analisis.get("df"), feedback=feedback_previo)
                else:
                    return {"tipo": "error", "texto": "El análisis no fue satisfactorio incluso después de una corrección."}
            else:
                return {"tipo": "error", "texto": "Respuesta ambigua del validador."}
        except Exception as e:
            return {"tipo": "error", "texto": f"Excepción durante la validación: {e}"}
    return {"tipo": "error", "texto": "Se alcanzó el límite de intentos de validación."}

def clasificar_intencion(pregunta: str) -> str:
    prompt_orq = f"""
    Tu tarea es clasificar la intención del usuario. Presta especial atención a los verbos de acción y palabras clave. Responde con UNA SOLA PALABRA.

    1. `analista`: Si la pregunta pide explícitamente una interpretación, resumen, comparación o explicación.
       PALABRAS CLAVE PRIORITARIAS: analiza, compara, resume, explica, por qué, tendencia, insights, dame un análisis, haz un resumen.
       Si una de estas palabras clave está presente, la intención SIEMPRE es `analista`.

    2. `consulta`: Si la pregunta pide datos crudos (listas, conteos, totales) y NO contiene una palabra clave prioritaria de `analista`.
       Ejemplos: 'cuántos proveedores hay', 'lista todos los productos', 'muéstrame el total', 'y ahora por mes'.

    3. `conversacional`: Si es un saludo o una pregunta general no relacionada con datos.
       Ejemplos: 'hola', 'gracias', 'qué puedes hacer'.

    Pregunta del usuario: "{pregunta}"
    Clasificación:
    """
    try:
        opciones_validas = ["consulta", "analista", "conversacional"]
        respuesta_llm = llm_orq.invoke(prompt_orq).content.strip().lower().replace('"', '').replace("'", "")
        if respuesta_llm in opciones_validas: return respuesta_llm
        return "conversacional"
    except Exception:
        return "conversacional"

def obtener_datos_sql(pregunta_usuario: str, hist_text: str) -> dict:
    if any(keyword in pregunta_usuario.lower() for keyword in ["anterior", "esos datos", "esa tabla"]):
        for msg in reversed(st.session_state.get('messages', [])):
            content = msg.get('content', {})
            if msg['role'] == 'assistant' and isinstance(content, dict) and content.get('df') is not None:
                st.info("💡 Usando datos de la respuesta anterior para la nueva solicitud.")
                return {"df": content['df']}

    res_real = ejecutar_sql_real(pregunta_usuario, hist_text)
    if res_real.get("df") is not None and not res_real["df"].empty:
        return res_real
    return ejecutar_sql_en_lenguaje_natural(pregunta_usuario, hist_text)

def orquestador(pregunta_usuario: str, chat_history: list):
    with st.expander("⚙️ Ver Proceso de IANA", expanded=False):
        hist_text = get_history_text(chat_history)
        clasificacion = clasificar_intencion(pregunta_usuario)
        st.success(f"✅ ¡Intención detectada! Tarea: {clasificacion.upper()}.")
        
        if clasificacion == "conversacional":
            return responder_conversacion(pregunta_usuario, hist_text)

        res_datos = obtener_datos_sql(pregunta_usuario, hist_text)
        
        if res_datos.get("df") is None or res_datos["df"].empty:
            return {"tipo": "error", "texto": "Lo siento, no pude obtener datos para tu pregunta. Intenta reformularla."}

        if clasificacion == "consulta":
            st.success("✅ Consulta directa completada.")
            return interpretar_resultado_sql(res_datos)
        
        if clasificacion == "analista":
            st.info("🧠 Generando análisis inicial...")
            res_datos["analisis"] = analizar_con_datos(pregunta_usuario, hist_text, res_datos.get("df"))
            return validar_y_corregir_respuesta_analista(pregunta_usuario, res_datos, hist_text)

# ============================================
# 3) Interfaz de Chat de Streamlit
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": {"texto": "¡Hola! Soy IANA, tu asistente de IA de Ventus. ¿Qué te gustaría saber?"}}]

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

            if res and res.get("tipo") != "error":
                if res.get("texto"): st.markdown(res["texto"])
                if res.get("df") is not None and not res["df"].empty: st.dataframe(res["df"])
                if res.get("analisis"):
                    st.markdown("---")
                    st.markdown("### 🧠 Análisis de IANA") 
                    st.markdown(res["analisis"])
            elif res:
                st.error(res.get("texto", "Ocurrió un error inesperado."))



