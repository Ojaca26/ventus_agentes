# app.py

import streamlit as st
import pandas as pd
import re
import tempfile
from typing import Optional

from sqlalchemy import text

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import create_sql_query_chain

# Transcripción
from faster_whisper import WhisperModel

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
    with st.spinner("🔌 Conectando a la base de datos de Ventus..."):
        try:
            creds = st.secrets["db_credentials"]
            db_user = creds["user"]
            db_pass = creds["password"]
            db_host = creds["host"]
            db_name = creds["database"]
            uri = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}/{db_name}"
            engine_args = {
                "pool_recycle": 3600,
                "pool_pre_ping": True,
                "pool_size": 5,
                "max_overflow": 10,
            }
            db = SQLDatabase.from_uri(uri, include_tables=["ventus"], engine_args=engine_args)
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
            # Importante: sin el prefijo "models/"
            common = dict(temperature=0.1, google_api_key=api_key)
            llm_sql        = ChatGoogleGenerativeAI(model="gemini-1.5-pro", **common)
            llm_analista   = ChatGoogleGenerativeAI(model="gemini-1.5-pro", **common)
            llm_orq        = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0, google_api_key=api_key)
            llm_validador  = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0, google_api_key=api_key)
            st.success("✅ Agentes de IANA listos.")
            return llm_sql, llm_analista, llm_orq, llm_validador
        except Exception as e:
            st.error(f"Error al inicializar los LLMs. Revisa tu API key. Detalle: {e}")
            return None, None, None, None

db = get_database_connection()
llm_sql, llm_analista, llm_orq, llm_validador = get_llms()

@st.cache_resource
def get_sql_agent(_llm, _db):
    if not _llm or not _db:
        return None
    with st.spinner("🛠️ Configurando agente SQL de IANA..."):
        toolkit = SQLDatabaseToolkit(db=_db, llm=_llm)
        agent = create_sql_agent(llm=_llm, toolkit=toolkit, verbose=False, top_k=1000)
        st.success("✅ Agente SQL configurado.")
        return agent

agente_sql = get_sql_agent(llm_sql, db)

# ============================================
# 1.b) Recurso del transcriptor de audio
# ============================================

@st.cache_resource
def get_transcriber():
    """
    Carga el modelo de transcripción una sola vez.
    Usa st.secrets['whisper_model_size'] si existe; por defecto 'small'.
    """
    try:
        model_size = st.secrets.get("whisper_model_size", "small")  # tiny/base/small/medium/large-v3 (CPU: tiny/base/small)
        model = WhisperModel(model_size, device="cpu", compute_type="int8")  # rápido en CPU
        return model
    except Exception as e:
        st.warning(f"No se pudo inicializar el transcriptor: {e}")
        return None

# ============================================
# 2) Funciones Auxiliares
# ============================================

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    # Limpia $ , % espacios duros y suaves
    s2 = (
        s.astype(str)
         .str.replace(r'[\u00A0\s]', '', regex=True)
         .str.replace(',', '', regex=False)
         .str.replace('$', '', regex=False)
         .str.replace('%', '', regex=False)
    )
    try:
        return pd.to_numeric(s2)
    except Exception:
        return s

def get_history_text(chat_history: list, n_turns=3) -> str:
    if not chat_history or len(chat_history) <= 1:
        return ""
    history_text = []
    relevant_history = chat_history[-(n_turns * 2 + 1) : -1]
    for msg in relevant_history:
        content = msg.get("content", {})
        text_content = ""
        if isinstance(content, dict):
            text_content = content.get("texto", "")
        elif isinstance(content, str):
            text_content = content
        if text_content:
            role = "Usuario" if msg["role"] == "user" else "IANA"
            history_text.append(f"{role}: {text_content}")
    if not history_text:
        return ""
    return "\n--- Contexto de Conversación Anterior ---\n" + "\n".join(history_text) + "\n--- Fin del Contexto ---\n"

def markdown_table_to_df(texto: str) -> pd.DataFrame:
    lineas = [l.rstrip() for l in texto.splitlines() if l.strip().startswith('|')]
    if not lineas:
        return pd.DataFrame()
    # Filtra separadores tipo |---|
    lineas = [l for l in lineas if not re.match(r'^\|\s*-{2,}', l)]
    filas = [[c.strip() for c in l.strip('|').split('|')] for l in lineas]
    if len(filas) < 2:
        return pd.DataFrame()
    header, data = filas[0], filas[1:]
    max_cols = len(header)
    data = [r + ['']*(max_cols - len(r)) if len(r) < max_cols else r[:max_cols] for r in data]
    df = pd.DataFrame(data, columns=header)
    for c in df.columns:
        df[c] = _coerce_numeric_series(df[c])
    return df

def _df_preview(df: pd.DataFrame, n: int = 5) -> str:
    if df is None or df.empty:
        return ""
    try:
        return df.head(n).to_markdown(index=False)
    except Exception:
        return df.head(n).to_string(index=False)

def interpretar_resultado_sql(res: dict) -> dict:
    df = res.get("df")
    if df is not None and not df.empty and res.get("texto") is None:
        if df.shape == (1, 1):
            valor = df.iloc[0, 0]
            nombre_columna = df.columns[0]
            res["texto"] = f"La respuesta para '{nombre_columna}' es: **{valor}**"
            st.info("💡 Resultado numérico interpretado para una respuesta directa.")
    return res

def _asegurar_select_only(sql: str) -> str:
    """Permite solo SELECTs. Quita ; finales y LIMIT si existe."""
    sql_clean = sql.strip().rstrip(';')
    if not re.match(r'(?is)^\s*select\b', sql_clean):
        raise ValueError("Solo se permite ejecutar consultas SELECT.")
    sql_clean = re.sub(r'(?is)\blimit\s+\d+\s*$', '', sql_clean).strip()
    return sql_clean

# ============================================
# 2.b) Transcripción
# ============================================

def transcribir_audio(archivo_audio) -> Optional[str]:
    """
    Recibe un archivo subido por Streamlit (UploadedFile),
    lo guarda temporalmente y devuelve la transcripción (auto-idioma).
    """
    if archivo_audio is None:
        return None

    model = get_transcriber()
    if model is None:
        st.error("El transcriptor no está disponible.")
        return None

    # Guardar temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{archivo_audio.name}") as tmp:
        tmp.write(archivo_audio.read())
        ruta = tmp.name

    st.info("🎧 Transcribiendo audio...")
    try:
        segments, info = model.transcribe(
            ruta,
            language=None,           # autodetect
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        texto = "".join([seg.text for seg in segments]).strip()
        if not texto:
            st.warning("No se pudo extraer texto del audio.")
            return None

        # Mostrar idioma detectado si lo hay
        if getattr(info, "language", None):
            st.caption(f"Idioma detectado: **{info.language}** · Prob: {getattr(info, 'language_probability', 0):.2f}")

        st.success("✅ Transcripción completada.")
        return texto
    except Exception as e:
        st.error(f"Error al transcribir: {e}")
        return None

# ============================================
# 3) Agentes
# ============================================

def ejecutar_sql_real(pregunta_usuario: str, hist_text: str):
    st.info("🤖 El agente de datos está traduciendo tu pregunta a SQL...")
    prompt_con_instrucciones = f"""
    Tu tarea es generar una consulta SQL limpia (SOLO SELECT) sobre la tabla `ventus` para responder la pregunta del usuario.
    ---
    <<< REGLA DE ORO PARA BÚSQUEDA DE PRODUCTOS >>>
    1. La columna `Producto` contiene descripciones largas.
    2. Si el usuario pregunta por un producto o servicio específico (ej: 'transporte', 'guantes'), SIEMPRE usa `WHERE LOWER(Producto) LIKE '%palabra%'`.
    3. Ejemplo: "cuántos transportes..." -> `WHERE LOWER(Producto) LIKE '%transporte%'`.
    4. No agregues LIMIT.
    ---
    {hist_text}
    Pregunta del usuario: "{pregunta_usuario}"
    Devuelve SOLO la consulta SQL (sin explicaciones).
    """
    try:
        query_chain = create_sql_query_chain(llm_sql, db)
        sql_query_bruta = query_chain.invoke({"question": prompt_con_instrucciones})
        # Captura el último SELECT para evitar preámbulos
        m = re.search(r'(?is)(select\b.+)$', sql_query_bruta.strip())
        sql_query_limpia = m.group(1).strip() if m else sql_query_bruta.strip()
        # Quita fences y LIMIT y valida que sea SELECT
        sql_query_limpia = re.sub(r'(?is)^```sql|```$', '', sql_query_limpia).strip()
        sql_query_limpia = _asegurar_select_only(sql_query_limpia)
        st.code(sql_query_limpia, language='sql')

        with st.spinner("⏳ Ejecutando consulta..."):
            with db._engine.connect() as conn:
                df = pd.read_sql(text(sql_query_limpia), conn)

        st.success(f"✅ ¡Consulta ejecutada! Filas: {len(df)}")
        return {"sql": sql_query_limpia, "df": df}
    except Exception as e:
        st.warning(f"❌ Error en la consulta directa. Intentando método alternativo... Detalle: {e}")
        return {"sql": None, "df": None, "error": str(e)}

def ejecutar_sql_en_lenguaje_natural(pregunta_usuario: str, hist_text: str):
    st.info("🤔 Activando el agente SQL experto como plan B.")
    prompt_sql = (
        f"Tu tarea es responder la pregunta consultando la tabla 'ventus'.\n"
        f"{hist_text}\n"
        f"Devuelve ÚNICAMENTE una tabla en formato Markdown (con encabezados). "
        f"NUNCA resumas ni expliques. El SQL interno NO DEBE CONTENER 'LIMIT'.\n"
        f"Pregunta: {pregunta_usuario}"
    )
    try:
        with st.spinner("💬 Pidiendo al agente SQL que responda..."):
            res = agente_sql.invoke(prompt_sql)
            texto = res["output"] if isinstance(res, dict) and "output" in res else str(res)

        st.info("📝 Intentando convertir la respuesta en una tabla de datos...")
        df_md = markdown_table_to_df(texto)
        if df_md.empty:
            st.warning("La conversión de Markdown a tabla no produjo filas. Se mostrará la salida cruda.")
        return {"texto": texto, "df": df_md}
    except Exception as e:
        st.error(f"❌ El agente SQL experto también encontró un problema: {e}")
        return {"texto": f"[SQL_ERROR] {e}", "df": pd.DataFrame()}

def analizar_con_datos(pregunta_usuario: str, hist_text: str, df: pd.DataFrame | None, feedback: str = None):
    st.info("\n🧠 El analista experto está examinando los datos...")
    correccion_prompt = ""
    if feedback:
        st.warning(f"⚠️ Reintentando con feedback: {feedback}")
        correccion_prompt = (
            f'INSTRUCCIÓN DE CORRECCIÓN: Tu respuesta anterior fue incorrecta. '
            f'Feedback: "{feedback}". Genera una NUEVA respuesta corrigiendo este error.'
        )

    preview = _df_preview(df, 50) or "(sin datos en vista previa; verifica la consulta)"

    prompt_analisis = f"""{correccion_prompt}
Eres IANA, un analista de datos senior EXTREMADAMENTE PRECISO y riguroso.
---
<<< REGLAS CRÍTICAS DE PRECISIÓN >>>
1. **NO ALUCINAR**: NUNCA inventes números, totales, porcentajes o nombres que no estén en 'Datos'.
2. **DATOS INCOMPLETOS**: Reporta los vacíos (p.ej., "sin datos para Marzo") sin inventar valores.
3. **VERIFICAR CÁLCULOS**: Revisa sumas/conteos/promedios con los datos.
4. **CITAR DATOS**: Cada afirmación debe inferirse de 'Datos'.
---
Pregunta Original: {pregunta_usuario}
{hist_text}
Datos (usa SÓLO estos):
{preview}
---
FORMATO OBLIGATORIO:
📌 Resumen Ejecutivo:
- (Hallazgos principales basados ESTRICTAMENTE en los datos.)
🔍 Números de referencia:
- (Cifras clave calculadas DIRECTAMENTE de los datos.)
"""
    with st.spinner("💡 Generando análisis avanzado..."):
        analisis = llm_analista.invoke(prompt_analisis).content
    st.success("💡 ¡Análisis completado!")
    return analisis

def responder_conversacion(pregunta_usuario: str, hist_text: str):
    st.info("💬 Activando modo de conversación...")
    prompt_personalidad = f"""
Tu nombre es IANA, una IA amable de Ventus. Ayuda a analizar datos.
Si el usuario hace un comentario casual, responde brevemente y redirígelo a tus capacidades.
{hist_text}
Pregunta: "{pregunta_usuario}"
"""
    respuesta = llm_analista.invoke(prompt_personalidad).content
    return {"texto": respuesta, "df": None, "analisis": None}

# ============================================
# 4) Orquestador y Validación
# ============================================

def validar_y_corregir_respuesta_analista(pregunta_usuario: str, res_analisis: dict, hist_text: str) -> dict:
    MAX_INTENTOS = 2
    for intento in range(MAX_INTENTOS):
        st.info(f"🕵️‍♀️ Supervisor de Calidad: Verificando análisis (Intento {intento + 1})...")

        contenido_respuesta = res_analisis.get("analisis", "") or ""
        if not contenido_respuesta.strip():
            return {"tipo": "error", "texto": "El análisis generado estaba vacío."}

        df_preview = _df_preview(res_analisis.get("df"), 50) or "(sin vista previa de datos)"

        prompt_validacion = f"""
Eres un supervisor de calidad estricto. Valida si el 'Análisis' se basa ESTRICTAMENTE en los 'Datos de Soporte'.
FORMATO:
- Si está 100% basado en los datos: APROBADO
- Si alucina/inventa/no es relevante: RECHAZADO: [razón corta y accionable]
---
Pregunta: "{pregunta_usuario}"
Datos de Soporte:
{df_preview}
---
Análisis a evaluar:
\"\"\"{contenido_respuesta}\"\"\"
---
Evaluación:
"""
        try:
            resultado = llm_validador.invoke(prompt_validacion).content.strip()
            up = resultado.upper()
            if up.startswith("APROBADO"):
                st.success("✅ Análisis aprobado por el Supervisor.")
                return res_analisis
            elif up.startswith("RECHAZADO"):
                feedback_previo = resultado.split(":", 1)[1].strip() if ":" in resultado else "Razón no especificada."
                st.warning(f"❌ Análisis rechazado. Feedback: {feedback_previo}")
                if intento < MAX_INTENTOS - 1:
                    st.info("🔄 Regenerando análisis con feedback...")
                    res_analisis["analisis"] = analizar_con_datos(pregunta_usuario, hist_text, res_analisis.get("df"), feedback=feedback_previo)
                else:
                    return {"tipo": "error", "texto": "El análisis no fue satisfactorio incluso después de una corrección."}
            else:
                return {"tipo": "error", "texto": f"Respuesta ambigua del validador: {resultado}"}
        except Exception as e:
            return {"tipo": "error", "texto": f"Excepción durante la validación: {e}"}
    return {"tipo": "error", "texto": "Se alcanzó el límite de intentos de validación."}

def clasificar_intencion(pregunta: str) -> str:
    prompt_orq = f"""
Clasifica la intención en UNA SOLA PALABRA:
1) analista: si pide interpretación / resumen / comparación / por qué / tendencias / insights.
2) consulta: si pide datos crudos (listas, conteos, totales) y NO hay palabras clave de analista.
3) conversacional: saludos o general.
Pregunta: "{pregunta}"
Clasificación:
"""
    try:
        opciones = {"consulta", "analista", "conversacional"}
        r = llm_orq.invoke(prompt_orq).content.strip().lower().replace('"', '').replace("'", "")
        return r if r in opciones else "conversacional"
    except Exception:
        return "conversacional"

def obtener_datos_sql(pregunta_usuario: str, hist_text: str) -> dict:
    # Reutiliza DF anterior si el usuario lo sugiere
    if any(keyword in pregunta_usuario.lower() for keyword in ["anterior", "esos datos", "esa tabla"]):
        for msg in reversed(st.session_state.get('messages', [])):
            if msg.get('role') == 'assistant':
                content = msg.get('content', {})
                df_prev = content.get('df')
                if isinstance(df_prev, pd.DataFrame) and not df_prev.empty:
                    st.info("💡 Usando datos de la respuesta anterior para la nueva solicitud.")
                    return {"df": df_prev}

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
# 5) Interfaz de Chat de Streamlit
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": {"texto": "¡Hola! Soy IANA, tu asistente de IA de Ventus. ¿Qué te gustaría saber?"}
    }]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message.get("content", {})
        if isinstance(content, dict):
            if content.get("texto"):
                st.markdown(content["texto"])
            if isinstance(content.get("df"), pd.DataFrame) and not content["df"].empty:
                st.dataframe(content["df"])
            if content.get("analisis"):
                st.markdown(content["analisis"])
        elif isinstance(content, str):
            st.markdown(content)

# --- Entrada por audio (opcional) ---
st.markdown("### 🎙️ Subir audio (opcional)")
audio_file = st.file_uploader(
    "Adjunta un archivo de audio (mp3, wav, m4a) con tu pregunta",
    type=["mp3", "wav", "m4a"],
    accept_multiple_files=False
)

texto_desde_audio = None
if audio_file is not None:
    texto_desde_audio = transcribir_audio(audio_file)
    if texto_desde_audio:
        st.text_area("Transcripción detectada", value=texto_desde_audio, height=120, help="Puedes editar el texto antes de enviar.")

# --- Unificar entrada (audio o texto) ---
prompt_text = texto_desde_audio if texto_desde_audio else st.chat_input("Pregunta por costos, proveedores, familia...")

if prompt_text:
    if not all([db, llm_sql, llm_analista, llm_orq, agente_sql, llm_validador]):
        st.error("La aplicación no está completamente inicializada. Revisa los errores de conexión o de API key.")
    else:
        st.session_state.messages.append({"role": "user", "content": {"texto": prompt_text}})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        with st.chat_message("assistant"):
            res = orquestador(prompt_text, st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": res})

            if res and res.get("tipo") != "error":
                if res.get("texto"):
                    st.markdown(res["texto"])
                if isinstance(res.get("df"), pd.DataFrame) and not res["df"].empty:
                    st.dataframe(res["df"])
                if res.get("analisis"):
                    st.markdown("---")
                    st.markdown("### 🧠 Análisis de IANA")
                    st.markdown(res["analisis"])
                    st.toast("Análisis generado ✅", icon="✅")
            elif res:
                st.error(res.get("texto", "Ocurrió un error inesperado."))
                st.toast("Hubo un error en la consulta ❌", icon="❌")
