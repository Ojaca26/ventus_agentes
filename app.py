# app.py

import streamlit as st
import pandas as pd
import re
import io
from typing import Optional
from sqlalchemy import text

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import create_sql_query_chain

# Micrófono en vivo (frontend) + fallback SR
from streamlit_mic_recorder import speech_to_text, mic_recorder
import speech_recognition as sr

# <<< NUEVOS IMPORTS para el Agente de Correo >>>
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import json

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
    # ... (código sin cambios)
    with st.spinner("🛰️ Conectando a la base de datos de Ventus..."):
        try:
            creds = st.secrets["db_credentials"]
            uri = f"mysql+pymysql://{creds['user']}:{creds['password']}@{creds['host']}/{creds['database']}"
            engine_args = {"pool_recycle": 3600, "pool_pre_ping": True}
            db = SQLDatabase.from_uri(uri, include_tables=["ventus"], engine_args=engine_args)
            st.success("✅ Conexión a la base de datos establecida.")
            return db
        except Exception as e:
            st.error(f"Error al conectar a la base de datos: {e}")
            return None

@st.cache_resource
def get_llms():
    # ... (código sin cambios)
    with st.spinner("🤝 Inicializando la red de agentes IANA..."):
        try:
            api_key = st.secrets["google_api_key"]
            common_config = dict(temperature=0.1, google_api_key=api_key)
            llm_sql = ChatGoogleGenerativeAI(model="gemini-1.5-pro", **common_config)
            llm_analista = ChatGoogleGenerativeAI(model="gemini-1.5-pro", **common_config)
            llm_orq = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0, google_api_key=api_key)
            llm_validador = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0, google_api_key=api_key)
            st.success("✅ Agentes de IANA listos.")
            return llm_sql, llm_analista, llm_orq, llm_validador
        except Exception as e:
            st.error(f"Error al inicializar los LLMs. Revisa tu API key. Detalle: {e}")
            return None, None, None, None

db = get_database_connection()
llm_sql, llm_analista, llm_orq, llm_validador = get_llms()

@st.cache_resource
def get_sql_agent(_llm, _db):
    # ... (código sin cambios)
    if not _llm or not _db: return None
    with st.spinner("🛠️ Configurando agente SQL de IANA..."):
        toolkit = SQLDatabaseToolkit(db=_db, llm=_llm)
        agent = create_sql_agent(llm=_llm, toolkit=toolkit, verbose=False, top_k=1000)
        st.success("✅ Agente SQL configurado.")
        return agent

agente_sql = get_sql_agent(llm_sql, db)

# ============================================
# 1.b) Reconocedor (fallback local) (SIN CAMBIOS)
# ============================================

@st.cache_resource
def get_recognizer():
    # ... (código sin cambios)
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.dynamic_energy_threshold = True
    return r

def transcribir_audio_bytes(data_bytes: bytes, language: str) -> Optional[str]:
    # ... (código sin cambios)
    try:
        r = get_recognizer()
        with sr.AudioFile(io.BytesIO(data_bytes)) as source:
            audio = r.record(source)
        texto = r.recognize_google(audio, language=language)
        return texto.strip() if texto else None
    except Exception:
        return None

# ============================================
# <<< NUEVO >>> Agente de Correo
# ============================================

def extraer_detalles_correo(pregunta_usuario: str, df: pd.DataFrame) -> dict:
    """Usa un LLM para extraer destinatario, asunto y cuerpo del prompt del usuario."""
    st.info("🧠 El agente de correo está interpretando tu solicitud...")
    
    default_recipient = st.secrets.get("email_credentials", {}).get("default_recipient", "")
    
    prompt = f"""
    Dada la pregunta del usuario, extrae la información para enviar un correo. El output DEBE SER un JSON válido.
    
    Pregunta: "{pregunta_usuario}"
    
    1.  `recipient`: Busca una dirección de correo. Si no la encuentras, usa "{default_recipient}".
    2.  `subject`: Crea un asunto corto y descriptivo.
    3.  `body`: Crea un cuerpo de texto breve y profesional para el correo.
    
    JSON Output:
    """
    
    try:
        response = llm_analista.invoke(prompt).content
        # Limpiar la respuesta para que sea un JSON válido
        json_response = response.strip().replace("```json", "").replace("```", "").strip()
        details = json.loads(json_response)
        return {
            "recipient": details.get("recipient", default_recipient),
            "subject": details.get("subject", "Reporte de Datos - IANA"),
            "body": details.get("body", "Adjunto encontrarás los datos solicitados.")
        }
    except Exception:
        st.warning("No pude interpretar los detalles del correo, usaré los valores por defecto.")
        return {
            "recipient": default_recipient,
            "subject": "Reporte de Datos - IANA",
            "body": "Adjunto encontrarás los datos solicitados."
        }


def enviar_correo_agente(recipient: str, subject: str, body: str, df: Optional[pd.DataFrame] = None):
    """Construye y envía un correo con un DataFrame como adjunto CSV."""
    with st.spinner(f"📧 Enviando correo a {recipient}..."):
        try:
            creds = st.secrets["email_credentials"]
            sender_email = creds["sender_email"]
            sender_password = creds["sender_password"]
            
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            if df is not None and not df.empty:
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                
                attachment = MIMEApplication(csv_buffer.getvalue(), _subtype='csv')
                attachment.add_header('Content-Disposition', 'attachment', filename="datos_iana.csv")
                msg.attach(attachment)
            
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            st.success(f"✅ Correo enviado exitosamente a {recipient}!")
            return {"texto": f"¡Listo! El correo fue enviado a {recipient}."}
            
        except Exception as e:
            st.error(f"❌ No se pudo enviar el correo. Error: {e}")
            return {"tipo": "error", "texto": f"Lo siento, no pude enviar el correo. Detalle del error: {e}"}


# ============================================
# 2) Funciones Auxiliares (SIN CAMBIOS)
# ============================================
def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    # ... (código sin cambios)
    s2 = s.astype(str).str.replace(r'[\u00A0\s]', '', regex=True).str.replace(',', '', regex=False).str.replace('$', '', regex=False).str.replace('%', '', regex=False)
    try: return pd.to_numeric(s2)
    except Exception: return s

def get_history_text(chat_history: list, n_turns=3) -> str:
    # ... (código sin cambios)
    if not chat_history or len(chat_history) <= 1: return ""
    history_text = []
    relevant_history = chat_history[-(n_turns * 2 + 1) : -1]
    for msg in relevant_history:
        content = msg.get("content", {}); text_content = ""
        if isinstance(content, dict): text_content = content.get("texto", "")
        elif isinstance(content, str): text_content = content
        if text_content:
            role = "Usuario" if msg["role"] == "user" else "IANA"
            history_text.append(f"{role}: {text_content}")
    if not history_text: return ""
    return "\n--- Contexto de Conversación Anterior ---\n" + "\n".join(history_text) + "\n--- Fin del Contexto ---\n"

def markdown_table_to_df(texto: str) -> pd.DataFrame:
    # ... (código sin cambios)
    lineas = [l.rstrip() for l in texto.splitlines() if l.strip().startswith('|')]
    if not lineas: return pd.DataFrame()
    lineas = [l for l in lineas if not re.match(r'^\|\s*-{2,}', l)]
    filas = [[c.strip() for c in l.strip('|').split('|')] for l in lineas]
    if len(filas) < 2: return pd.DataFrame()
    header, data = filas[0], filas[1:]
    max_cols = len(header); data = [r + ['']*(max_cols - len(r)) if len(r) < max_cols else r[:max_cols] for r in data]
    df = pd.DataFrame(data, columns=header)
    for c in df.columns: df[c] = _coerce_numeric_series(df[c])
    return df

def _df_preview(df: pd.DataFrame, n: int = 5) -> str:
    # ... (código sin cambios)
    if df is None or df.empty: return ""
    try: return df.head(n).to_markdown(index=False)
    except Exception: return df.head(n).to_string(index=False)

def interpretar_resultado_sql(res: dict) -> dict:
    # ... (código sin cambios)
    df = res.get("df")
    if df is not None and not df.empty and res.get("texto") is None:
        if df.shape == (1, 1):
            valor = df.iloc[0, 0]; nombre_columna = df.columns[0]
            res["texto"] = f"La respuesta para '{nombre_columna}' es: **{valor}**"
            st.info("💡 Resultado numérico interpretado para una respuesta directa.")
    return res

def _asegurar_select_only(sql: str) -> str:
    # ... (código sin cambios)
    sql_clean = sql.strip().rstrip(';')
    if not re.match(r'(?is)^\s*select\b', sql_clean):
        raise ValueError("Solo se permite ejecutar consultas SELECT.")
    sql_clean = re.sub(r'(?is)\blimit\s+\d+\s*$', '', sql_clean).strip()
    return sql_clean

# ============================================
# 3) Agentes (SIN CAMBIOS)
# ============================================

def ejecutar_sql_real(pregunta_usuario: str, hist_text: str):
    # ... (código sin cambios)
    st.info("🤖 El agente de datos está traduciendo tu pregunta a SQL...")
    prompt_con_instrucciones = f"""Tu tarea es generar una consulta SQL limpia (SOLO SELECT) sobre la tabla `ventus` para responder la pregunta del usuario.\n---\n<<< REGLA DE ORO PARA BÚSQUEDA DE PRODUCTOS >>>\n1. La columna `Producto` contiene descripciones largas.\n2. Si el usuario pregunta por un producto o servicio específico (ej: 'transporte', 'guantes'), usa `WHERE LOWER(Producto) LIKE '%palabra%'.\n3. Ejemplo: "cuántos transportes..." -> `WHERE LOWER(Producto) LIKE '%transporte%'`.\n4. No agregues LIMIT.\n---\n{hist_text}\nPregunta del usuario: "{pregunta_usuario}"\nDevuelve SOLO la consulta SQL (sin explicaciones)."""
    try:
        query_chain = create_sql_query_chain(llm_sql, db)
        sql_query_bruta = query_chain.invoke({"question": prompt_con_instrucciones})
        m = re.search(r'(?is)(select\b.+)$', sql_query_bruta.strip()); sql_query_limpia = m.group(1).strip() if m else sql_query_bruta.strip()
        sql_query_limpia = re.sub(r'(?is)^```sql|```$', '', sql_query_limpia).strip(); sql_query_limpia = _asegurar_select_only(sql_query_limpia)
        st.code(sql_query_limpia, language='sql')
        with st.spinner("⏳ Ejecutando consulta..."):
            with db._engine.connect() as conn: df = pd.read_sql(text(sql_query_limpia), conn)
        st.success(f"✅ ¡Consulta ejecutada! Filas: {len(df)}")
        return {"sql": sql_query_limpia, "df": df}
    except Exception as e:
        st.warning(f"❌ Error en la consulta directa. Intentando método alternativo... Detalle: {e}")
        return {"sql": None, "df": None, "error": str(e)}

def ejecutar_sql_en_lenguaje_natural(pregunta_usuario: str, hist_text: str):
    # ... (código sin cambios)
    st.info("🤔 Activando el agente SQL experto como plan B.")
    prompt_sql = (f"Tu tarea es responder la pregunta consultando la tabla 'ventus'.\n{hist_text}\nDevuelve ÚNICAMENTE una tabla en formato Markdown (con encabezados). NUNCA resumas ni expliques. El SQL interno NO DEBE CONTENER 'LIMIT'.\nPregunta: {pregunta_usuario}")
    try:
        with st.spinner("💬 Pidiendo al agente SQL que responda..."):
            res = agente_sql.invoke(prompt_sql)
            texto = res["output"] if isinstance(res, dict) and "output" in res else str(res)
        st.info("📝 Intentando convertir la respuesta en una tabla de datos..."); df_md = markdown_table_to_df(texto)
        if df_md.empty: st.warning("La conversión de Markdown a tabla no produjo filas. Se mostrará la salida cruda.")
        return {"texto": texto, "df": df_md}
    except Exception as e:
        st.error(f"❌ El agente SQL experto también encontró un problema: {e}")
        return {"texto": f"[SQL_ERROR] {e}", "df": pd.DataFrame()}

def analizar_con_datos(pregunta_usuario: str, hist_text: str, df: pd.DataFrame | None, feedback: str = None):
    # ... (código sin cambios)
    st.info("\n🧠 El analista experto está examinando los datos...")
    correccion_prompt = ""
    if feedback:
        st.warning(f"⚠️ Reintentando con feedback: {feedback}")
        correccion_prompt = (f'INSTRUCCIÓN DE CORRECCIÓN: Tu respuesta anterior fue incorrecta. Feedback: "{feedback}". Genera una NUEVA respuesta corrigiendo este error.')
    preview = _df_preview(df, 50) or "(sin datos en vista previa; verifica la consulta)"
    prompt_analisis = f"""{correccion_prompt}\nEres IANA, un analista de datos senior EXTREMADAMENTE PRECISO y riguroso.\n---\n<<< REGLAS CRÍTICAS DE PRECISIÓN >>>\n1. **NO ALUCINAR**: NUNCA inventes números, totales, porcentajes o nombres de productos/categorías que no estén EXPRESAMENTE en la tabla de 'Datos'.\n2. **DATOS INCOMPLETOS**: Reporta los vacíos (p.ej., "sin datos para Marzo") sin inventar valores.\n3. **VERIFICAR CÁLCULOS**: Antes de escribir un número, revisa el cálculo (sumas/conteos/promedios) con los datos.\n4. **CITAR DATOS**: Basa CADA afirmación que hagas en los datos visibles en la tabla.\n---\nPregunta Original: {pregunta_usuario}\n{hist_text}\nDatos para tu análisis (usa SÓLO estos):\n{preview}\n---\nFORMATO OBLIGATORIO:\n📌 Análisis Ejecutivo de datos:\n1. Calcular totales y porcentajes clave.\n2. Detectar concentración.\n3. Identificar patrones temporales.\n4. Analizar dispersión.\nEntregar el resultado en 3 bloques:\n📌 Resumen Ejecutivo: hallazgos principales con números.\n🔍 Números de referencia: totales, promedios, ratios.\n⚠ Importante: Sé muy breve, directo y diciente."""
    with st.spinner("💡 Generando análisis avanzado..."):
        analisis = llm_analista.invoke(prompt_analisis).content
    st.success("💡 ¡Análisis completado!")
    return analisis

def responder_conversacion(pregunta_usuario: str, hist_text: str):
    # ... (código sin cambios)
    st.info("💬 Activando modo de conversación...")
    prompt_personalidad = f"""Tu nombre es IANA, una IA amable de Ventus. Ayuda a analizar datos.\nSi el usuario hace un comentario casual, responde amablemente de forma natural, muy humana y redirígelo a tus capacidades.\n{hist_text}\nPregunta: "{pregunta_usuario}" """
    respuesta = llm_analista.invoke(prompt_personalidad).content
    return {"texto": respuesta, "df": None, "analisis": None}

# ============================================
# 4) Orquestador y Validación
# ============================================

def validar_y_corregir_respuesta_analista(pregunta_usuario: str, res_analisis: dict, hist_text: str) -> dict:
    # ... (código sin cambios)
    MAX_INTENTOS = 2
    for intento in range(MAX_INTENTOS):
        st.info(f"🕵️‍♀️ Supervisor de Calidad: Verificando análisis (Intento {intento + 1})..."); contenido_respuesta = res_analisis.get("analisis", "") or ""
        if not contenido_respuesta.strip(): return {"tipo": "error", "texto": "El análisis generado estaba vacío."}
        df_preview = _df_preview(res_analisis.get("df"), 50) or "(sin vista previa de datos)"
        prompt_validacion = f"""Eres un supervisor de calidad estricto. Valida si el 'Análisis' se basa ESTRICTAMENTE en los 'Datos de Soporte'.\nFORMATO:\n- Si está 100% basado en los datos: APROBADO\n- Si alucina/inventa/no es relevante: RECHAZADO: [razón corta y accionable]\n---\nPregunta: "{pregunta_usuario}"\nDatos de Soporte:\n{df_preview}\n---\nAnálisis a evaluar:\n\"\"\"{contenido_respuesta}\"\"\"\n---\nEvaluación:"""
        try:
            resultado = llm_validador.invoke(prompt_validacion).content.strip(); up = resultado.upper()
            if up.startswith("APROBADO"):
                st.success("✅ Análisis aprobado por el Supervisor."); return res_analisis
            elif up.startswith("RECHAZADO"):
                feedback_previo = resultado.split(":", 1)[1].strip() if ":" in resultado else "Razón no especificada."
                st.warning(f"❌ Análisis rechazado. Feedback: {feedback_previo}")
                if intento < MAX_INTENTOS - 1:
                    st.info("🔄 Regenerando análisis con feedback...")
                    res_analisis["analisis"] = analizar_con_datos(pregunta_usuario, hist_text, res_analisis.get("df"), feedback=feedback_previo)
                else: return {"tipo": "error", "texto": "El análisis no fue satisfactorio incluso después de una corrección."}
            else: return {"tipo": "error", "texto": f"Respuesta ambigua del validador: {resultado}"}
        except Exception as e: return {"tipo": "error", "texto": f"Excepción durante la validación: {e}"}
    return {"tipo": "error", "texto": "Se alcanzó el límite de intentos de validación."}

def clasificar_intencion(pregunta: str) -> str:
    # <<< MODIFICADO para incluir la nueva intención 'correo' >>>
    prompt_orq = f"""
Clasifica la intención del usuario en UNA SOLA PALABRA. Presta especial atención a los verbos de acción y palabras clave.
1. `analista`: Si la pregunta pide explícitamente una interpretación, resumen, comparación o explicación.
   PALABRAS CLAVE PRIORITARIAS: analiza, compara, resume, explica, por qué, tendencia, insights, dame un análisis, haz un resumen.
   Si una de estas palabras clave está presente, la intención SIEMPRE es `analista`.

2. `consulta`: Si la pregunta pide datos crudos (listas, conteos, totales) y NO contiene una palabra clave prioritaria de `analista`.
   Ejemplos: 'cuántos proveedores hay', 'lista todos los productos', 'muéstrame el total', 'y ahora por mes'.

3. `correo`: Si la pregunta pide explícitamente enviar un correo, email o reporte.
   PALABRAS CLAVE: envía, mandar, correo, email, reporte a.

4. `conversacional`: Si es un saludo o una pregunta general no relacionada con datos.
   Ejemplos: 'hola', 'gracias', 'qué puedes hacer'.

Pregunta: "{pregunta}"
Clasificación:
"""
    try:
        opciones = {"consulta", "analista", "conversacional", "correo"}
        r = llm_orq.invoke(prompt_orq).content.strip().lower().replace('"', '').replace("'", "")
        return r if r in opciones else "conversacional"
    except Exception:
        return "conversacional"

def obtener_datos_sql(pregunta_usuario: str, hist_text: str) -> dict:
    # ... (código sin cambios)
    if any(keyword in pregunta_usuario.lower() for keyword in ["anterior", "esos datos", "esa tabla"]):
        for msg in reversed(st.session_state.get('messages', [])):
            if msg.get('role') == 'assistant':
                content = msg.get('content', {}); df_prev = content.get('df')
                if isinstance(df_prev, pd.DataFrame) and not df_prev.empty:
                    st.info("💡 Usando datos de la respuesta anterior para la nueva solicitud.")
                    return {"df": df_prev}
    res_real = ejecutar_sql_real(pregunta_usuario, hist_text)
    if res_real.get("df") is not None and not res_real["df"].empty:
        return res_real
    return ejecutar_sql_en_lenguaje_natural(pregunta_usuario, hist_text)

def orquestador(pregunta_usuario: str, chat_history: list):
    # <<< MODIFICADO para manejar la nueva intención 'correo' >>>
    with st.expander("⚙️ Ver Proceso de IANA", expanded=False):
        hist_text = get_history_text(chat_history)
        clasificacion = clasificar_intencion(pregunta_usuario)
        st.success(f"✅ ¡Intención detectada! Tarea: {clasificacion.upper()}.")

        if clasificacion == "conversacional":
            return responder_conversacion(pregunta_usuario, hist_text)
        
        # <<< NUEVO >>> Flujo para el agente de correo
        if clasificacion == "correo":
            df_para_enviar = None
            for msg in reversed(st.session_state.get('messages', [])):
                if msg.get('role') == 'assistant':
                    content = msg.get('content', {}); df_prev = content.get('df')
                    if isinstance(df_prev, pd.DataFrame) and not df_prev.empty:
                        df_para_enviar = df_prev
                        st.info("📧 Datos de la tabla anterior encontrados para adjuntar al correo.")
                        break
            
            if df_para_enviar is None:
                st.warning("No encontré una tabla en la conversación reciente para enviar. El correo irá sin datos adjuntos.")

            detalles = extraer_detalles_correo(pregunta_usuario, df_para_enviar)
            return enviar_correo_agente(
                recipient=detalles["recipient"],
                subject=detalles["subject"],
                body=detalles["body"],
                df=df_para_enviar
            )

        # Flujos existentes
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
# 5) Interfaz: Micrófono en vivo + Chat (SIN CAMBIOS)
# ============================================

if "messages" not in st.session_state:
    # ... (código sin cambios)
    st.session_state.messages = [{"role": "assistant", "content": {"texto": "¡Hola! Soy IANA, tu asistente de IA de Ventus. ¿Qué te gustaría saber?"}}]

for message in st.session_state.messages:
    # ... (código sin cambios)
    with st.chat_message(message["role"]):
        content = message.get("content", {});
        if isinstance(content, dict):
            if content.get("texto"): st.markdown(content["texto"])
            if isinstance(content.get("df"), pd.DataFrame) and not content["df"].empty: st.dataframe(content["df"])
            if content.get("analisis"): st.markdown(content["analisis"])
        elif isinstance(content, str): st.markdown(content)

st.markdown("### 🎤 Habla con IANA")
lang = st.secrets.get("stt_language", "es-CO")

text = speech_to_text(language=lang, start_prompt="🎙️ Hablar", stop_prompt="🛑 Detener", use_container_width=True, just_once=True, key="stt")

if text:
    st.text_area("Transcripción:", value=text, key="voice_transcript")

prompt_text = st.chat_input("... o escribe tu pregunta aquí")
final_prompt = prompt_text if prompt_text else st.session_state.get("voice_transcript")

if final_prompt:
    # Limpiar el transcript para evitar re-envíos
    if "voice_transcript" in st.session_state: st.session_state.voice_transcript = ""

    if not all([db, llm_sql, llm_analista, llm_orq, agente_sql, llm_validador]):
        st.error("La aplicación no está completamente inicializada. Revisa los errores de conexión o de API key.")
    else:
        st.session_state.messages.append({"role": "user", "content": {"texto": final_prompt}})
        with st.chat_message("user"): st.markdown(final_prompt)
        with st.chat_message("assistant"):
            res = orquestador(final_prompt, st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": res})
            if res and res.get("tipo") != "error":
                if res.get("texto"): st.markdown(res["texto"])
                if isinstance(res.get("df"), pd.DataFrame) and not res["df"].empty: st.dataframe(res["df"])
                if res.get("analisis"):
                    st.markdown("---"); st.markdown("### 🧠 Análisis de IANA"); st.markdown(res["analisis"])
                    st.toast("Análisis generado ✅", icon="✅")
            elif res:
                st.error(res.get("texto", "Ocurrió un error inesperado."))
                st.toast("Hubo un error ❌", icon="❌")
    
