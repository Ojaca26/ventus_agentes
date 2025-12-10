# app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
from typing import Optional
from sqlalchemy import text, create_engine

# LangChain + Gemini / OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain


from streamlit_mic_recorder import speech_to_text, mic_recorder
import speech_recognition as sr

# Agente de Correo
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
    with st.spinner("🛰️ Conectando a la base de datos de Ventus..."):
        try:
            creds = st.secrets["db_credentials"]
            uri = f"mysql+pymysql://{creds['user']}:{creds['password']}@{creds['host']}/{creds['database']}"
            engine_args = {
                "pool_recycle": 3600,
                "pool_pre_ping": True,
                "connect_args": {"connect_timeout": 10}  # ⏱️ límite de conexión 10 segundos
            }
            db = SQLDatabase.from_uri(uri, include_tables=["ventus_bi"], engine_args=engine_args)
            st.success("✅ Conexión a la base de datos establecida.")
            return db
        except Exception as e:
            st.error(f"❌ Error al conectar a la base de datos: {e}")
            return None

@st.cache_resource
def get_llms():
    with st.spinner("🤝 Inicializando la red de agentes IANA..."):
        try:
            api_key = st.secrets["openai_api_key"]
            model_name = "gpt-4o"
            llm_sql = ChatOpenAI(model=model_name, temperature=0.1, openai_api_key=api_key)
            llm_analista = ChatOpenAI(model=model_name, temperature=0.1, openai_api_key=api_key)
            llm_orq = ChatOpenAI(model=model_name, temperature=0.0, openai_api_key=api_key)
            llm_validador = ChatOpenAI(model=model_name, temperature=0.0, openai_api_key=api_key)

            #api_key = st.secrets["google_api_key"]
            #common_config = dict(temperature=0.1, google_api_key=api_key)
            #llm_sql = ChatGoogleGenerativeAI(model="gemini-1.5-pro", **common_config)
            #llm_analista = ChatGoogleGenerativeAI(model="gemini-1.5-pro", **common_config)
            #llm_orq = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0, google_api_key=api_key)
            #llm_validador = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0, google_api_key=api_key)
            
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
        try:
            toolkit = SQLDatabaseToolkit(db=_db, llm=_llm)

            agent = create_sql_agent(
                llm=_llm,
                toolkit=toolkit,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3,
                early_stopping_method="generate"  # 🛡️ fuerza salida segura en errores
            )

            st.success("✅ Agente SQL configurado correctamente.")
            return agent
        except Exception as e:
            st.error(f"❌ No se pudo inicializar el agente SQL. Detalle: {e}")
            return None

agente_sql = get_sql_agent(llm_sql, db)

# ============================================
# 1.b) Reconocedor de Voz (fallback local)
# ============================================

@st.cache_resource
def get_recognizer():
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.dynamic_energy_threshold = True
    return r

def transcribir_audio_bytes(data_bytes: bytes, language: str) -> Optional[str]:
    try:
        r = get_recognizer()
        with sr.AudioFile(io.BytesIO(data_bytes)) as source:
            audio = r.record(source)
        texto = r.recognize_google(audio, language=language)
        return texto.strip() if texto else None
    except Exception:
        return None

# ============================================
# 2) Agente de Correo (Lógica Mejorada)
# ============================================

def extraer_detalles_correo(pregunta_usuario: str) -> dict:
    st.info("🧠 El agente de correo está interpretando tu solicitud...")
    
    # Cargar la "agenda de contactos" desde los secretos
    contactos = dict(st.secrets.get("named_recipients", {}))
    default_recipient_name = st.secrets.get("email_credentials", {}).get("default_recipient", "")
    
    prompt = f"""
    Tu tarea es analizar la pregunta de un usuario y extraer los detalles para enviar un correo. Tu output DEBE SER un JSON válido.

    Agenda de Contactos Disponibles: {', '.join(contactos.keys())}

    Pregunta del usuario: "{pregunta_usuario}"

    Instrucciones para extraer:
    1.  `recipient_name`: Busca un nombre de la "Agenda de Contactos" en la pregunta. Si encuentras un nombre (ej: "Oscar"), pon ese nombre aquí. Si encuentras una dirección de correo explícita (ej: "test@test.com"), pon la dirección completa aquí. Si no encuentras ni nombre ni correo, usa "default".
    2.  `subject`: Crea un asunto corto y descriptivo basado en la pregunta.
    3.  `body`: Crea un cuerpo de texto breve y profesional para el correo.

    Ejemplo:
    Pregunta: "envía el reporte a Oscar por favor"
    JSON Output:
    {{
        "recipient_name": "Oscar",
        "subject": "Reporte de Datos Solicitado",
        "body": "Hola, como solicitaste, aquí tienes el reporte con los datos."
    }}
    
    JSON Output para la pregunta actual:
    """
    
    try:
        response = llm_analista.invoke(prompt).content
        json_response = response.strip().replace("```json", "").replace("```", "").strip()
        details = json.loads(json_response)
        
        recipient_identifier = details.get("recipient_name", "default")
        
        # Resolver el identificador a un correo real
        if "@" in recipient_identifier:
            final_recipient = recipient_identifier  # Ya es un correo
        elif recipient_identifier in contactos:
            final_recipient = contactos[recipient_identifier] # Buscar en la agenda
        else:
            final_recipient = default_recipient_name # Usar el por defecto

        return {
            "recipient": final_recipient,
            "subject": details.get("subject", "Reporte de Datos - IANA"),
            "body": details.get("body", "Adjunto encontrarás los datos solicitados.")
        }
    except Exception as e:
        st.warning(f"No pude interpretar los detalles del correo (error: {e}), usaré los valores por defecto.")
        return {
            "recipient": default_recipient_name,
            "subject": "Reporte de Datos - IANA",
            "body": "Adjunto encontrarás los datos solicitados."
        }


def enviar_correo_agente(recipient: str, subject: str, body: str, df: Optional[pd.DataFrame] = None):
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
# 3) Funciones Auxiliares y Agentes (SIN CAMBIOS)
# ============================================
# (Todas las funciones desde _coerce_numeric_series hasta responder_conversacion se mantienen igual)
def seleccionar_columnas_valor(df: pd.DataFrame) -> list[str]:
    """
    Selecciona columnas numéricas que probablemente representen valores a totalizar,
    usando palabras clave en el nombre.
    """
    # Todas las columnas numéricas
    num_cols = df.select_dtypes(include="number").columns.tolist()  # pandas recomienda usar np.number o 'number'. :contentReference[oaicite:0]{index=0}

    # Filtrar por palabras clave de negocio relevantes
    patron = re.compile(r"(COP|USD|Monto|Valor|Total|Costo|Facturado|Pendiente)", re.IGNORECASE)

    columnas_valor = [c for c in num_cols if patron.search(c)]

    # Excluir columnas de control/ID si existieran
    excluir = {"id", "codigo"}
    columnas_valor = [c for c in columnas_valor if c.lower() not in excluir]

    return columnas_valor

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.replace(r'[\u00A0\s]', '', regex=True).str.replace(',', '', regex=False).str.replace('$', '', regex=False).str.replace('%', '', regex=False)
    try: return pd.to_numeric(s2)
    except Exception: return s
def get_history_text(chat_history: list, n_turns=3) -> str:
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
    if df is None or df.empty: return ""
    try: return df.head(n).to_markdown(index=False)
    except Exception: return df.head(n).to_string(index=False)
def interpretar_resultado_sql(res: dict) -> dict:
    df = res.get("df")
    if df is not None and not df.empty and res.get("texto") is None:
        if df.shape == (1, 1):
            valor = df.iloc[0, 0]; nombre_columna = df.columns[0]
            res["texto"] = f"La respuesta para '{nombre_columna}' es: **{valor}**"
            st.info("💡 Resultado numérico interpretado para una respuesta directa.")
    return res
def _asegurar_select_only(sql: str) -> str:
    sql_clean = sql.strip().rstrip(';')
    if not re.match(r'(?is)^\s*select\b', sql_clean): raise ValueError("Solo se permite ejecutar consultas SELECT.")
    sql_clean = re.sub(r'(?is)\blimit\s+\d+\s*$', '', sql_clean).strip()
    return sql_clean

def limpiar_sql(sql_texto: str) -> str:
    """
    Limpia texto generado por LLM para dejar solo la consulta SQL válida.
    - Elimina prefijos como 'sql', 'sql:', 'SQL\n'
    - Elimina etiquetas ```sql``` o ``` ```
    - Recorta espacios y saltos de línea.
    """
    if not sql_texto:
        return ""

    # 🔥 Elimina etiquetas markdown primero
    limpio = re.sub(r'```sql|```', '', sql_texto, flags=re.I)

    # 🔥 Elimina cualquier prefijo 'sql' seguido de espacio, ':' o salto de línea
    # Usa '+' para capturar uno o más separadores (más robusto que \n?)
    limpio = re.sub(r'(?im)^\s*sql[\s:]+', '', limpio)

    # 🔥 Busca el primer SELECT si todavía hay texto explicativo
    m = re.search(r'(?is)(select\b.+)$', limpio)
    if m:
        limpio = m.group(1)

    # Limpieza final
    return limpio.strip().rstrip(';')


def ejecutar_sql_real(pregunta_usuario: str, hist_text: str):
    st.info("🤖 El agente de datos está traduciendo tu pregunta a SQL...")

    # --- Obtener Esquema ---
    try:
        schema_info = db.get_table_info(table_names=["ventus_bi"])
    except Exception as e:
        st.error(f"Error crítico: No se pudo obtener el esquema de la tabla 'ventus_bi'. {e}")
        schema_info = "Error al obtener esquema. Asume columnas estándar."
    
    # --- Crear Prompt ---
    prompt_con_instrucciones = f"""
    Tu tarea es generar una consulta SQL limpia (SOLO SELECT) para responder la pregunta del usuario, basándote ESTRICTAMENTE en el siguiente esquema de tabla.

    --- ESQUEMA DE LA TABLA 'ventus_bi' ---
    {schema_info}
    --- FIN DEL ESQUEMA ---

    ---
    <<< NUEVA REGLA: SIEMPRE MOSTRAR COP Y USD >>>
    1. Revisa el esquema de arriba. Si hay columnas financieras con versiones `_COP` y `_USD` (o similar), úsalas.
    2. Si la pregunta es sobre un valor monetario (costo, valor, total, facturación), DEBES seleccionar AMBAS columnas (COP y USD) si existen en el esquema.
    3. **IMPORTANTE**: NO INVENTES columnas que no estén en el esquema. Si el usuario pregunta por "facturación" y en el esquema solo existe la columna `Monto_Factura`, usa `SUM(Monto_Factura)`. Si existen `Facturado_COP` y `Facturado_USD`, usa `SUM(Facturado_COP), SUM(Facturado_USD)`.
    ---
    <<< REGLA CRÍTICA PARA FILTRAR POR FECHA >>>
    1. Si en el esquema ves una columna de fecha (ej: `Fecha_Facturacion`), úsala para filtrar.
    2. Si el usuario especifica un año (ej: "del 2025", "en 2024"), SIEMPRE debes añadir una condición `WHERE YEAR(TuColumnaDeFecha) = [año]` a la consulta.
    ---
    <<< REGLA DE ORO PARA BÚSQUEDA DE PRODUCTOS >>>
    1. Si en el esquema hay una columna de producto (ej: `Nombre_Producto`), usa `WHERE LOWER(Nombre_Producto) LIKE '%palabra%'.
    ---
    {hist_text}
    Pregunta del usuario: "{pregunta_usuario}"

    Devuelve SOLO la consulta SQL (sin explicaciones).
    """
    
    try:
        # Llama al LLM directamente para OBTENER el SQL (sin ejecutarlo)
        sql_query_bruta = llm_sql.invoke(prompt_con_instrucciones).content

        st.text_area("🧩 SQL generado por el modelo:", sql_query_bruta, height=100)

        # 🧹 Limpieza robusta del SQL generado
        sql_query_limpia = limpiar_sql(sql_query_bruta)

        # ⚠️ Si aún no empieza con SELECT, intenta extraer la parte válida
        if not sql_query_limpia.lower().startswith("select"):
            m = re.search(r'(?is)(select\b.+)$', sql_query_limpia)
            if m:
                sql_query_limpia = m.group(1).strip()

        # ✅ Asegura que solo sea un SELECT permitido
        sql_query_limpia = _asegurar_select_only(sql_query_limpia)

        # Mostrar resultado final
        st.code(sql_query_limpia, language="sql")

        # 🚀 Ejecutar la consulta SQL
        with st.spinner("⏳ Ejecutando consulta..."):
            with db._engine.connect() as conn:
                df = pd.read_sql(text(sql_query_limpia), conn)

        st.success(f"✅ ¡Consulta ejecutada! Filas: {len(df)}")

        # 🟢 CAMBIO 1: guardar versión sin fila Total
        df_original = df.copy()

        # 🧮 Post-procesamiento
        value_cols = []
        try:
            if not df.empty:
                year_match = re.search(r"YEAR\([^)]*\)\s*=\s*(\d{4})", sql_query_limpia)
                year_value = year_match.group(1) if year_match else None
                if year_value and "Año" not in df.columns:
                    df.insert(0, "Año", year_value)

                value_cols = [
                    c for c in df.select_dtypes("number").columns
                    if not re.search(r"(?i)\b(mes|año|dia|fecha|id|codigo)\b", c) # Excluimos IDs también
                ]

                # --- ⬇️ CORRECCIÓN PARA EL ERROR DE PYARROW ⬇️ ---
                if value_cols and len(df) > 1: # Solo añade Total si hay datos y columnas de valor
                    total_row = {}
                    for col in df.columns:
                        if col in value_cols:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                total_row[col] = df[col].sum()
                            else:
                                total_row[col] = np.nan
                        elif pd.api.types.is_numeric_dtype(df[col]):
                            total_row[col] = np.nan
                        else:
                            total_row[col] = ""
                    
                    total_row[df.columns[0]] = "Total"
                    # Usamos pd.concat en lugar de .loc para evitar advertencias futuras
                    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
                # --- ⬆️ FIN DE LA CORRECCIÓN ⬆️ ---

            # --- ⬇️ INICIO DE LA MODIFICACIÓN DE FORMATO ⬇️ ---

            def highlight_total(row):
                # Esta es la línea que probablemente tenía el error U+00A0
                if isinstance(row.iloc[0], str) and row.iloc[0].lower() == "total":
                    return ["font-weight: bold; background-color: #f8f9fa; border-top: 2px solid #999;"] * len(row)
                else:
                    return [""] * len(row)

            styled_df = df.style.apply(highlight_total, axis=1)

            # 1. Crear mapa de formato base para columnas de valor (miles, 0 decimales)
            format_map = {col: "{:,.0f}" for col in value_cols}

            # 2. Añadir formato específico para 'Mes' (entero, 0 decimales)
            if "Mes" in df.columns:
                format_map["Mes"] = "{:.0f}"

            # 3. (A futuro) Añadir formato para columnas de porcentaje
            percent_cols = [col for col in df.columns if "porcentaje" in col.lower() or "%" in col.lower()]
            for col in percent_cols:
                format_map[col] = "{:,.2f}%" # 2 decimales y el símbolo %

            # 4. Aplicar TODOS los formatos.
            def safe_formatter(fmt):
                def inner(x):
                    try:
                        # Manejar NaN explícitamente
                        if x is None or (isinstance(x, float) and np.isnan(x)):
                            return ""
                        return fmt.format(float(x))
                    except:
                        # Si no se puede formatear (ej: 'Total', ''), devolver tal cual
                        return x
                return inner

            safe_format_map = {col: safe_formatter(fmt) for col, fmt in format_map.items()}
            styled_df = styled_df.format(safe_format_map, na_rep="")
            
            # --- ⬆️ FIN DE LA MODIFICACIÓN DE FORMATO ⬆️ ---

            # ==========================
            # Cálculo de totales relevantes
            # ==========================
            try:
                # 🟢 CAMBIO 2: usar df_original, NO df
                columnas_valor = seleccionar_columnas_valor(df_original)
                totales_dict = {col: df_original[col].sum() for col in columnas_valor}

                texto_totales = "\n".join([f"{c}: {int(v):,}" for c, v in totales_dict.items()]) \
                    if totales_dict else "(no se detectaron columnas de valor para totales)"

                res_totales = {
                    "totales_texto": texto_totales,
                    "totales_dict": totales_dict
                }
            except Exception as e:
                res_totales = {
                    "totales_texto": "(error al calcular totales)",
                    "totales_dict": {}
                }
            
            res = {"sql": sql_query_limpia, "df": df, "styled": styled_df}
            res.update(res_totales)
            return res

        except Exception as e:
            st.warning(f"No se pudo aplicar formato ni totales: {e}")
            # Si falla el estilo, al menos devolvemos los datos crudos
            return {"sql": sql_query_limpia, "df": df}


    except Exception as e:
        st.warning(f"❌ Error en la consulta directa. Intentando método alternativo... Detalle: {e}")
        return {"sql": None, "df": None, "error": str(e)}


def ejecutar_sql_en_lenguaje_natural(pregunta_usuario: str, hist_text: str):
    st.info("🤔 Activando el agente SQL experto como plan B.")
    prompt_sql = (f"Tu tarea es responder la pregunta consultando la tabla 'ventus_bi'.\n{hist_text}\nDevuelve ÚNICAMENTE una tabla en formato Markdown (con encabezados). NUNCA resumas ni expliques. El SQL interno NO DEBE CONTENER 'LIMIT'.\nPregunta: {pregunta_usuario}")
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

def analizar_con_datos(pregunta_usuario: str, hist_text: str, df: pd.DataFrame | None, totales_texto: str = "", totales_dict: dict = None, feedback: str = None):
    st.info("\n🧠 El analista experto está examinando los datos...")

    if df is None or df.empty:
        return "No hay datos para analizar."

    # ---- PREVIEW COMPLETO ----
    try:
        preview = df.to_markdown(index=False)
    except Exception:
        preview = df.head(200).to_string(index=False)

    # ---- BLOQUE DE TOTALES ----
    if not totales_texto:
        totales_texto = "(totales no detectados)"
    if totales_dict is None:
        totales_dict = {}

    # ---- FEEDBACK ----
    correccion = ""
    if feedback:
        correccion = f"""INSTRUCCIÓN DE CORRECCIÓN:
Tu análisis anterior contenía errores.
Feedback: "{feedback}".
Corrige esos errores SIN inventar nada.
"""

    # ---- PROMPT FINAL ----
    prompt_analisis = f"""
{correccion}

Eres IANA, un analista de datos senior extremadamente preciso.
NO puedes inventar valores. TODAS las cifras deben salir exclusivamente de:

1) la tabla de datos completa
2) los totales reales de la base, calculados por el sistema

--- TOTALES REALES ---
{totales_texto}
--- FIN TOTALES ---

--- DATOS COMPLETOS (NO INVENTAR) ---
{preview}
--- FIN DATOS ---

Pregunta del usuario:
"{pregunta_usuario}"

--- FORMATO OBLIGATORIO ---
📌 Resumen Ejecutivo:
- Hallazgos principales con cifras REALES.

🔍 Números de referencia:
- Totales, promedios, máximos, mínimos, % (solo si se derivan de los datos).

⚠ Importante:
- Riesgos o alertas.
- NO inventar. Si falta un dato, dilo.
    """

    with st.spinner("💡 Generando análisis avanzado..."):
        analisis = llm_analista.invoke(prompt_analisis).content

    st.success("💡 ¡Análisis completado!")
    return analisis


def responder_conversacion(pregunta_usuario: str, hist_text: str):
    st.info("💬 Activando modo de conversación...")
    prompt_personalidad = f"""Tu nombre es IANA, una IA amable de Ventus. Ayuda a analizar datos.\nSi el usuario hace un comentario casual, responde amablemente de forma natural, muy humana y redirígelo a tus capacidades.\n{hist_text}\nPregunta: "{pregunta_usuario}" """
    respuesta = llm_analista.invoke(prompt_personalidad).content
    return {"texto": respuesta, "df": None, "analisis": None}

def generar_resumen_tabla(pregunta_usuario: str, res: dict) -> dict:
    st.info("✍️ Generando un resumen introductorio para la tabla...")
    df = res.get("df")
    if df is None or df.empty:
        return res

    # --- INICIO DE LA MODIFICACIÓN DEL PROMPT ---
    prompt = f"""
    Actúa como IANA, un analista de datos amable y servicial.
    Tu tarea es escribir una breve y conversacional introducción para la tabla de datos que estás a punto de mostrar.
    Basa tu respuesta en la pregunta del usuario para que se sienta como una continuación natural de la conversación.
    Si la respuesta no le gustó al USUARIO, disculpate es posible que le entendiste mal.
    
    IMPORTANTE: Varía tus respuestas. No uses siempre la misma frase. Suena natural y humana.

    Pregunta del usuario: "{pregunta_usuario}"
    
    ---
    Aquí tienes varios ejemplos de cómo responder:

    Ejemplo 1:
    Pregunta: "cuáles son los proveedores"
    Respuesta: "¡Listo! Aquí tienes la lista de proveedores que encontré:"

    Ejemplo 2:
    Pregunta: "y sus ventas?"
    Respuesta: "He consultado las cifras de ventas. Te las muestro en la siguiente tabla:"

    Ejemplo 3:
    Pregunta: "y en q % esta su consumo?"
    Respuesta: "Perfecto, aquí está el desglose de los porcentajes de consumo que pediste:"

    Ejemplo 4:
    Pregunta: "dame el total por mes"
    Respuesta: "Claro que sí. He preparado la tabla con los totales por mes:"
    ---

    Ahora, genera la introducción para la pregunta del usuario actual:
    """
    # --- FIN DE LA MODIFICACIÓN DEL PROMPT ---
    try:
        introduccion = llm_analista.invoke(prompt).content
        res["texto"] = introduccion
    except Exception as e:
        st.warning(f"No se pudo generar el resumen introductorio. Error: {e}")
        res["texto"] = "Aquí están los datos que solicitaste:"
    return res


# ============================================
# BLOQUE OML: Gestión de Equipos Externos
# ============================================

@st.cache_resource
def get_connection_oml():
    """Conexión específica para la base de datos de OML (Equipos)."""
    try:
        creds = st.secrets["db_credentials_oml"]
        # Usamos pymysql directo para rapidez o SQLAlchemy
        uri = f"mysql+pymysql://{creds['user']}:{creds['password']}@{creds['host']}/{creds['database']}"
        engine = create_engine(uri, pool_recycle=3600, connect_args={"connect_timeout": 10})
        return engine
    except Exception as e:
        st.error(f"❌ Error conectando a OML: {e}")
        return None

def consultar_equipo_oml(pregunta: str):
    """
    Detecta líderes específicos o la palabra 'todos' para consultar OML.
    """
    mapa_equipos = {
        "andres": "tbl_gen_andres", "andrés": "tbl_gen_andres", "staff 1": "tbl_gen_andres", "grupo 1": "tbl_gen_andres",
        "camilo": "tbl_gen_camilo", "staff 2": "tbl_gen_camilo", "grupo 2": "tbl_gen_camilo",
        "maira": "tbl_gen_maira", "mayra": "tbl_gen_maira", "staf 3": "tbl_gen_maira", "grupo 3": "tbl_gen_maira",
        "mateo": "tbl_gen_mateo", "staff 4": "tbl_gen_mateo", "grupo 4": "tbl_gen_mateo",
        "simon": "tbl_gen_simon", "simón": "tbl_gen_simon", "staff 5": "tbl_gen_simon", "grupo 5": "tbl_gen_simon"
    }

    pregunta_lower = pregunta.lower()
    
    # --- LÓGICA OPCIÓN A: VER TODOS ---
    keywords_todos = ["todos", "general", "global", "resumen de equipos", "toda la flota"]
    if any(k in pregunta_lower for k in keywords_todos):
        st.info("🔎 Recopilando información de TODOS los equipos OML...")
        engine_oml = get_connection_oml()
        if not engine_oml: return None
        
        tablas_unicas = list(set(mapa_equipos.values())) # Evitamos duplicados
        dfs = []
        
        try:
            with engine_oml.connect() as conn:
                for tabla in tablas_unicas:
                    try:
                        sql = f"SELECT * FROM {tabla}"
                        df_temp = pd.read_sql(text(sql), conn)
                        if not df_temp.empty:
                            dfs.append(df_temp)
                    except Exception:
                        continue # Si falla una tabla, seguimos con las otras
            
            if dfs:
                df_final = pd.concat(dfs, ignore_index=True)
                # Ordenamos por Líder para que se vea ordenado
                if "Lider" in df_final.columns:
                    df_final = df_final.sort_values(by="Lider")
                return {
                    "texto": "Aquí tienes el consolidado de **todos los equipos**:",
                    "df": df_final,
                    "origen": "oml"
                }
            else:
                return {"texto": "No encontré datos en ninguna tabla de equipos.", "df": None}
        except Exception as e:
             return {"tipo": "error", "texto": f"Error general consultando todos: {e}"}

    # --- LÓGICA ORIGINAL: UN SOLO LÍDER ---
    tabla_objetivo = None
    nombre_lider = None

    # Buscamos si algún nombre está en la pregunta
    for key, table_name in mapa_equipos.items():
        if key in pregunta_lower:
            tabla_objetivo = table_name
            nombre_lider = key.capitalize()
            break
    
    # Si no encontramos ningún nombre conocido, devolvemos None
    if not tabla_objetivo:
        return None

    st.info(f"🔎 Buscando información del equipo de **{nombre_lider}** en base de datos OML...")

    # 2. Ejecutar la consulta
    engine_oml = get_connection_oml()
    if not engine_oml:
        return {"tipo": "error", "texto": "Fallo de conexión con servidor OML."}

    try:
        # Consulta directa a la tabla detectada
        # Ordenamos por 'Orden_Visual' como se ve en la imagen 2
        sql = f"SELECT * FROM {tabla_objetivo} ORDER BY Orden_Visual ASC"
        
        with engine_oml.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        
        if df.empty:
            return {"texto": f"El equipo de {nombre_lider} no tiene datos registrados actualmente.", "df": None}

        # 3. Preparar respuesta
        # Nota: Devolvemos el DF para que el orquestador lo pase por tu función de estilos (Totales, etc.)
        return {
            "texto": f"Aquí tienes el reporte actual del **Grupo {nombre_lider}**:",
            "df": df,
            "origen": "oml" # Marca interna para saber de dónde vino
        }

    except Exception as e:
        return {"tipo": "error", "texto": f"Error consultando tabla {tabla_objetivo}: {str(e)}"}


# ============================================
# 4) Orquestador y Validación
# ============================================
def clasificar_intencion(pregunta: str) -> str:
    prompt_orq = f"""
Clasifica la intención del usuario en UNA SOLA PALABRA: `consulta`, `analista`, `correo` o `conversacional`.

Reglas:
1. `analista`: si el usuario pide interpretación, resumen, comparación o explicación.
   PALABRAS CLAVE: analiza, compara, resume, explica, por qué, tendencia, insights, interpretación, conclusiones.
2. `consulta`: si el usuario pide ver datos, cifras, totales, listados o información específica de una base de datos.
   PALABRAS CLAVE: total, valor, ventas, facturación, consumo, costo, proveedores, productos, mes, año, lista, dime, dame, cuántos, muéstrame.
   Si la pregunta contiene una fecha o número de año (por ejemplo, 2023, 2024, 2025), clasifícala como `consulta`.
3. `correo`: si menciona enviar, mandar, correo, email o reporte.
4. `conversacional`: si es un saludo, agradecimiento o comentario general (hola, gracias, quién eres, qué haces, cómo estás).

Pregunta: "{pregunta}"
Clasificación:
"""
    try:
        opciones = {"consulta", "analista", "conversacional", "correo"}
        r = llm_orq.invoke(prompt_orq).content.strip().lower().replace('"', '').replace("'", "")
        return r if r in opciones else "consulta"  # 👈 Fallback seguro a 'consulta'
    except Exception:
        return "consulta"

def obtener_datos_sql(pregunta_usuario: str, hist_text: str) -> dict:
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
    with st.expander("⚙️ Ver Proceso de IANA", expanded=False):

        # 1. Clasificamos la intención PRIMERO (para saber si quiere análisis)
        hist_text = get_history_text(chat_history)
        clasificacion = clasificar_intencion(pregunta_usuario)

        # 2. Revisión de Equipos OML
        res_oml = consultar_equipo_oml(pregunta_usuario)
                        
        if res_oml is not None:
            st.success("✅ Intención detectada: Consulta de Equipo OML (Base Externa).")

            # Aplicamos estilos y cálculos de totales
            if isinstance(res_oml.get("df"), pd.DataFrame) and not res_oml["df"].empty:
                try:
                    # Pasando el resultado OML como si fuera SQL normal 
                    res_con_estilo = interpretar_resultado_sql(res_oml)

                    # Si la intención es "analista", le pasamos los datos al LLM
                    if clasificacion == "analista":
                        st.info("🧠 IANA está analizando el rendimiento del equipo...")
                        analisis = analizar_con_datos(
                            pregunta_usuario,
                            hist_text,
                            res_con_estilo.get("df"), # Usamos el DF limpio
                            res_con_estilo.get("totales_texto", ""),
                            res_con_estilo.get("totales_dict", {}),
                            feedback=None
                        )
                        res_con_estilo["analisis"] = analisis
                    
                    return res_con_estilo
                except Exception as e:
                    st.warning(f"No pude aplicar estilos avanzados: {e}")
                    return res_oml
            return res_oml

        # 3. Flujo Normal (Ventus BI)
        st.success(f"✅ Intención detectada: {clasificacion.upper()}.")

        if clasificacion == "conversacional":
            return responder_conversacion(pregunta_usuario, hist_text)
        
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

            detalles = extraer_detalles_correo(pregunta_usuario)
            return enviar_correo_agente(
                recipient=detalles["recipient"],
                subject=detalles["subject"],
                body=detalles["body"],
                df=df_para_enviar
            )

        res_datos = obtener_datos_sql(pregunta_usuario, hist_text)
        
        if res_datos.get("df") is None or res_datos["df"].empty:
            return {"tipo": "error", "texto": "Lo siento, no pude obtener datos para tu pregunta. Intenta reformularla."}

        if clasificacion == "consulta":
            st.success("✅ Consulta directa completada.")
            # Primero, intentamos interpretar el resultado como siempre
            res_interpretado = interpretar_resultado_sql(res_datos)
    
            # Luego, si no se generó texto (porque es una tabla), creamos la introducción
            if res_interpretado.get("texto") is None and res_interpretado.get("df") is not None and not res_interpretado["df"].empty:
                res_interpretado = generar_resumen_tabla(pregunta_usuario, res_interpretado)
    
            return res_interpretado

        if clasificacion == "analista":
            st.info("🧠 Generando análisis inicial...")
            # 💡 Eliminación del supervisor: Ya no se llama a validar_y_corregir_respuesta_analista.
            # Se genera el análisis y se devuelve inmediatamente.
            res_datos["analisis"] = analizar_con_datos(
                pregunta_usuario,
                hist_text,
                res_datos.get("df"),
                res_datos.get("totales_texto", ""),
                res_datos.get("totales_dict", {}),
                feedback=None
            )
            return res_datos # 👈 CAMBIO CLAVE: Devolver el resultado inmediatamente sin validación.
            # return validar_y_corregir_respuesta_analista(pregunta_usuario, res_datos, hist_text) # 👈 Línea eliminada

# ============================================
# 5) Interfaz: Micrófono en vivo + Chat
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": {"texto": "¡Hola! Soy IANA, tu asistente de IA de Ventus. ¿Qué te gustaría saber?"}}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message.get("content", {});
        if isinstance(content, dict):
            if content.get("texto"): st.markdown(content["texto"])
            if isinstance(content.get("df"), pd.DataFrame) and not content["df"].empty: st.dataframe(content["df"])
            if content.get("analisis"): st.markdown(content["analisis"])
        elif isinstance(content, str): st.markdown(content)

st.markdown("### 🎤 Habla con IANA o escribe tu pregunta")
lang = st.secrets.get("stt_language", "es-CO")

# Unificar el procesamiento de la pregunta
def procesar_pregunta(prompt):
    if prompt:
        if not all([db, llm_sql, llm_analista, llm_orq, agente_sql, llm_validador]):
            st.error("La aplicación no está completamente inicializada. Revisa los errores de conexión o de API key.")
            return

        st.session_state.messages.append({"role": "user", "content": {"texto": prompt}})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            res = orquestador(prompt, st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": res})

            # ... (dentro de procesar_pregunta)

            if res and res.get("tipo") != "error":
                # La línea 666 es esta:
                if res.get("texto"): st.markdown(res["texto"])

            # --- ⬇️ INICIO DE LA MODIFICACIÓN ⬇️ ---
            # Revisa si existe la versión "styled" (con formato)
                if res.get("styled") is not None:
                    st.dataframe(res["styled"])
            # Si no, muestra la versión "cruda" (df)
                elif isinstance(res.get("df"), pd.DataFrame) and not res["df"].empty:
                    st.dataframe(res["df"])
            # --- ⬆️ FIN DE LA MODIFICACIÓN ⬆️ ---

                if res.get("analisis"):
                     st.markdown("---"); st.markdown("### 🧠 Análisis de IANA"); st.markdown(res["analisis"])
                     st.toast("Análisis generado ✅", icon="✅")
            elif res: # <-- El error también podría estar en la indentación de esta línea
                 st.error(res.get("texto", "Ocurrió un error inesperado."))
                 st.toast("Hubo un error ❌", icon="❌")
                 
# Contenedor para los inputs
input_container = st.container()
with input_container:
    col1, col2 = st.columns([1, 4])
    with col1:
        voice_text = speech_to_text(language=lang, start_prompt="🎙️ Hablar", stop_prompt="🛑 Detener", use_container_width=True, just_once=True, key="stt")
    with col2:
        prompt_text = st.chat_input("... o escribe tu pregunta aquí")

# Determinar qué prompt usar
prompt_a_procesar = None
if voice_text:
    prompt_a_procesar = voice_text
elif prompt_text:
    prompt_a_procesar = prompt_text

# Procesar el prompt si existe
if prompt_a_procesar:
    procesar_pregunta(prompt_a_procesar)
    




