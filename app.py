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
    
    # << VERSIÓN FINAL DEL PROMPT >>
    # Ahora que los campos son DECIMAL, eliminamos la regla de CAST. El prompt es más limpio.
    prompt_con_instrucciones = f"""
    Tu tarea es generar una consulta SQL **únicamente contra la tabla 'ventus'** para responder la pregunta del usuario.
    
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
    - `Subtotal_COP` (DECIMAL) <- Numérico. Listo para sumar.
    - `IVA_COP` (DECIMAL) <- Numérico. Listo para sumar.
    - `Total_COP` (DECIMAL) <- Numérico. Métrica clave de costo.
    - `SubTotal_USD` (DECIMAL) <- Numérico. Listo para sumar.
    - `IVA_USD` (DECIMAL) <- Numérico. Listo para sumar.
    - `Total_USD` (DECIMAL) <- Numérico. Métrica clave de costo.
    - `Comentarios` (Texto)
    - `Condicion_de_pago` (Texto)
    - `Condiciones_Comerciales` (Texto)
    - `Comprador` (Texto, Categoría para agrupar)
    - `Cantidad` (DECIMAL) <- Numérico. Listo para sumar.
    - `Cluster` (Texto, Categoría)
    - `Producto` (Texto, Aquí está el campo más importante donde se indentifica el trabajo realizado en el proyecto, ej: Transporte de materiales, Bultos de Cemento, Bolsas de basura, etc...)
    - `Grupo_Producto` (Texto, Categoría para agrupar)
    - `Familia` (Texto, Categoría para agrupar)
    - `Tipo` (Texto, Categoría para agrupar)

    REGLA 3 (Agrupación): Presta mucha atención a palabras como 'diariamente' (GROUP BY Fecha_aprobacion), 'mensual' (GROUP BY MONTH(Fecha_aprobacion), YEAR(Fecha_aprobacion)), 'por Tipo', 'por Proveedor', 'por Comprador', 'por Familia', etc.
    REGLA 4 (LIMIT): Nunca agregues un 'LIMIT'.
    
    Pregunta original: "{pregunta_usuario}"
    """
    try:
        query_chain = create_sql_query_chain(llm_sql, db)
        sql_query = query_chain.invoke({"question": prompt_con_instrucciones})
        
        sql_query = re.sub(r"^\s*```sql\s*|\s*```\s*$", "", sql_query, flags=re.IGNORECASE).strip()

        st.code(sql_query, language='sql')
        with st.spinner("⏳ Ejecutando la consulta en la base de datos..."):
            with db._engine.connect() as conn:
                df = pd.read_sql(text(sql_query), conn)
        st.success("✅ ¡Consulta ejecutada!")
        return {"sql": sql_query, "df": df}
    except Exception as e:
        st.warning(f"❌ Error en la consulta directa. Intentando un método alternativo... Error: {e}")
        return {"sql": None, "df": None, "error": str(e)}


def ejecutar_sql_en_lenguaje_natural(pregunta_usuario: str):
    st.info("🤔 La consulta directa falló. Activando el agente SQL experto de IANA como plan B.")
    
    prompt_sql = (
        "Tu tarea es responder la pregunta del usuario consultando la base de datos. "
        "La tabla se llama 'ventus'. " # Añadimos contexto extra por si acaso
        "Debes devolver ÚNICAMENTE una tabla de datos en formato Markdown. "
        "REGLA CRÍTICA: Devuelve SIEMPRE TODAS las filas de datos que encuentres. NUNCA resumas, trunques ni expliques los resultados. No agregues texto como 'Se muestran las 10 primeras filas' o 'Aquí está la tabla'. "
        "REGLA NUMÉRICA: Columnas como Total_COP, Total_USD y Cantidad ya son numéricas (DECIMAL) y se pueden sumar directamente. "
        "Responde siempre en español. "
        "Pregunta del usuario: "
        f"{pregunta_usuario}"
    )
    try:
        with st.spinner("💬 Pidiendo al agente SQL que responda en lenguaje natural..."):
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
    
    prompt_analisis = f"""
    Tu nombre es IANA. Eres un analista de datos senior en Ventus, experto en proyectos de infraestructura, energía e industriales.
    Tu tarea es generar un análisis ejecutivo, breve y fácil de leer para un gerente de proyecto o director de área.
    Responde siempre en español.

    REGLAS DE FORMATO MUY IMPORTANTES:
    1.  Inicia con el título: "Análisis Ejecutivo de Datos para Ventus".
    2.  Debajo del título, presenta tus conclusiones como una lista de ítems (viñetas con markdown `-`).
    3.  Cada ítem debe ser una oración corta, clara y directa al punto.
    4.  Limita el análisis a un máximo de 5 ítems clave; si el cliente especifica una cantidad de ítems, genera el número exacto que pidió.
    5.  No escribas párrafos largos.

    Pregunta del usuario: {pregunta_usuario}
    Datos disponibles para tu análisis (columnas clave: Total_COP, Total_USD, Proveedor, Comprador, Tipo, Familia, Fecha_aprobacion):
    {_df_preview(df, 20)}

    Ahora, genera el análisis siguiendo estrictamente las reglas de formato.
    """
    with st.spinner("💡 Generando análisis y recomendaciones..."):
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

    Pregunta del usuario: "{pregunta_usuario}"
    """
    respuesta = llm_analista.invoke(prompt_personalidad).content
    # Usamos la clave "texto" para la respuesta principal y "analisis" como nulo.
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

def obtener_datos_sql(pregunta_usuario: str) -> dict:
    res_real = ejecutar_sql_real(pregunta_usuario)
    if res_real.get("df") is not None and not res_real["df"].empty:
        return {"sql": res_real["sql"], "df": res_real["df"], "texto": None}
    
    # Si la consulta directa falla (quizás por un error de SQL), probamos el agente general
    res_nat = ejecutar_sql_en_lenguaje_natural(pregunta_usuario)
    return {"sql": None, "df": res_nat["df"], "texto": res_nat["texto"]}


def orquestador(pregunta_usuario: str, chat_history: list):
    with st.expander("⚙️ Ver Proceso de IANA", expanded=False):
        st.info(f"🚀 Recibido: '{pregunta_usuario}'")
        with st.spinner("🔍 IANA está analizando tu pregunta..."):
            clasificacion = clasificar_intencion(pregunta_usuario)
        st.success(f"✅ ¡Intención detectada! Tarea: {clasificacion.upper()}.")

        if clasificacion == "conversacional":
            return responder_conversacion(pregunta_usuario)

        # --- Lógica de Memoria para Análisis de Contexto ---
        if clasificacion == "analista":
            palabras_clave_contexto = [
                "esto", "esos", "esa", "información", 
                "datos", "tabla", "anterior", "acabas de dar"
            ]
            es_pregunta_de_contexto = any(palabra in pregunta_usuario.lower() for palabra in palabras_clave_contexto)

            if es_pregunta_de_contexto and len(chat_history) > 1:
                mensaje_anterior = chat_history[-2] # El penúltimo mensaje (la respuesta previa de IANA)
                
                if mensaje_anterior["role"] == "assistant" and "df" in mensaje_anterior["content"]:
                    df_contexto = mensaje_anterior["content"]["df"]
                    
                    if df_contexto is not None and not df_contexto.empty:
                        st.info("💡 Usando datos de la conversación anterior para el análisis...")
                        analisis = analizar_con_datos(pregunta_usuario, "Datos de la tabla anterior.", df_contexto)
                        # Devolvemos la tabla anterior junto con el nuevo análisis
                        return {"tipo": "analista", "df": df_contexto, "texto": None, "analisis": analisis}
        # --- Fin Lógica de Memoria ---

        # Flujo normal (Consulta o Análisis nuevo)
        res_datos = obtener_datos_sql(pregunta_usuario)
        resultado = {"tipo": clasificacion, **res_datos, "analisis": None}
        
        if clasificacion == "analista":
            if res_datos.get("df") is not None and not res_datos["df"].empty:
                analisis = analizar_con_datos(pregunta_usuario, res_datos.get("texto", ""), res_datos["df"])
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





