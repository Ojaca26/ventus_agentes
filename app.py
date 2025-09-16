import streamlit as st
import cv2
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import io
import google.generativeai as genai
import json
import re
import face_recognition

# --- CONFIGURACIÓN ---
ARCHIVO_EXCEL = 'datos_cedulas_colombia.xlsx'

# Configurar la API de Gemini (la clave se toma de st.secrets)
try:
    api_key = st.secrets.get("GEMINI_API_KEY")
except Exception:
    api_key = None

GEMINI_CONFIGURADO = bool(api_key)
if GEMINI_CONFIGURADO:
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.warning(f"No se pudo configurar Gemini: {e}")
        GEMINI_CONFIGURADO = False
else:
    st.info("⚠️ Gemini no configurado. Podrás cargar y comparar imágenes, pero no extraer datos con IA.")

# ---------- UTILIDADES DE IMAGEN ----------

def fix_orientation(img_pil: Image.Image) -> Image.Image:
    """Corrige orientación EXIF y retorna RGB."""
    img = ImageOps.exif_transpose(img_pil)
    return img.convert("RGB")

def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR"""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    """OpenCV BGR -> PIL RGB"""
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def enhance_for_ocr(img_cv: np.ndarray) -> np.ndarray:
    """Ligero realce para texto/fotografía ID."""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    sharp = cv2.GaussianBlur(cl, (0,0), 1.0)
    unsharp = cv2.addWeighted(cl, 1.5, sharp, -0.5, 0)  # unsharp masking suave
    return cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR)

def reordenar_puntos(puntos):
    rect = np.zeros((4, 2), dtype="float32")
    s = puntos.sum(axis=1)
    rect[0] = puntos[np.argmin(s)]
    rect[2] = puntos[np.argmax(s)]
    diff = np.diff(puntos, axis=1)
    rect[1] = puntos[np.argmin(diff)]
    rect[3] = puntos[np.argmax(diff)]
    return rect

def corregir_perspectiva(imagen_pil: Image.Image) -> Image.Image:
    """Detecta el borde del carnet y corrige perspectiva."""
    try:
        img = pil_to_cv(imagen_pil)
        total_image_area = img.shape[0] * img.shape[1]
        min_area_ratio = 0.08  # 8% del área total (ligeramente más permisivo)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        img_canny = cv2.Canny(img_blur, 60, 180)

        contornos, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:8]

        carnet_contour = None
        for contorno in contornos:
            perimetro = cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
            if len(approx) == 4 and cv2.contourArea(approx) > min_area_ratio * total_image_area:
                carnet_contour = approx
                break

        if carnet_contour is None:
            st.warning("No se detectó un contorno claro de tarjeta. Se usa la imagen completa.")
            return imagen_pil

        pts_src = reordenar_puntos(carnet_contour.reshape(4, 2))
        wA = np.linalg.norm(pts_src[0] - pts_src[1])
        wB = np.linalg.norm(pts_src[2] - pts_src[3])
        hA = np.linalg.norm(pts_src[0] - pts_src[3])
        hB = np.linalg.norm(pts_src[1] - pts_src[2])
        width = int(max(wA, wB))
        height = int(max(hA, hB))

        dst = np.float32([[0,0],[width,0],[width,height],[0,height]])
        M = cv2.getPerspectiveTransform(pts_src, dst)
        warped = cv2.warpPerspective(img, M, (width, height))

        # Realce suave (ayuda a OCR/foto)
        warped = enhance_for_ocr(warped)
        return cv_to_pil(warped)

    except Exception as e:
        st.error(f"Error en la corrección de perspectiva: {e}")
        return imagen_pil

# ---------- GEMINI: EXTRACCIÓN ESTRUCTURADA ----------

def extraer_datos_con_gemini(imagenes_pil):
    """
    Envía 1+ imágenes a Gemini y fuerza respuesta JSON.
    Si no está configurado, retorna dict de error.
    """
    if not GEMINI_CONFIGURADO:
        return {"Error": "API de Gemini no configurada.", "es_cedula_colombiana": False}

    # Modelo rápido para visión + JSON
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={
            "temperature": 0.1,
            "response_mime_type": "application/json"
        }
    )

    prompt = (
        "Eres experto analizando Cédula de Ciudadanía de Colombia. "
        "Analiza las imágenes proporcionadas. "
        "Responde SOLO un JSON con estas claves exactamente:\n"
        "{\n"
        "  \"es_cedula_colombiana\": boolean,\n"
        "  \"NUIP\": string,\n"
        "  \"Apellidos\": string,\n"
        "  \"Nombres\": string,\n"
        "  \"Fecha de nacimiento\": string,\n"
        "  \"Lugar de nacimiento\": string,\n"
        "  \"Estatura\": string,\n"
        "  \"Sexo\": string,\n"
        "  \"GS RH\": string,\n"
        "  \"Fecha y lugar de expedición\": string\n"
        "}\n"
        "Si no es cédula colombiana, responde: {\"es_cedula_colombiana\": false}.\n"
        "Si algún campo no aparece, usa exactamente \"No encontrado\"."
    )

    parts = [prompt] + imagenes_pil
    try:
        resp = model.generate_content(parts)
        # Con response_mime_type=application/json, resp.text DEBE ser JSON puro
        return json.loads(resp.text)
    except Exception as e:
        st.error(f"Error al procesar respuesta de Gemini: {e}")
        # Fallback suave: intenta “pescar” JSON si vino texto mezclado
        try:
            text = getattr(resp, "text", "") if 'resp' in locals() else ""
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass
        return {"Error": "Fallo en la comunicación con la IA.", "es_cedula_colombiana": False}

# ---------- VERIFICACIÓN FACIAL ----------

def crop_largest_face(img_pil: Image.Image, pad_ratio: float = 0.25) -> Image.Image | None:
    """
    Encuentra el rostro más grande y devuelve un recorte ampliado.
    Retorna None si no hay rostros.
    """
    np_img = np.array(img_pil)  # RGB
    boxes = face_recognition.face_locations(np_img, model="hog")  # 'cnn' si tienes CUDA
    if not boxes:
        return None

    # face_locations -> (top, right, bottom, left)
    # Seleccionar el rostro con mayor área
    def area(b): 
        t, r, btm, l = b
        return max(0, btm - t) * max(0, r - l)

    box = max(boxes, key=area)
    t, r, btm, l = box
    h, w = np_img.shape[:2]

    # Padding proporcional
    pad_y = int((btm - t) * pad_ratio)
    pad_x = int((r - l) * pad_ratio)
    t2 = max(0, t - pad_y)
    b2 = min(h, btm + pad_y)
    l2 = max(0, l - pad_x)
    r2 = min(w, r + pad_x)

    face = np_img[t2:b2, l2:r2]
    if face.size == 0:
        return None
    return Image.fromarray(face)

def comparar_rostros(img_cedula_pil: Image.Image, img_selfie_pil: Image.Image, tolerance: float = 0.6):
    """Compara rostros retornando mensaje, booleano y distancia."""
    try:
        # Extrae rostro de la cédula (recorte)
        cedula_face = crop_largest_face(img_cedula_pil)
        if cedula_face is None:
            return "No se encontró un rostro en la cédula.", False, None

        # Rostro de la selfie (usa el más grande)
        selfie_face = crop_largest_face(img_selfie_pil)
        if selfie_face is None:
            return "No se encontró un rostro en la selfie.", False, None

        # Encodings
        enc_ced = face_recognition.face_encodings(np.array(cedula_face))
        enc_self = face_recognition.face_encodings(np.array(selfie_face))

        if not enc_ced:
            return "No fue posible codificar el rostro de la cédula.", False, None
        if not enc_self:
            return "No fue posible codificar el rostro de la selfie.", False, None

        # Distancia y match
        distances = face_recognition.face_distance([enc_ced[0]], enc_self[0])
        dist = float(distances[0])
        matches = face_recognition.compare_faces([enc_ced[0]], enc_self[0], tolerance=tolerance)
        ok = bool(matches[0])

        if ok:
            msg = f"✅ Verificación Exitosa: rostros coinciden (distancia {dist:.3f} ≤ tol {tolerance})."
        else:
            msg = f"❌ Verificación Fallida: no coinciden (distancia {dist:.3f} > tol {tolerance})."

        return msg, ok, dist

    except Exception as e:
        return f"Ocurrió un error durante la comparación facial: {e}", False, None

# ---------- INTERFAZ STREAMLIT ----------

st.set_page_config(page_title="Verificación de Identidad IA", layout="wide")
st.title("🚀 Verificación de Identidad con IA (Gemini)")

# Estado
if 'stage' not in st.session_state:
    st.session_state.stage = 'inicio'
if 'datos_capturados' not in st.session_state:
    st.session_state.datos_capturados = []

if 'anverso_buffer' not in st.session_state: st.session_state.anverso_buffer = None
if 'selfie_buffer' not in st.session_state: st.session_state.selfie_buffer = None
if 'cedula_corregida' not in st.session_state: st.session_state.cedula_corregida = None
if 'datos_cedula' not in st.session_state: st.session_state.datos_cedula = {}

def limpiar_y_empezar_de_nuevo():
    st.session_state.stage = 'inicio'
    st.session_state.anverso_buffer = None
    st.session_state.selfie_buffer = None
    st.session_state.cedula_corregida = None
    st.session_state.datos_cedula = {}

# --- PESTAÑAS DE ENTRADA ---
st.info("Paso 1: Proporciona la imagen de la cédula.")
tab1, tab2 = st.tabs(["📸 Tomar Foto", "⬆️ Subir Foto"])

if st.session_state.stage == 'inicio':
    with tab1:
        anverso_cam = st.camera_input("Toma una foto del **Anverso**", key="cam_anverso")
        if anverso_cam:
            st.session_state.anverso_buffer = anverso_cam
            st.session_state.stage = 'procesar_cedula'
            st.rerun()
    with tab2:
        anverso_up = st.file_uploader("Sube una foto del **Anverso**", type=['jpg', 'jpeg', 'png'], key="up_anverso")
        if anverso_up:
            st.session_state.anverso_buffer = anverso_up
            st.session_state.stage = 'procesar_cedula'
            st.rerun()

# --- PROCESAR CÉDULA ---
if st.session_state.stage == 'procesar_cedula':
    st.info("Procesando cédula...")
    img_anverso_pil = Image.open(st.session_state.anverso_buffer)
    img_anverso_pil = fix_orientation(img_anverso_pil)
    st.session_state.cedula_corregida = corregir_perspectiva(img_anverso_pil)

    if GEMINI_CONFIGURADO:
        with st.spinner('La IA está analizando la cédula...'):
            st.session_state.datos_cedula = extraer_datos_con_gemini([st.session_state.cedula_corregida])
    else:
        st.session_state.datos_cedula = {"es_cedula_colombiana": None, "Nota": "Gemini no configurado"}

    st.session_state.stage = 'mostrar_resultados_cedula'
    st.rerun()

# --- RESULTADOS + SELFIE ---
if st.session_state.stage == 'mostrar_resultados_cedula':
    datos = st.session_state.get('datos_cedula', {})
    st.subheader("Paso 1: Datos extraídos de la cédula")

    # Mostrar imagen corregida
    st.image(st.session_state.cedula_corregida, caption="Cédula (corregida)", use_container_width=True)

    # Mostrar JSON si está disponible
    if datos:
        st.json(datos)

    # Validación para continuar
    puede_continuar = (
        not GEMINI_CONFIGURADO or
        (isinstance(datos, dict) and datos.get("es_cedula_colombiana") is not False)
    )

    if puede_continuar:
        st.info("Paso 2: Verificación facial - compara tu rostro con la foto de la cédula.")
        selfie_cam = st.camera_input("Toma una selfie para la comparación", key="cam_selfie")
        if selfie_cam:
            st.session_state.selfie_buffer = selfie_cam
            st.session_state.stage = 'procesar_selfie'
            st.rerun()
    else:
        st.error("EL DOCUMENTO NO PARECE SER UNA CÉDULA DE CIUDADANÍA DE COLOMBIA.")
        st.button("Empezar de Nuevo", on_click=limpiar_y_empezar_de_nuevo)

# --- PROCESAR SELFIE / COMPARACIÓN ---
if st.session_state.stage == 'procesar_selfie':
    st.subheader("Paso 2: Verificación facial - resultados")

    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.cedula_corregida, caption="Cédula", use_container_width=True)
    with col2:
        st.image(st.session_state.selfie_buffer, caption="Selfie", use_container_width=True)

    selfie_pil = fix_orientation(Image.open(st.session_state.selfie_buffer))

    with st.spinner("Comparando rostros..."):
        mensaje, exito, distancia = comparar_rostros(
            st.session_state.cedula_corregida,
            selfie_pil,
            tolerance=0.6
        )

    if exito:
        st.success(mensaje)
        st.balloons()
        if st.button("Confirmar y Guardar Registro"):
            registro_final = dict(st.session_state.datos_cedula) if isinstance(st.session_state.datos_cedula, dict) else {}
            registro_final["verificacion_facial"] = "Exitosa"
            if distancia is not None:
                registro_final["distancia_rostros"] = round(distancia, 4)
            st.session_state.datos_capturados.append(registro_final)
            st.success("¡Registro guardado!")
            limpiar_y_empezar_de_nuevo()
            st.rerun()
    else:
        st.error(mensaje)

    st.button("Intentar de Nuevo", on_click=limpiar_y_empezar_de_nuevo)

# --- LISTADO REGISTROS / DESCARGA ---
if st.session_state.datos_capturados:
    st.subheader("Registros Verificados")
    df = pd.DataFrame(st.session_state.datos_capturados)
    st.dataframe(df, use_container_width=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Datos')
    excel_data = output.getvalue()

    st.download_button(
        label="📥 Descargar todo como Excel",
        data=excel_data,
        file_name=ARCHIVO_EXCEL,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
