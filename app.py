import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pdf2image import convert_from_bytes
from io import BytesIO


st.set_page_config(page_title="Formations en déblai", page_icon="📊")
st.title("📊 Pourcentage des formations en déblai")


def load_image(file):
    data = file.read()
    name = file.name.lower()

    if name.endswith(".pdf"):
        return convert_from_bytes(data, dpi=250)[0].convert("RGB")

    return Image.open(BytesIO(data)).convert("RGB")


def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def color_mask_rgb(img, rgb, tolerance):
    arr = np.array(img).astype(np.int16)
    target = np.array(rgb).astype(np.int16)
    dist = np.sqrt(np.sum((arr - target) ** 2, axis=2))
    return (dist <= tolerance).astype(np.uint8) * 255


def line_y_by_x(mask):
    h, w = mask.shape
    y_line = np.full(w, -1, dtype=int)

    for x in range(w):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) > 0:
            y_line[x] = int(np.median(ys))

    valid = y_line >= 0

    if valid.sum() < 20:
        return y_line

    return np.interp(
        np.arange(w),
        np.where(valid)[0],
        y_line[valid]
    ).astype(int)


def build_deblai_mask(img, project_rgb, tn_rgb, tol_project, tol_tn):
    arr = np.array(img)
    h, w = arr.shape[:2]

    project_mask = color_mask_rgb(img, project_rgb, tol_project)
    tn_mask = color_mask_rgb(img, tn_rgb, tol_tn)

    kernel = np.ones((5, 5), np.uint8)
    project_mask = cv2.morphologyEx(project_mask, cv2.MORPH_CLOSE, kernel)
    tn_mask = cv2.morphologyEx(tn_mask, cv2.MORPH_CLOSE, kernel)

    y_project = line_y_by_x(project_mask)
    y_tn = line_y_by_x(tn_mask)

    mask = np.zeros((h, w), dtype=np.uint8)

    for x in range(w):
        yp = y_project[x]
        yt = y_tn[x]

        if yp > 0 and yt > 0 and yt < yp:
            mask[yt:yp, x] = 255

    return mask


def classify_formations(img, deblai_mask):
    arr = np.array(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    zone = deblai_mask > 0
    assigned = np.zeros(zone.shape, dtype=bool)

    masks = {}

    # Rouge hachuré : argiles / limons / tufs
    red = zone & (((h <= 8) | (h >= 170)) & (s > 70) & (v > 40))
    red = cv2.dilate(red.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=2) > 0
    red = cv2.bitwise_and(red.astype(np.uint8), zone.astype(np.uint8)).astype(bool)
    masks["Argiles / limons / tufs"] = red
    assigned |= red

    # Rose : marnes
    pink = zone & ~assigned & (h >= 135) & (h <= 165) & (s > 50) & (v > 90)
    masks["Marnes et argiles marneuses"] = pink
    assigned |= pink

    # Jaune : grès
    yellow = zone & ~assigned & (h >= 22) & (h <= 38) & (s > 80) & (v > 120)
    masks["Grès"] = yellow
    assigned |= yellow

    # Marron : basaltes
    brown = zone & ~assigned & (h >= 8) & (h <= 25) & (s > 80) & (v > 40) & (v < 210)
    masks["Basaltes"] = brown
    assigned |= brown

    # Gris : schistes sains
    gray = zone & ~assigned & (s < 45) & (v > 50) & (v < 190)
    masks["Schistes sains"] = gray
    assigned |= gray

    rows = []
    total = sum(int(m.sum()) for m in masks.values())

    for name, mask in masks.items():
        pixels = int(mask.sum())
        pct = round(pixels / total * 100, 2) if total else 0

        rows.append({
            "Formation": name,
            "Pixels": pixels,
            "Pourcentage (%)": pct
        })

    return pd.DataFrame(rows).sort_values("Pourcentage (%)", ascending=False), masks


def make_overlay(img, deblai_mask):
    arr = np.array(img).copy()
    overlay = arr.copy()

    zone = deblai_mask > 0
    overlay[zone] = (
        overlay[zone] * 0.45 + np.array([255, 255, 0]) * 0.55
    ).astype(np.uint8)

    return Image.fromarray(overlay)


uploaded = st.file_uploader(
    "Importer le profil découpé",
    type=["png", "jpg", "jpeg", "pdf"]
)

st.sidebar.header("Paramètres")

project_color = st.sidebar.color_picker(
    "Couleur ligne projet",
    value="#0000ff"
)

tn_color = st.sidebar.color_picker(
    "Couleur TN",
    value="#00ff00"
)

tol_project = st.sidebar.slider("Tolérance ligne projet", 5, 120, 45)
tol_tn = st.sidebar.slider("Tolérance TN", 5, 120, 45)

if uploaded:
    img = load_image(uploaded)

    st.image(img, caption="Image importée", use_container_width=True)

    if st.button("Calculer"):
        deblai_mask = build_deblai_mask(
            img,
            hex_to_rgb(project_color),
            hex_to_rgb(tn_color),
            tol_project,
            tol_tn
        )

        if deblai_mask.sum() == 0:
            st.error("Zone déblai non détectée. Vérifie couleur ligne projet et TN.")
        else:
            df, masks = classify_formations(img, deblai_mask)

            st.success("Calcul terminé ✔️")
            st.dataframe(df)

            st.image(
                make_overlay(img, deblai_mask),
                caption="Zone analysée : entre TN vert et ligne projet",
                use_container_width=True
            )

            output = BytesIO()

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Formations_deblai", index=False)

            output.seek(0)

            st.download_button(
                "📥 Télécharger Excel",
                data=output,
                file_name="pourcentage_formations_deblai.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
