import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pdf2image import convert_from_bytes
from io import BytesIO


st.set_page_config(page_title="Formations en déblai", page_icon="📊")
st.title("📊 Pourcentage des formations en déblai")


def load_image(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".pdf"):
        pages = convert_from_bytes(data, dpi=200)
        return pages[0].convert("RGB")

    return Image.open(BytesIO(data)).convert("RGB")


def detect_blue_line(img):
    arr = np.array(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    lower = np.array([95, 70, 40])
    upper = np.array([135, 255, 255])

    return cv2.inRange(hsv, lower, upper)


def detect_green_tn(img):
    arr = np.array(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    lower = np.array([40, 60, 40])
    upper = np.array([85, 255, 255])

    return cv2.inRange(hsv, lower, upper)


def line_y_by_x(mask):
    h, w = mask.shape
    ys_line = np.full(w, -1, dtype=int)

    for x in range(w):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) > 0:
            ys_line[x] = int(np.median(ys))

    valid = ys_line >= 0

    if valid.sum() < 20:
        return ys_line

    xs = np.where(valid)[0]
    ys = ys_line[valid]

    return np.interp(np.arange(w), xs, ys).astype(int)


def build_deblai_mask(img):
    h, w = np.array(img).shape[:2]

    blue = detect_blue_line(img)
    green = detect_green_tn(img)

    y_blue = line_y_by_x(blue)
    y_green = line_y_by_x(green)

    mask = np.zeros((h, w), dtype=np.uint8)

    for x in range(w):
        yb = y_blue[x]
        yg = y_green[x]

        if yb > 0 and yg > 0 and yg < yb:
            mask[yg:yb, x] = 255

    return mask


def classify_formations(img, deblai_mask):
    arr = np.array(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    area = deblai_mask > 0

    masks = {
        "Les basaltes": area & (h >= 10) & (h <= 30) & (s > 80) & (v > 40) & (v < 210),
        "Les grès": area & (h >= 25) & (h <= 38) & (s > 80) & (v > 120),
        "Les argiles, limons et tufs": area & (((h <= 8) | (h >= 170)) & (s > 80) & (v > 60)),
        "Les marnes et argiles marneuses": area & (h >= 135) & (h <= 165) & (s > 60) & (v > 100),
        "Les schistes sains": area & (s < 45) & (v > 60) & (v < 190),
    }

    rows = []
    total = 0

    for name, mask in masks.items():
        pixels = int(mask.sum())
        total += pixels
        rows.append({"Formation": name, "Pixels": pixels})

    df = pd.DataFrame(rows)

    df["Pourcentage (%)"] = df["Pixels"].apply(
        lambda x: round(x / total * 100, 2) if total else 0
    )

    return df.sort_values("Pourcentage (%)", ascending=False)


def make_overlay(img, deblai_mask):
    arr = np.array(img).copy()
    overlay = arr.copy()

    area = deblai_mask > 0
    overlay[area] = (overlay[area] * 0.6 + np.array([255, 255, 0]) * 0.4).astype(np.uint8)

    return Image.fromarray(overlay)


uploaded = st.file_uploader("Importer image ou PDF du profil", type=["png", "jpg", "jpeg", "pdf"])

if uploaded:
    img = load_image(uploaded)

    st.image(img, caption="Profil importé", use_container_width=True)

    if st.button("Calculer les pourcentages"):
        deblai_mask = build_deblai_mask(img)

        if deblai_mask.sum() == 0:
            st.error("Zone de déblai non détectée. Vérifie que TN est vert et ligne projet bleue.")
        else:
            df = classify_formations(img, deblai_mask)

            st.success("Calcul terminé ✔️")
            st.dataframe(df)

            st.image(
                make_overlay(img, deblai_mask),
                caption="Zone analysée : entre TN vert et ligne bleue",
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
