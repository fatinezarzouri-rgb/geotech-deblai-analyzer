import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from io import BytesIO
from sklearn.cluster import KMeans


st.set_page_config(page_title="Formations en déblai", page_icon="📊")
st.title("📊 Pourcentage des formations en déblai")
st.write("Analyse par pixels au-dessus de la ligne rouge.")


def detect_red_line(image_rgb):
    img = np.array(image_rgb)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))

    red_mask = cv2.bitwise_or(mask1, mask2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    return red_mask


def get_red_line_y(red_mask):
    h, w = red_mask.shape
    line_y = np.full(w, -1, dtype=int)

    for x in range(w):
        ys = np.where(red_mask[:, x] > 0)[0]
        if len(ys) > 0:
            line_y[x] = int(np.median(ys))

    valid = line_y >= 0

    if valid.sum() < 10:
        return line_y

    return np.interp(
        np.arange(w),
        np.where(valid)[0],
        line_y[valid]
    ).astype(int)


def create_deblai_mask(line_y, height):
    mask = np.zeros((height, len(line_y)), dtype=np.uint8)

    for x, y in enumerate(line_y):
        if y > 0:
            mask[:y, x] = 255

    return mask


def remove_white_black_red_pixels(image_rgb, mask):
    img = np.array(image_rgb)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    area = mask > 0

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    not_white = value < 245
    not_black = value > 40
    colorful = saturation > 25

    not_red = ~(
        ((hue <= 10) | (hue >= 170)) &
        (saturation > 60) &
        (value > 50)
    )

    valid = area & not_white & not_black & colorful & not_red

    return valid


def detect_dominant_colors(image_rgb, deblai_mask, color_count):
    img = np.array(image_rgb)
    valid = remove_white_black_red_pixels(image_rgb, deblai_mask)

    pixels = img[valid]

    if len(pixels) < 100:
        return []

    if len(pixels) > 50000:
        idx = np.random.choice(len(pixels), 50000, replace=False)
        pixels = pixels[idx]

    kmeans = KMeans(n_clusters=color_count, random_state=42, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)

    return [tuple(map(int, color)) for color in colors]


def color_mask(image_rgb, target_rgb, tolerance, deblai_mask):
    img = np.array(image_rgb).astype(np.int16)
    target = np.array(target_rgb).astype(np.int16)

    distance = np.sqrt(np.sum((img - target) ** 2, axis=2))

    mask = (distance <= tolerance) & (deblai_mask > 0)

    return mask


def analyze(image_rgb, colors_info, tolerance):
    img = np.array(image_rgb)
    h, w = img.shape[:2]

    red_mask = detect_red_line(image_rgb)
    line_y = get_red_line_y(red_mask)
    deblai_mask = create_deblai_mask(line_y, h)

    results = []
    masks = {}
    total_pixels = 0

    for info in colors_info:
        name = info["name"]
        color = info["color"]

        mask = color_mask(image_rgb, color, tolerance, deblai_mask)
        pixels = int(mask.sum())

        masks[name] = mask
        total_pixels += pixels

        results.append({
            "Formation": name,
            "Couleur RGB": str(color),
            "Pixels": pixels
        })

    for row in results:
        row["Pourcentage (%)"] = round(
            row["Pixels"] / total_pixels * 100,
            2
        ) if total_pixels else 0

    return pd.DataFrame(results), deblai_mask, masks


def make_overlay(image_rgb, deblai_mask, masks):
    img = np.array(image_rgb).copy()
    overlay = img.copy()

    area = deblai_mask > 0
    overlay[area] = (overlay[area] * 0.65 + np.array([255, 255, 0]) * 0.35).astype(np.uint8)

    for mask in masks.values():
        overlay[mask] = (overlay[mask] * 0.35 + np.array([0, 255, 0]) * 0.65).astype(np.uint8)

    return Image.fromarray(overlay)


uploaded_pdf = st.file_uploader("Importer le PDF du profil", type=["pdf"])

st.sidebar.header("⚙️ Paramètres")
dpi = st.sidebar.slider("DPI", 100, 300, 200, 50)
color_count = st.sidebar.slider("Nombre de couleurs/formations à détecter", 2, 15, 6)
tolerance = st.sidebar.slider("Tolérance couleur", 5, 120, 35, 5)

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.read()

    pages = convert_from_bytes(pdf_bytes, dpi=dpi)

    page_number = st.number_input(
        "Page à analyser",
        min_value=1,
        max_value=len(pages),
        value=1
    )

    image = pages[page_number - 1].convert("RGB")

    st.image(image, caption=f"Page {page_number}", use_container_width=True)

    red_mask = detect_red_line(image)
    line_y = get_red_line_y(red_mask)
    deblai_mask = create_deblai_mask(line_y, np.array(image).shape[0])

    if st.button("Détecter les couleurs du PDF"):
        colors = detect_dominant_colors(image, deblai_mask, color_count)
        st.session_state["colors"] = colors

    if "colors" in st.session_state:
        st.subheader("Couleurs détectées")

        colors_info = []

        for i, color in enumerate(st.session_state["colors"], start=1):
            col1, col2 = st.columns([1, 4])

            with col1:
                color_img = np.zeros((60, 120, 3), dtype=np.uint8)
                color_img[:, :] = color
                st.image(color_img)

            with col2:
                name = st.text_input(
                    f"Nom de la formation couleur {i}",
                    value=f"Formation_{i}"
                )

            colors_info.append({
                "name": name,
                "color": color
            })

        if st.button("Calculer les pourcentages"):
            df, deblai_mask, masks = analyze(
                image,
                colors_info,
                tolerance
            )

            st.subheader("Résultats")
            st.dataframe(df)

            overlay = make_overlay(image, deblai_mask, masks)

            st.subheader("Contrôle visuel")
            st.image(
                overlay,
                caption="Jaune = zone déblai / Vert = pixels classés",
                use_container_width=True
            )

            output = BytesIO()

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Pourcentages", index=False)

            output.seek(0)

            st.download_button(
                label="📥 Télécharger Excel",
                data=output,
                file_name="pourcentage_formations_deblai.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
