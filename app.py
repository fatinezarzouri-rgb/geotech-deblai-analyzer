import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pdf2image import convert_from_bytes
from io import BytesIO
from sklearn.cluster import KMeans


st.set_page_config(page_title="Formations en déblai", page_icon="📊")
st.title("📊 Pourcentages des formations au-dessus de la ligne projet")


def load_image(uploaded_file, dpi):
    data = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        pages = convert_from_bytes(data, dpi=dpi)
        return pages[0].convert("RGB")

    return Image.open(BytesIO(data)).convert("RGB")


def hex_to_rgb(hex_color):
    return tuple(
        int(hex_color[i:i + 2], 16)
        for i in (1, 3, 5)
    )


def detect_line_by_color(img, rgb, tolerance):
    arr = np.array(img).astype(np.int16)
    target = np.array(rgb).astype(np.int16)

    dist = np.sqrt(np.sum((arr - target) ** 2, axis=2))
    mask = (dist <= tolerance).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


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


def build_above_line_mask(img, line_mask):
    h, w = np.array(img).shape[:2]
    y_line = line_y_by_x(line_mask)

    mask = np.zeros((h, w), dtype=np.uint8)

    for x in range(w):
        y = y_line[x]

        if y > 0:
            mask[:y, x] = 255

    return mask


def remove_background_pixels(img, area_mask, line_rgb, tolerance):
    arr = np.array(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    area = area_mask > 0

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    not_white = v < 245
    not_black = v > 30
    colorful_or_gray = (s > 20) | ((v > 60) & (v < 220))

    line_mask = detect_line_by_color(img, line_rgb, tolerance) > 0

    valid = area & not_white & not_black & colorful_or_gray & (~line_mask)

    return valid


def detect_dominant_colors(img, area_mask, line_rgb, tolerance, color_count):
    arr = np.array(img)
    valid = remove_background_pixels(img, area_mask, line_rgb, tolerance)

    pixels = arr[valid]

    if len(pixels) < 50:
        return []

    if len(pixels) > 60000:
        idx = np.random.choice(len(pixels), 60000, replace=False)
        pixels = pixels[idx]

    kmeans = KMeans(
        n_clusters=min(color_count, len(pixels)),
        random_state=42,
        n_init=10
    )

    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)

    return [tuple(map(int, c)) for c in colors]


def color_mask(img, rgb, area_mask, tolerance):
    arr = np.array(img).astype(np.int16)
    target = np.array(rgb).astype(np.int16)

    dist = np.sqrt(np.sum((arr - target) ** 2, axis=2))

    return (dist <= tolerance) & (area_mask > 0)


def analyze_colors(img, area_mask, colors, tolerance):
    rows = []
    total = 0

    masks = {}

    for i, color in enumerate(colors, start=1):
        mask = color_mask(img, color, area_mask, tolerance)

        pixels = int(mask.sum())

        masks[f"Couleur_{i}"] = mask
        total += pixels

        rows.append({
            "Formation": f"Couleur_{i}",
            "RGB": str(color),
            "Pixels": pixels
        })

    df = pd.DataFrame(rows)

    if total > 0:
        df["Pourcentage (%)"] = (df["Pixels"] / total * 100).round(2)
    else:
        df["Pourcentage (%)"] = 0

    return df.sort_values("Pourcentage (%)", ascending=False), masks


def make_overlay(img, area_mask):
    arr = np.array(img).copy()
    overlay = arr.copy()

    area = area_mask > 0

    overlay[area] = (
        overlay[area] * 0.60 + np.array([255, 255, 0]) * 0.40
    ).astype(np.uint8)

    return Image.fromarray(overlay)


uploaded = st.file_uploader(
    "Importer image ou PDF du profil",
    type=["png", "jpg", "jpeg", "pdf"]
)

st.sidebar.header("Paramètres")

dpi = st.sidebar.slider("DPI PDF", 100, 300, 200, 50)

line_color_hex = st.sidebar.color_picker(
    "Couleur de la ligne projet",
    value="#ff00ff"
)

line_tolerance = st.sidebar.slider(
    "Tolérance détection ligne",
    5,
    120,
    45,
    5
)

color_count = st.sidebar.slider(
    "Nombre de couleurs/formations à détecter",
    2,
    12,
    5
)

formation_tolerance = st.sidebar.slider(
    "Tolérance couleurs formations",
    5,
    120,
    35,
    5
)

if uploaded:
    img = load_image(uploaded, dpi)
    line_rgb = hex_to_rgb(line_color_hex)

    st.image(img, caption="Profil importé", use_container_width=True)

    if st.button("Analyser"):
        line_mask = detect_line_by_color(
            img,
            line_rgb,
            line_tolerance
        )

        area_mask = build_above_line_mask(
            img,
            line_mask
        )

        if area_mask.sum() == 0:
            st.error("Ligne projet non détectée. Augmente la tolérance ou change la couleur.")
        else:
            colors = detect_dominant_colors(
                img,
                area_mask,
                line_rgb,
                line_tolerance,
                color_count
            )

            if not colors:
                st.error("Aucune couleur de formation détectée.")
            else:
                df, masks = analyze_colors(
                    img,
                    area_mask,
                    colors,
                    formation_tolerance
                )

                st.success("Analyse terminée ✔️")
                st.dataframe(df)

                st.image(
                    make_overlay(img, area_mask),
                    caption="Zone analysée : au-dessus de la ligne projet",
                    use_container_width=True
                )

                output = BytesIO()

                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Pourcentages", index=False)

                output.seek(0)

                st.download_button(
                    label="📥 Télécharger Excel",
                    data=output,
                    file_name="pourcentage_formations.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
