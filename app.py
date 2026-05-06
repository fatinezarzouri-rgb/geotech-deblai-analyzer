import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from sklearn.cluster import KMeans

st.set_page_config(page_title="Pourcentage formations", page_icon="📊")
st.title("📊 Calcul des pourcentages des formations depuis une image")

uploaded = st.file_uploader("Importer l’image découpée des formations", type=["png", "jpg", "jpeg"])

def crop_image(img, x1, x2, y1, y2):
    return img.crop((x1, y1, x2, y2))

def rgb_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=2))

def get_dominant_colors(img, n_colors):
    arr = np.array(img.convert("RGB"))
    pixels = arr.reshape(-1, 3)

    # enlever fond sombre, blanc, gris très faible
    brightness = pixels.mean(axis=1)
    saturation = pixels.max(axis=1) - pixels.min(axis=1)
    keep = (brightness > 35) & (brightness < 245) & (saturation > 20)

    pixels = pixels[keep]

    if len(pixels) > 60000:
        idx = np.random.choice(len(pixels), 60000, replace=False)
        pixels = pixels[idx]

    model = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    model.fit(pixels)

    return model.cluster_centers_.astype(int)

def calculate_percentages(img, colors, names, tolerance):
    arr = np.array(img.convert("RGB")).astype(int)
    assigned = np.zeros(arr.shape[:2], dtype=bool)

    rows = []

    for color, name in zip(colors, names):
        color = np.array(color).astype(int)
        dist = rgb_distance(arr, color)
        mask = (dist <= tolerance) & (~assigned)

        pixels = int(mask.sum())
        assigned |= mask

        rows.append({
            "Formation": name,
            "RGB": tuple(color.tolist()),
            "Pixels": pixels
        })

    total = sum(r["Pixels"] for r in rows)

    for r in rows:
        r["Pourcentage (%)"] = round(r["Pixels"] / total * 100, 2) if total else 0

    return pd.DataFrame(rows).sort_values("Pourcentage (%)", ascending=False)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    w, h = img.size

    st.image(img, caption="Image originale", use_container_width=True)

    st.subheader("1. Découper la zone utile")

    x1 = st.slider("X début", 0, w, 0)
    x2 = st.slider("X fin", 0, w, w)
    y1 = st.slider("Y début", 0, h, 0)
    y2 = st.slider("Y fin", 0, h, h)

    cropped = crop_image(img, x1, x2, y1, y2)
    st.image(cropped, caption="Zone analysée", use_container_width=True)

    st.subheader("2. Détection des couleurs")

    n_colors = st.slider("Nombre de formations/couleurs", 2, 10, 5)
    tolerance = st.slider("Tolérance couleur", 10, 100, 35)

    if st.button("Détecter les couleurs"):
        st.session_state.colors = get_dominant_colors(cropped, n_colors)

    if "colors" in st.session_state:
        names = []

        for i, color in enumerate(st.session_state.colors):
            col1, col2 = st.columns([1, 4])

            with col1:
                color_img = np.zeros((60, 120, 3), dtype=np.uint8)
                color_img[:, :] = color
                st.image(color_img)

            with col2:
                name = st.text_input(f"Nom formation {i+1}", value=f"Formation_{i+1}")
                names.append(name)

        if st.button("Calculer les pourcentages"):
            df = calculate_percentages(cropped, st.session_state.colors, names, tolerance)

            st.success("Calcul terminé ✔️")
            st.dataframe(df)

            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Pourcentages", index=False)

            output.seek(0)

            st.download_button(
                "📥 Télécharger Excel",
                data=output,
                file_name="pourcentage_formations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
