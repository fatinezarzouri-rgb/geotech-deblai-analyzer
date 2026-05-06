import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from io import BytesIO


st.set_page_config(page_title="Pourcentage formations", page_icon="📊")
st.title("📊 Pourcentage des formations dans le déblai")


def load_image(file):
    return Image.open(file).convert("RGB")


def classify_pixels(img):
    arr = np.array(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # ignorer fond sombre, blanc, traits fins
    valid = (v > 40) & (s > 40)

    # ignorer bleu ligne projet
    blue = (h >= 95) & (h <= 135) & (s > 60)

    # ignorer vert TN
    green = (h >= 40) & (h <= 90) & (s > 60)

    valid = valid & ~blue & ~green

    masks = {}

    # Marron = basaltes
    masks["Basaltes"] = (
        valid &
        (h >= 5) & (h <= 25) &
        (s > 80) &
        (v > 45) & (v < 210)
    )

    # Jaune = grès
    masks["Grès"] = (
        valid &
        (h >= 22) & (h <= 38) &
        (s > 80) &
        (v > 120)
    )

    # Rose = marnes
    masks["Marnes et argiles marneuses"] = (
        valid &
        (h >= 135) & (h <= 165) &
        (s > 50) &
        (v > 90)
    )

    # Gris = schistes sains
    masks["Schistes sains"] = (
        (v > 60) & (v < 190) &
        (s < 45)
    )

    # Rouge hachuré = argiles / limons / tufs
    red_lines = (
        valid &
        (((h <= 8) | (h >= 170)) &
        (s > 70) &
        (v > 50))
    )

    # épaissir les hachures rouges pour représenter la zone hachurée
    red_zone = cv2.dilate(
        red_lines.astype(np.uint8),
        np.ones((11, 11), np.uint8),
        iterations=2
    ).astype(bool)

    red_zone = red_zone & ~blue & ~green

    masks["Argiles / limons / tufs"] = red_zone

    # éviter double comptage
    assigned = np.zeros(h.shape, dtype=bool)
    final_masks = {}

    priority = [
        "Argiles / limons / tufs",
        "Marnes et argiles marneuses",
        "Grès",
        "Basaltes",
        "Schistes sains"
    ]

    for name in priority:
        final_masks[name] = masks[name] & ~assigned
        assigned |= final_masks[name]

    total = sum(int(m.sum()) for m in final_masks.values())

    rows = []

    for name, mask in final_masks.items():
        pixels = int(mask.sum())
        pct = round(pixels / total * 100, 2) if total else 0

        rows.append({
            "Formation": name,
            "Pixels": pixels,
            "Pourcentage (%)": pct
        })

    return pd.DataFrame(rows).sort_values("Pourcentage (%)", ascending=False), final_masks


def overlay_result(img, masks):
    arr = np.array(img).copy()

    colors = {
        "Basaltes": [165, 85, 0],
        "Grès": [255, 255, 0],
        "Marnes et argiles marneuses": [255, 80, 255],
        "Schistes sains": [130, 130, 130],
        "Argiles / limons / tufs": [255, 0, 0]
    }

    out = arr.copy()

    for name, mask in masks.items():
        color = np.array(colors[name])
        out[mask] = (out[mask] * 0.35 + color * 0.65).astype(np.uint8)

    return Image.fromarray(out)


uploaded = st.file_uploader(
    "Importer l’image nettoyée du déblai",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    img = load_image(uploaded)

    st.image(img, caption="Image importée", use_container_width=True)

    if st.button("Calculer"):
        df, masks = classify_pixels(img)

        st.success("Calcul terminé ✔️")
        st.dataframe(df)

        st.image(
            overlay_result(img, masks),
            caption="Pixels classés par formation",
            use_container_width=True
        )

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
