import streamlit as st
import ezdxf
import pandas as pd
from io import BytesIO
import tempfile
import os
from shapely.geometry import Polygon
from shapely.ops import unary_union


st.set_page_config(page_title="Pourcentage formations DXF", page_icon="📐")
st.title("📐 Pourcentage des formations distinctes")
st.write("Importer un DXF déjà découpé dans la zone de déblai.")


def entity_color(e):
    try:
        if e.dxf.hasattr("true_color") and e.dxf.true_color:
            tc = e.dxf.true_color
            r = (tc >> 16) & 255
            g = (tc >> 8) & 255
            b = tc & 255
            return f"RGB({r},{g},{b})"
    except Exception:
        pass

    try:
        return f"ACI({int(e.dxf.color)})"
    except Exception:
        return "UNKNOWN"


def get_entities(msp):
    entities = []

    for e in msp:
        if e.dxftype() == "INSERT":
            try:
                entities.extend(list(e.virtual_entities()))
            except Exception:
                pass
        else:
            entities.append(e)

    return entities


def hatch_to_polygon(e):
    polygons = []

    try:
        for path in e.paths:
            pts = []

            if hasattr(path, "vertices"):
                pts = [(p[0], p[1]) for p in path.vertices]

            elif hasattr(path, "edges"):
                for edge in path.edges:
                    if edge.EDGE_TYPE == "LineEdge":
                        pts.append((edge.start.x, edge.start.y))

            if len(pts) >= 3:
                poly = Polygon(pts).buffer(0)

                if poly.is_valid and poly.area > 0:
                    polygons.append(poly)

    except Exception:
        pass

    if not polygons:
        return None

    return unary_union(polygons)


def closed_polyline_to_polygon(e):
    try:
        if e.dxftype() == "LWPOLYLINE" and e.closed:
            pts = [(p[0], p[1]) for p in e.get_points()]
            return Polygon(pts).buffer(0)

        if e.dxftype() == "POLYLINE" and e.is_closed:
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            return Polygon(pts).buffer(0)

    except Exception:
        return None

    return None


def extract_formations(dxf_path, group_by):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    entities = get_entities(msp)

    rows = []

    for e in entities:
        poly = None

        if e.dxftype() == "HATCH":
            poly = hatch_to_polygon(e)

        elif e.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
            poly = closed_polyline_to_polygon(e)

        if poly is None or poly.area <= 0:
            continue

        layer = str(e.dxf.layer)
        color = entity_color(e)

        if group_by == "Couleur":
            formation = color
        elif group_by == "Layer":
            formation = layer
        else:
            formation = f"{layer} | {color}"

        rows.append({
            "Formation détectée": formation,
            "Layer": layer,
            "Couleur": color,
            "Surface": poly.area
        })

    if not rows:
        raise ValueError("Aucune hachure ou polygone fermé détecté.")

    df = pd.DataFrame(rows)

    df = df.groupby(
        ["Formation détectée"],
        as_index=False
    )["Surface"].sum()

    total = df["Surface"].sum()

    df["Pourcentage (%)"] = (
        df["Surface"] / total * 100
    ).round(2)

    return df.sort_values("Pourcentage (%)", ascending=False)


uploaded = st.file_uploader("Importer le DXF découpé", type=["dxf"])

group_by = st.selectbox(
    "Grouper les formations par",
    ["Couleur", "Layer", "Layer + Couleur"]
)

if uploaded:
    if st.button("Calculer les pourcentages"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            tmp.write(uploaded.read())
            path = tmp.name

        try:
            df = extract_formations(path, group_by)

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

        except Exception as e:
            st.error(str(e))

        finally:
            if os.path.exists(path):
                os.remove(path)
