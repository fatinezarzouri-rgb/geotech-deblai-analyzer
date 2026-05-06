import streamlit as st
import ezdxf
import pandas as pd
from io import BytesIO
import tempfile
import os

from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union


st.set_page_config(page_title="Déblai DXF", page_icon="📐")

st.title("📐 Analyse déblai depuis AutoCAD DXF")
st.write("Ligne projet bleue + formations AutoCAD → pourcentages en déblai.")


def get_color_name(entity):
    color = entity.dxf.color

    if color == 5:
        return "blue"
    if color == 1:
        return "red"
    if color == 3:
        return "green"
    if color == 2:
        return "yellow"
    return str(color)


def entity_to_linestring(entity):
    try:
        if entity.dxftype() == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in entity.get_points()]
            if len(pts) >= 2:
                return LineString(pts)

        if entity.dxftype() == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
            if len(pts) >= 2:
                return LineString(pts)

        if entity.dxftype() == "LINE":
            start = entity.dxf.start
            end = entity.dxf.end
            return LineString([(start.x, start.y), (end.x, end.y)])

    except Exception:
        return None

    return None


def hatch_to_polygon(entity):
    polygons = []

    try:
        for path in entity.paths:
            pts = []

            for edge in path.edges:
                if edge.EDGE_TYPE == "LineEdge":
                    pts.append((edge.start.x, edge.start.y))

            if len(pts) >= 3:
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 0:
                    polygons.append(poly)

    except Exception:
        pass

    if not polygons:
        return None

    return unary_union(polygons)


def closed_polyline_to_polygon(entity):
    try:
        if entity.dxftype() == "LWPOLYLINE" and entity.closed:
            pts = [(p[0], p[1]) for p in entity.get_points()]
            if len(pts) >= 3:
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 0:
                    return poly

        if entity.dxftype() == "POLYLINE" and entity.is_closed:
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
            if len(pts) >= 3:
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 0:
                    return poly

    except Exception:
        return None

    return None


def find_project_line(msp, project_layer, use_blue=True):
    candidates = []

    for entity in msp:
        if entity.dxftype() not in ["LINE", "LWPOLYLINE", "POLYLINE"]:
            continue

        layer_match = entity.dxf.layer.lower() == project_layer.lower()
        blue_match = use_blue and entity.dxf.color == 5

        if layer_match or blue_match:
            line = entity_to_linestring(entity)
            if line and line.length > 0:
                candidates.append(line)

    if not candidates:
        return None

    return max(candidates, key=lambda g: g.length)


def find_tn_line(msp, tn_layer):
    candidates = []

    for entity in msp:
        if entity.dxftype() not in ["LINE", "LWPOLYLINE", "POLYLINE"]:
            continue

        if entity.dxf.layer.lower() == tn_layer.lower():
            line = entity_to_linestring(entity)
            if line and line.length > 0:
                candidates.append(line)

    if not candidates:
        return None

    return max(candidates, key=lambda g: g.length)


def line_y_at_x(line, x):
    coords = list(line.coords)

    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]

        if min(x1, x2) <= x <= max(x1, x2) and x1 != x2:
            t = (x - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)

    return None


def build_deblai_polygon(tn_line, project_line, samples=1000):
    min_x = max(tn_line.bounds[0], project_line.bounds[0])
    max_x = min(tn_line.bounds[2], project_line.bounds[2])

    if min_x >= max_x:
        return None

    top_points = []
    bottom_points = []

    for i in range(samples + 1):
        x = min_x + (max_x - min_x) * i / samples

        y_tn = line_y_at_x(tn_line, x)
        y_project = line_y_at_x(project_line, x)

        if y_tn is None or y_project is None:
            continue

        if y_tn > y_project:
            top_points.append((x, y_tn))
            bottom_points.append((x, y_project))

    if len(top_points) < 3:
        return None

    coords = top_points + bottom_points[::-1]
    poly = Polygon(coords)

    if not poly.is_valid:
        poly = poly.buffer(0)

    return poly


def extract_formations(msp, ignored_layers):
    formations = []

    for entity in msp:
        layer = entity.dxf.layer

        if layer.lower() in [x.lower() for x in ignored_layers]:
            continue

        poly = None

        if entity.dxftype() == "HATCH":
            poly = hatch_to_polygon(entity)

        elif entity.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
            poly = closed_polyline_to_polygon(entity)

        if poly is not None and poly.area > 0:
            formations.append({
                "formation": layer,
                "geometry": poly,
                "color": get_color_name(entity)
            })

    return formations


def analyze_dxf(dxf_path, tn_layer, project_layer, use_blue_project):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    project_line = find_project_line(
        msp,
        project_layer=project_layer,
        use_blue=use_blue_project
    )

    tn_line = find_tn_line(
        msp,
        tn_layer=tn_layer
    )

    if project_line is None:
        raise ValueError("Ligne projet introuvable. Vérifie le layer ou la couleur bleue.")

    if tn_line is None:
        raise ValueError("Ligne TN introuvable. Vérifie le nom du layer TN.")

    deblai_poly = build_deblai_polygon(tn_line, project_line)

    if deblai_poly is None or deblai_poly.area == 0:
        raise ValueError("Aucune zone en déblai détectée : TN n'est pas au-dessus de la ligne projet.")

    formations = extract_formations(
        msp,
        ignored_layers=[tn_layer, project_layer]
    )

    rows = []
    total_area = 0

    for f in formations:
        inter = f["geometry"].intersection(deblai_poly)

        if not inter.is_empty and inter.area > 0:
            area = inter.area
            total_area += area

            rows.append({
                "Formation": f["formation"],
                "Couleur": f["color"],
                "Surface en déblai": area
            })

    if not rows:
        raise ValueError("Aucune formation/hachure trouvée dans la zone de déblai.")

    df = pd.DataFrame(rows)

    df = df.groupby(["Formation", "Couleur"], as_index=False)["Surface en déblai"].sum()

    df["Pourcentage (%)"] = round(
        df["Surface en déblai"] / df["Surface en déblai"].sum() * 100,
        2
    )

    return df, deblai_poly.area


uploaded_file = st.file_uploader("Importer fichier AutoCAD DXF", type=["dxf"])

st.sidebar.header("Paramètres AutoCAD")

tn_layer = st.sidebar.text_input("Nom du layer TN", value="TN")
project_layer = st.sidebar.text_input("Nom du layer ligne projet", value="PROJET")

use_blue_project = st.sidebar.checkbox(
    "Détecter aussi la ligne projet bleue",
    value=True
)

if uploaded_file is not None:
    if st.button("Calculer les pourcentages"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            tmp.write(uploaded_file.read())
            dxf_path = tmp.name

        try:
            with st.spinner("Analyse du DXF..."):
                df, deblai_area = analyze_dxf(
                    dxf_path,
                    tn_layer,
                    project_layer,
                    use_blue_project
                )

            st.success("Analyse terminée ✔️")

            st.write(f"Surface totale déblai détectée : `{deblai_area:.2f}`")
            st.dataframe(df)

            output = BytesIO()

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Formations_deblai", index=False)

            output.seek(0)

            st.download_button(
                label="📥 Télécharger Excel",
                data=output,
                file_name="formations_deblai.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(str(e))

        finally:
            if os.path.exists(dxf_path):
                os.remove(dxf_path)
