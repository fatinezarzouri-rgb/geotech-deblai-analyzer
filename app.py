import streamlit as st
import ezdxf
import pandas as pd
from io import BytesIO
import tempfile
import os

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


st.set_page_config(page_title="Analyse déblai DXF", page_icon="📐")
st.title("📐 Analyse des formations en déblai depuis DXF")

BLUE = 5
GREEN = 3


def entity_color(entity):
    return int(entity.dxf.color) if entity.dxf.hasattr("color") else 256


def entity_layer(entity):
    return str(entity.dxf.layer)


def entity_to_line(entity):
    try:
        if entity.dxftype() == "LINE":
            s = entity.dxf.start
            e = entity.dxf.end
            return LineString([(s.x, s.y), (e.x, e.y)])

        if entity.dxftype() == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in entity.get_points()]
            if len(pts) >= 2:
                return LineString(pts)

        if entity.dxftype() == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
            if len(pts) >= 2:
                return LineString(pts)

    except Exception:
        return None

    return None


def closed_polyline_to_polygon(entity):
    try:
        if entity.dxftype() == "LWPOLYLINE" and entity.closed:
            pts = [(p[0], p[1]) for p in entity.get_points()]
            poly = Polygon(pts)
            return poly.buffer(0) if poly.is_valid else None

        if entity.dxftype() == "POLYLINE" and entity.is_closed:
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
            poly = Polygon(pts)
            return poly.buffer(0) if poly.is_valid else None

    except Exception:
        return None

    return None


def hatch_to_polygon(entity):
    polygons = []

    try:
        for path in entity.paths:
            pts = []

            if hasattr(path, "vertices"):
                pts = [(p[0], p[1]) for p in path.vertices]

            elif hasattr(path, "edges"):
                for edge in path.edges:
                    if edge.EDGE_TYPE == "LineEdge":
                        pts.append((edge.start.x, edge.start.y))

            if len(pts) >= 3:
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 0:
                    polygons.append(poly.buffer(0))

    except Exception:
        pass

    if not polygons:
        return None

    return unary_union(polygons)


def find_longest_line_by_color(msp, color):
    lines = []

    for e in msp:
        if e.dxftype() not in ["LINE", "LWPOLYLINE", "POLYLINE"]:
            continue

        if entity_color(e) == color:
            line = entity_to_line(e)

            if line is not None and line.length > 0:
                lines.append(line)

    if not lines:
        return None

    return max(lines, key=lambda g: g.length)


def line_y_at_x(line, x):
    coords = list(line.coords)

    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]

        if x1 == x2:
            continue

        if min(x1, x2) <= x <= max(x1, x2):
            t = (x - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)

    return None


def build_deblai_polygon(tn_line, project_line, samples=1500):
    min_x = max(tn_line.bounds[0], project_line.bounds[0])
    max_x = min(tn_line.bounds[2], project_line.bounds[2])

    if min_x >= max_x:
        return None

    top = []
    bottom = []

    for i in range(samples + 1):
        x = min_x + (max_x - min_x) * i / samples

        y_tn = line_y_at_x(tn_line, x)
        y_project = line_y_at_x(project_line, x)

        if y_tn is None or y_project is None:
            continue

        if y_tn > y_project:
            top.append((x, y_tn))
            bottom.append((x, y_project))

    if len(top) < 3:
        return None

    poly = Polygon(top + bottom[::-1])

    if not poly.is_valid:
        poly = poly.buffer(0)

    return poly


def extract_formations(msp):
    formations = []

    for e in msp:
        color = entity_color(e)

        if color in [BLUE, GREEN]:
            continue

        poly = None

        if e.dxftype() == "HATCH":
            poly = hatch_to_polygon(e)

        elif e.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
            poly = closed_polyline_to_polygon(e)

        if poly is not None and poly.area > 0:
            formations.append({
                "formation": entity_layer(e),
                "color": color,
                "geometry": poly
            })

    return formations


def analyze_dxf(path):
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    project_line = find_longest_line_by_color(msp, BLUE)
    tn_line = find_longest_line_by_color(msp, GREEN)

    if project_line is None:
        raise ValueError("Ligne projet bleue introuvable.")

    if tn_line is None:
        raise ValueError("Terrain naturel vert introuvable.")

    deblai = build_deblai_polygon(tn_line, project_line)

    if deblai is None or deblai.area <= 0:
        raise ValueError("Aucune zone en déblai détectée.")

    formations = extract_formations(msp)

    if not formations:
        raise ValueError("Aucune formation/hachure détectée.")

    rows = []

    for f in formations:
        inter = f["geometry"].intersection(deblai)

        if not inter.is_empty and inter.area > 0:
            rows.append({
                "Formation": f["formation"],
                "Couleur AutoCAD": f["color"],
                "Surface en déblai": inter.area
            })

    if not rows:
        raise ValueError("Aucune formation dans la zone de déblai.")

    df = pd.DataFrame(rows)

    df = df.groupby(
        ["Formation", "Couleur AutoCAD"],
        as_index=False
    )["Surface en déblai"].sum()

    total = df["Surface en déblai"].sum()

    df["Pourcentage (%)"] = (
        df["Surface en déblai"] / total * 100
    ).round(2)

    return df, deblai.area


uploaded = st.file_uploader("Importer fichier DXF", type=["dxf"])

if uploaded is not None:
    if st.button("Calculer"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            tmp.write(uploaded.read())
            path = tmp.name

        try:
            with st.spinner("Analyse du fichier DXF..."):
                df, total_deblai = analyze_dxf(path)

            st.success("Calcul terminé ✔️")
            st.write(f"Surface totale déblai : `{total_deblai:.2f}`")

            st.dataframe(df)

            output = BytesIO()

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(
                    writer,
                    sheet_name="Formations_deblai",
                    index=False
                )

            output.seek(0)

            st.download_button(
                "📥 Télécharger Excel",
                output,
                file_name="formations_deblai.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(str(e))

        finally:
            if os.path.exists(path):
                os.remove(path)
