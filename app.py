import streamlit as st
import ezdxf
import pandas as pd
from io import BytesIO
import tempfile
import os
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


BLUE = 5
GREEN = 3

COLOR_TO_FORMATION = {
    30: "Les basaltes",
    34: "Les basaltes",
    2: "Les grès",
    50: "Les grès",
    1: "Les argiles, les limons et les tufs",
    6: "Les marnes et argiles marneuses",
    8: "Les schistes sains",
    9: "Les schistes altérés",
    4: "Les conglomérats",
}

st.set_page_config(page_title="Déblai formations DXF", page_icon="📐")
st.title("📐 Calcul des formations en déblai depuis DXF")


def entity_color(e):
    try:
        return int(e.dxf.color)
    except Exception:
        return 256


def get_all_entities(msp):
    entities = []

    for e in msp:
        if e.dxftype() == "INSERT":
            try:
                for ve in e.virtual_entities():
                    entities.append(ve)
            except Exception:
                pass
        else:
            entities.append(e)

    return entities


def entity_to_line(e):
    try:
        if e.dxftype() == "LINE":
            s, p = e.dxf.start, e.dxf.end
            return LineString([(s.x, s.y), (p.x, p.y)])

        if e.dxftype() == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in e.get_points()]
            if len(pts) >= 2:
                return LineString(pts)

        if e.dxftype() == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            if len(pts) >= 2:
                return LineString(pts)

    except Exception:
        return None

    return None


def closed_polyline_to_polygon(e):
    try:
        if e.dxftype() == "LWPOLYLINE" and e.closed:
            pts = [(p[0], p[1]) for p in e.get_points()]
            poly = Polygon(pts)
            return poly.buffer(0)

        if e.dxftype() == "POLYLINE" and e.is_closed:
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            poly = Polygon(pts)
            return poly.buffer(0)

    except Exception:
        return None

    return None


def hatch_to_polygon(e):
    polys = []

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
                if poly.area > 0:
                    polys.append(poly)

    except Exception:
        pass

    if not polys:
        return None

    return unary_union(polys)


def find_longest_line_by_color(entities, color):
    lines = []

    for e in entities:
        if e.dxftype() not in ["LINE", "LWPOLYLINE", "POLYLINE"]:
            continue

        if entity_color(e) == color:
            line = entity_to_line(e)
            if line is not None and line.length > 0:
                lines.append(line)

    if not lines:
        return None

    return max(lines, key=lambda x: x.length)


def y_at_x(line, x):
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


def build_deblai_polygon(tn, projet, samples=2000):
    min_x = max(tn.bounds[0], projet.bounds[0])
    max_x = min(tn.bounds[2], projet.bounds[2])

    top = []
    bottom = []

    for i in range(samples + 1):
        x = min_x + (max_x - min_x) * i / samples
        y_tn = y_at_x(tn, x)
        y_pr = y_at_x(projet, x)

        if y_tn is None or y_pr is None:
            continue

        if y_tn > y_pr:
            top.append((x, y_tn))
            bottom.append((x, y_pr))

    if len(top) < 3:
        return None

    return Polygon(top + bottom[::-1]).buffer(0)


def extract_formation_polygons(entities):
    formations = []

    for e in entities:
        color = entity_color(e)

        if color in [BLUE, GREEN, 7, 256]:
            continue

        name = COLOR_TO_FORMATION.get(color)

        if not name:
            continue

        poly = None

        if e.dxftype() == "HATCH":
            poly = hatch_to_polygon(e)

        elif e.dxftype() in ["LWPOLYLINE", "POLYLINE"]:
            poly = closed_polyline_to_polygon(e)

        if poly is not None and poly.area > 0:
            formations.append({
                "Formation": name,
                "Couleur AutoCAD": color,
                "geometry": poly
            })

    return formations


def analyze_dxf(path):
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    entities = get_all_entities(msp)

    projet = find_longest_line_by_color(entities, BLUE)
    tn = find_longest_line_by_color(entities, GREEN)

    if projet is None:
        raise ValueError("Ligne projet bleue introuvable.")

    if tn is None:
        raise ValueError("Terrain naturel vert introuvable.")

    deblai = build_deblai_polygon(tn, projet)

    if deblai is None or deblai.area <= 0:
        raise ValueError("Aucune zone en déblai détectée.")

    formations = extract_formation_polygons(entities)

    rows = []

    for f in formations:
        inter = f["geometry"].intersection(deblai)

        if not inter.is_empty and inter.area > 0:
            rows.append({
                "Formation": f["Formation"],
                "Couleur AutoCAD": f["Couleur AutoCAD"],
                "Surface en déblai": inter.area
            })

    if not rows:
        raise ValueError("Aucune formation détectée dans la zone de déblai.")

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


uploaded = st.file_uploader("Importer le fichier DXF / XREF exporté", type=["dxf"])

if uploaded:
    if st.button("Calculer les pourcentages"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            tmp.write(uploaded.read())
            path = tmp.name

        try:
            df, total = analyze_dxf(path)

            st.success("Calcul terminé ✔️")
            st.write(f"Surface totale en déblai : `{total:.2f}`")
            st.dataframe(df)

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

        except Exception as e:
            st.error(str(e))

        finally:
            if os.path.exists(path):
                os.remove(path)
