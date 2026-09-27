"""
Genera los archivos del deck HTML (Artifact tipo Slides) a partir de content.py:
  <out>/project/deck.json y <out>/project/slides/<id>.html
Las figuras deben estar subidas como assets del artifact; deck_assets.json mapea
nombre de figura -> "/_blob/<id>".
  lab/.venv/bin/python presentacion/build_deck.py <out_dir>
"""
import html
import json
import sys
from pathlib import Path

import content as C

OUT = Path(sys.argv[1])
ASSETS = json.loads((Path(__file__).parent / "deck_assets.json").read_text())
DARK, LIGHT, INK, SOFT, TEAL, ORANGE, RULE, BAND = ("#14213D", "#F4F6F3", "#1F2933", "#4A5568",
                                                    "#1B7F8C", "#B85714", "#D5DBD6", "#E6ECE8")
HEAD, BODY = "'Source Serif 4', Georgia, serif", "'IBM Plex Sans', Arial, sans-serif"
e = html.escape


def table(header, rows, size=26):
    w = f"{100 / len(header):.2f}%"
    out = [f'<table style="font-family:{BODY}; font-size:{size}px; color:{INK}">',
           "<tr>" + "".join(f'<th style="width:{w}">{e(h)}</th>' for h in header) + "</tr>"]
    for k, row in enumerate(rows):
        bgc = f' style="background:{BAND}"' if k % 2 else ""
        out.append(f"<tr{bgc}>" + "".join(f"<td>{e(str(v))}</td>" for v in row) + "</tr>")
    return "\n".join(out) + "\n</table>"


def img(name, alt, w, h):
    if name not in ASSETS:
        return f'<p style="font-size:24px; color:{SOFT}">[figura pendiente: {e(name)}]</p>'
    return (f'<img src="{ASSETS[name]}" alt="{e(alt)}" '
            f'style="width:{w}px; height:{h}px; object-fit:contain; background:#FBFBF8; border-radius:12px">')


def block(b, side=False):
    k = b[0]
    if k == "text":
        return f'<p style="font-size:30px; line-height:1.4; color:{SOFT}">{e(b[1])}</p>'
    if k == "bullets":
        return (f'<ul style="font-size:32px; line-height:1.35; color:{INK}">'
                + "".join(f"<li>{e(i)}</li>" for i in b[1]) + "</ul>")
    if k == "eq":
        return "".join(f'<p style="font-family:{HEAD}; font-size:44px; color:{DARK}; text-align:center">{e(l)}</p>'
                       for l in b[1])
    if k == "big":
        cards = "".join(
            f'<div style="flex:1; display:flex; flex-direction:column; gap:8px; background:#FBFBF8; padding:32px; '
            f'border:1px solid {RULE}; border-radius:16px">'
            f'<p style="font-family:{HEAD}; font-size:72px; font-weight:600; color:{ORANGE}; line-height:1.1">{e(v)}</p>'
            f'<p style="font-size:28px; color:{SOFT}">{e(lab)}</p></div>' for v, lab in b[1])
        return f'<div style="display:flex; gap:32px">{cards}</div>'
    if k == "table":
        return table(b[1], b[2], 24 if len(b[2]) > 6 else 26)
    if k == "image":
        return img(b[1], b[2], 760 if side else 1664, 560 if side else 420)
    if k == "chart_line":   # en HTML va como tabla: modelos x largo de ventana
        return table(["Modelo"] + b[1], [[m] + [C.f4(v) for v in vals] for m, vals in b[2].items()], 26)
    if k == "flow":
        parts = []
        for i, lab in enumerate(b[1]):
            fill = TEAL if "Capa" in lab else (ORANGE if "ŷ" in lab else "#FBFBF8")
            col = LIGHT if fill != "#FBFBF8" else INK
            parts.append(f'<div style="flex:1; background:{fill}; border:1px solid {RULE}; border-radius:12px; padding:20px">'
                         f'<p style="font-size:24px; color:{col}; text-align:center">{e(lab)}</p></div>')
            if i < len(b[1]) - 1:
                parts.append(f'<x-shape kind="arrow-right" style="background:{SOFT}; width:40px; height:24px"></x-shape>')
        return f'<div style="display:flex; gap:12px; align-items:center">{"".join(parts)}</div>'
    raise ValueError(k)


def slide(s, n):
    notes = f"<aside>{e(s['notes'])}</aside>"
    if s["id"] == "cover":
        return (f'<section id="cover" style="background:{DARK}; color:{LIGHT}; font-family:{BODY}; padding:128px; '
                f'display:flex; flex-direction:column; justify-content:center; gap:40px">\n'
                f'<p style="font-size:28px; color:#7FC4CC; letter-spacing:4px">LABORATORIO 2 · DEEP LEARNING</p>\n'
                f'<h1 style="font-family:{HEAD}; font-size:112px; font-weight:600; line-height:1.05; color:{LIGHT}">{e(s["title"])}</h1>\n'
                f'<p style="font-size:36px; line-height:1.4; color:#BFD8D5">{e(s["body"][0][1])}</p>\n{notes}\n</section>\n')
    kinds = [b[0] for b in s["body"]]
    side = "table" in kinds and "image" in kinds
    if side:
        body = ('<div style="display:flex; gap:64px; align-items:start">'
                + "".join(f'<div style="flex:1; display:flex; flex-direction:column">{block(b, True)}</div>'
                          for b in s["body"]) + "</div>")
    else:
        body = "\n".join(block(b) for b in s["body"])
    return (f'<section id="{s["id"]}" data-transition="fade" style="background:{LIGHT}; color:{INK}; font-family:{BODY}; '
            f'padding:112px 128px 160px; display:flex; flex-direction:column; gap:36px">\n'
            f'<p style="font-size:24px; font-weight:600; color:{TEAL}; letter-spacing:3px">{e(C.LABEL[s["section"]].upper())}</p>\n'
            f'<h2 style="font-family:{HEAD}; font-size:64px; font-weight:600; line-height:1.1; color:{DARK}">{e(s["title"])}</h2>\n'
            f'{body}\n'
            f'<p style="position:absolute; left:128px; bottom:64px; width:1664px; font-size:24px; color:{SOFT}; '
            f'text-align:right">{n} / {len(C.SLIDES)}</p>\n{notes}\n</section>\n')


(OUT / "project/slides").mkdir(parents=True, exist_ok=True)
for n, s in enumerate(C.SLIDES, 1):
    (OUT / f"project/slides/{s['id']}.html").write_text(slide(s, n))
sections, seen = {}, set()
for s in C.SLIDES:
    if s["section"] not in seen:
        seen.add(s["section"])
        sections[s["section"].lower()] = {"description": {"Metodologia": "El problema y el método del paper",
                                                          "Implementacion": "Cómo se llevó el paper al caudal",
                                                          "Resultados": "Ablaciones, baselines y errores"}[s["section"]],
                                          "start": s["id"]}
deck = {"v": 4, "createdOnFiles": {"v": 1, "at": "2026-09-25T20:45:00Z"},
        "title": "RFKAN Caudal — Laboratorio 2", "order": [s["id"] for s in C.SLIDES], "sections": sections,
        "faces": {"source-serif-4": {"family": "Source Serif 4",
                                     "href": "https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400..700&display=swap"},
                  "ibm-plex-sans": {"family": "IBM Plex Sans",
                                    "href": "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap"}},
        "designSystems": []}
(OUT / "project/deck.json").write_text(json.dumps(deck, ensure_ascii=False, indent=1))
print(OUT, len(C.SLIDES), "slides")
