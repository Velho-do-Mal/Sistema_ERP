from __future__ import annotations

from pathlib import Path
from datetime import date
import base64
from jinja2 import Template


def _svg_to_datauri(svg_path: Path) -> str:
    svg_bytes = svg_path.read_bytes()
    b64 = base64.b64encode(svg_bytes).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def money_br(v: float) -> str:
    try:
        v = float(v or 0)
    except Exception:
        v = 0.0
    # Formato pt-BR simples sem locale global
    s = f"{v:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def render_proposta_html(
    template_path: Path,
    logo_path: Path,
    context: dict,
) -> str:
    tpl = Template(template_path.read_text(encoding="utf-8"))
    ctx = dict(context)
    ctx.setdefault("LOGO_DATAURI", _svg_to_datauri(logo_path))
    return tpl.render(**ctx)
