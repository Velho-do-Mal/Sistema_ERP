"""
Renderização do Boletim de Medição em HTML (padrão BK)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def _fmt_money(v: float) -> str:
    try:
        return f"R$ {float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0,00"


def _badge(status: str) -> str:
    s = (status or "").lower()
    if s == "aprovado":
        cls = "ok"
        label = "APROVADO"
    elif s in ("enviado", "submetido"):
        cls = "warn"
        label = "ENVIADO"
    elif s in ("rejeitado", "cancelado"):
        cls = "bad"
        label = "REJEITADO"
    else:
        cls = ""
        label = (status or "RASCUNHO").upper()
    return f'<span class="badge {cls}">{label}</span>'


def _cards(summary: Dict[str, float]) -> str:
    items = [
        ("Contrato", _fmt_money(summary.get("contrato", 0))),
        ("Medição no período", _fmt_money(summary.get("periodo", 0))),
        ("Acumulado", _fmt_money(summary.get("acumulado", 0))),
        ("Saldo", _fmt_money(summary.get("saldo", 0))),
    ]
    html = []
    for k, v in items:
        html.append(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>')
    return "\n".join(html)


def _svg_bar(items: pd.DataFrame, value_col: str = "valor_periodo") -> str:
    if items is None or items.empty or value_col not in items.columns:
        return '<div class="muted">Sem dados para gráfico.</div>'

    df = items.copy()
    df["label"] = df["descricao"].astype(str).str.slice(0, 28)
    df["value"] = df[value_col].astype(float).fillna(0.0)
    df = df.sort_values("value", ascending=False).head(10)
    maxv = float(df["value"].max()) if not df.empty else 0.0
    if maxv <= 0:
        return '<div class="muted">Sem valores medidos no período.</div>'

    w = 860
    h = 240
    pad = 24
    bar_h = 16
    gap = 8
    y0 = 20
    svg = [f'<svg width="100%" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<rect x="0" y="0" width="{w}" height="{h}" rx="12" fill="#ffffff"/>')
    y = y0
    for _, r in df.iterrows():
        lab = str(r["label"])
        val = float(r["value"])
        bw = int((w - 240) * (val / maxv))
        svg.append(f'<text x="{pad}" y="{y+12}" font-size="12" fill="#334155">{lab}</text>')
        svg.append(f'<rect x="220" y="{y}" width="{bw}" height="{bar_h}" rx="6" fill="#0ea5e9"/>')
        svg.append(f'<text x="{220 + bw + 8}" y="{y+12}" font-size="12" fill="#0f172a">{_fmt_money(val)}</text>')
        y += bar_h + gap
        if y > h - 24:
            break
    svg.append("</svg>")
    return "\n".join(svg)


def _items_table(items: pd.DataFrame) -> str:
    if items is None or items.empty:
        return '<div class="card muted">Sem itens.</div>'

    df = items.copy()
    cols = [
        ("descricao", "Descrição"),
        ("unidade", "Un"),
        ("qtde_contratada", "Qtde Contr."),
        ("qtde_periodo", "Qtde Período"),
        ("qtde_acumulada", "Qtde Acum."),
        ("qtde_saldo", "Saldo Qtde"),
        ("valor_unit", "Vlr Unit"),
        ("valor_periodo", "Vlr Período"),
        ("valor_acumulado", "Vlr Acum."),
        ("valor_saldo", "Saldo R$"),
    ]
    def fmt(x, is_money=False):
        if x is None:
            return ""
        try:
            if is_money:
                return _fmt_money(float(x))
            return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(x)

    rows = []
    rows.append("<table><thead><tr>" + "".join([f"<th>{h}</th>" for _, h in cols]) + "</tr></thead><tbody>")
    for _, r in df.iterrows():
        tds = []
        for c, _h in cols:
            val = r.get(c, "")
            if c in ("valor_unit", "valor_periodo", "valor_acumulado", "valor_saldo"):
                tds.append(f'<td class="right">{fmt(val, True)}</td>')
            elif c in ("qtde_contratada","qtde_periodo","qtde_acumulada","qtde_saldo"):
                tds.append(f'<td class="right">{fmt(val)}</td>')
            else:
                tds.append(f'<td>{val}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def render_boletim_html(
    template_path: str | Path,
    header: Dict[str, Any],
    project_name: str,
    items: pd.DataFrame,
    summary: Dict[str, float],
) -> str:
    tpl = Path(template_path).read_text(encoding="utf-8")

    period = ""
    ps = header.get("period_start")
    pe = header.get("period_end")
    if ps and pe:
        period = f"{ps} a {pe}"
    elif ps:
        period = f"{ps}"
    elif pe:
        period = f"{pe}"
    else:
        period = "-"

    status_badge = _badge(header.get("status", "rascunho"))

    html = tpl
    html = html.replace("{{PROJETO}}", project_name or "(sem nome)")
    html = html.replace("{{PERIODO}}", period)
    html = html.replace("{{REFERENCIA}}", str(header.get("reference") or ""))
    html = html.replace("{{STATUS}}", status_badge)
    html = html.replace("{{CARDS_RESUMO}}", _cards(summary))
    html = html.replace("{{GRAFICO_SVG}}", _svg_bar(items, "valor_periodo"))
    html = html.replace("{{TABELA_ITENS}}", _items_table(items))
    html = html.replace("{{OBSERVACOES}}", str(header.get("notes") or "—"))
    html = html.replace("{{GERADO_EM}}", datetime.now().strftime("%d/%m/%Y %H:%M"))
    return html
