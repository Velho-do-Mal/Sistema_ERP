"""
BK_ERP - Relatório de Controle de Projetos (sem matplotlib)

Este arquivo NÃO depende de matplotlib. Ele gera gráficos simples em SVG embutidos no HTML.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import text

TEMPLATE_PATH = Path("reports/templates/controle_projetos_BK.html")


def _svg_bar(values: list[tuple[str, float]], width: int = 560, min_height: int = 140) -> str:
    if not values:
        return "<svg width='560' height='80'><text x='10' y='30' font-size='14'>Sem dados</text></svg>"
    maxv = max(v for _, v in values) or 1.0

    pad = 14
    bar_h = 18
    gap = 8
    n = len(values)
    height = max(min_height, pad * 2 + n * (bar_h + gap))
    x_label = 170
    usable = width - x_label - pad

    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts = [f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"]
    parts.append(f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>")
    y = pad
    for label, v in values:
        bw = int(usable * (float(v) / maxv))
        parts.append(f"<text x='{pad}' y='{y+13}' font-size='12' fill='#111827'>{esc(label)[:32]}</text>")
        parts.append(f"<rect x='{x_label}' y='{y}' width='{usable}' height='{bar_h}' rx='6' fill='#EEF2FF'/>")
        parts.append(f"<rect x='{x_label}' y='{y}' width='{bw}' height='{bar_h}' rx='6' fill='#2563EB'/>")
        parts.append(f"<text x='{x_label + bw + 8}' y='{y+13}' font-size='12' fill='#111827'>{float(v):.0f}</text>")
        y += bar_h + gap
    parts.append("</svg>")
    return "".join(parts)


def _fmt_date(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (datetime, date)):
        return x.strftime("%d/%m/%Y")
    try:
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return ""
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return str(x)


def build_report(SessionLocal, project_id: Optional[int] = None) -> str:
    tpl = TEMPLATE_PATH.read_text(encoding="utf-8") if TEMPLATE_PATH.exists() else "<html><body>{{content}}</body></html>"

    with SessionLocal() as conn:
        where = "WHERE 1=1"
        if project_id:
            where += " AND id=:pid"

        projects = pd.read_sql(
            text(
                f"""
                SELECT id,
                       COALESCE(nome,'(sem nome)') AS nome,
                       COALESCE(status,'') AS status,
                       planned_end_date,
                       actual_end_date,
                       COALESCE(progress_pct,0) AS progress_pct,
                       COALESCE(delay_responsibility,'N/A') AS delay_responsibility
                FROM projects
                {where}
                ORDER BY id DESC
                """
            ),
            conn,
            params={"pid": project_id} if project_id else None,
        )

        try:
            tasks = pd.read_sql(
                text(
                    """
                    SELECT project_id,
                           COALESCE(title,'') AS title,
                           due_date,
                           COALESCE(is_done,FALSE) AS is_done,
                           COALESCE(delay_responsibility,'N/A') AS delay_responsibility
                    FROM project_tasks
                    """
                ),
                conn,
            )
        except Exception:
            tasks = pd.DataFrame(columns=["project_id", "title", "due_date", "is_done", "delay_responsibility"])

    total = int(len(projects))
    hoje = pd.Timestamp.today().normalize()
    atrasados = 0
    if not projects.empty and "planned_end_date" in projects.columns:
        planned = pd.to_datetime(projects["planned_end_date"], errors="coerce")
        actual = pd.to_datetime(projects.get("actual_end_date"), errors="coerce")
        atrasados = int(((planned < hoje) & (actual.isna())).sum())

    status_counts = (
        projects.assign(status=projects["status"].fillna("").replace("", "Sem status"))
        .groupby("status")["id"]
        .count()
        .sort_values(ascending=False)
        .head(10)
    )
    svg_status = _svg_bar([(k, float(v)) for k, v in status_counts.items()], min_height=170)

    svg_tasks = "<svg width='560' height='80'><text x='10' y='30' font-size='14'>Sem tarefas</text></svg>"
    if not tasks.empty and "due_date" in tasks.columns:
        due = pd.to_datetime(tasks["due_date"], errors="coerce")
        is_done = tasks["is_done"].fillna(False).astype(bool)
        overdue = tasks[(due.notna()) & (due < hoje) & (~is_done)]
        if not overdue.empty:
            by_resp = (
                overdue.assign(delay_responsibility=overdue["delay_responsibility"].fillna("N/A").replace("", "N/A"))
                .groupby("delay_responsibility")["title"]
                .count()
                .sort_values(ascending=False)
            )
            svg_tasks = _svg_bar([(k, float(v)) for k, v in by_resp.items()], min_height=140)

    rows = ""
    for _, r in projects.head(25).iterrows():
        rows += "<tr>"
        rows += f"<td>{int(r['id'])}</td>"
        rows += f"<td>{r.get('nome','')}</td>"
        rows += f"<td>{r.get('status','')}</td>"
        rows += f"<td>{_fmt_date(r.get('planned_end_date'))}</td>"
        rows += f"<td>{_fmt_date(r.get('actual_end_date'))}</td>"
        rows += f"<td>{float(r.get('progress_pct') or 0):.0f}%</td>"
        rows += f"<td>{r.get('delay_responsibility','N/A')}</td>"
        rows += "</tr>"

    content = f"""
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-label">Projetos</div><div class="kpi-value">{total}</div></div>
      <div class="kpi"><div class="kpi-label">Em atraso</div><div class="kpi-value">{atrasados}</div></div>
      <div class="kpi"><div class="kpi-label">Gerado em</div><div class="kpi-value" style="font-size:16px">{datetime.now().strftime("%d/%m/%Y %H:%M")}</div></div>
    </div>

    <h2>Status dos Projetos</h2>
    <div class="chart">{svg_status}</div>

    <h2>Tarefas vencidas por responsabilidade</h2>
    <div class="chart">{svg_tasks}</div>

    <h2>Projetos (últimos)</h2>
    <table class="table">
      <thead>
        <tr><th>ID</th><th>Projeto</th><th>Status</th><th>Previsto</th><th>Real</th><th>Progresso</th><th>Resp. atraso</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """
    return tpl.replace("{{content}}", content).replace("{{generated_at}}", datetime.now().strftime("%d/%m/%Y %H:%M"))
