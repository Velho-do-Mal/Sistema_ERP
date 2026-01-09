# reports/render_controle_projetos.py
# -*- coding: utf-8 -*-
"""
Renderer do relatório de Controle de Projetos.

- Gera gráficos (matplotlib) e converte para base64 para inclusão no template.
- Injeta a logo (assets/logo.svg) no template via data-uri.
- Preenche created_by_text com "Criado pela BK Engenharia e Tecnologia".
- Mantém compatibilidade com o template reports/templates/controle_projetos_BK.html.
"""

from __future__ import annotations

import base64
from datetime import datetime, date
from io import BytesIO
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt


def _fig_to_b64(fig) -> str:
    """Converte figura matplotlib em base64 PNG."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def chart_projects_status(projects: pd.DataFrame) -> str:
    """Gera gráfico projetos por status e retorna base64 PNG."""
    if projects.empty:
        fig = plt.figure()
        plt.text(0.5, 0.5, "Sem dados", ha="center", va="center")
        plt.axis("off")
        return _fig_to_b64(fig)
    s = projects["status"].fillna("sem status").astype(str).str.lower()
    counts = s.value_counts().sort_values(ascending=False)
    fig = plt.figure()
    plt.bar(counts.index, counts.values, color="#0f172a")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Qtd")
    plt.title("Projetos por status")
    return _fig_to_b64(fig)


def chart_tasks_overdue(tasks: pd.DataFrame) -> str:
    """Gera gráfico tarefas vencidas por responsabilidade (base64 PNG)."""
    if tasks.empty:
        fig = plt.figure()
        plt.text(0.5, 0.5, "Sem dados", ha="center", va="center")
        plt.axis("off")
        return _fig_to_b64(fig)

    today = date.today()
    t = tasks.copy()
    # tenta interpretar coluna data_conclusao (sevier a due_date)
    t["data_conclusao"] = pd.to_datetime(t.get("data_conclusao", t.get("due_date", pd.NaT)), errors="coerce")
    overdue = t[(t["data_conclusao"].notna()) & (t["data_conclusao"].dt.date < today) & (t.get("status_tarefa", "").fillna("").str.lower() != "concluida")]
    if overdue.empty:
        fig = plt.figure()
        plt.text(0.5, 0.5, "Sem tarefas vencidas", ha="center", va="center")
        plt.axis("off")
        return _fig_to_b64(fig)

    grp = overdue["atraso_responsabilidade"].fillna("não informado").astype(str).str.upper().value_counts()
    fig = plt.figure()
    plt.bar(grp.index, grp.values, color="#0b5ed7")
    plt.ylabel("Qtd")
    plt.title("Tarefas vencidas por responsabilidade")
    return _fig_to_b64(fig)


def render_html(template_html: str, context: Dict[str, Any]) -> str:
    """Substitui placeholders {{key}} no template por valores do context."""
    out = template_html
    for k, v in context.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


def chart_docs_leadtime(docs_summary: pd.DataFrame) -> str:
    """Gráfico (dias) destacando Análise com Cliente e Revisão com BK."""
    if docs_summary is None or docs_summary.empty:
        fig = plt.figure()
        plt.text(0.5, 0.5, "Sem dados de documentos", ha="center", va="center")
        plt.axis("off")
        return _fig_to_b64(fig)

    total_analise_cliente = float(docs_summary.get("dias_analise_CLIENTE", pd.Series(dtype=float)).sum())
    total_revisao_bk = float(docs_summary.get("dias_revisao_BK", pd.Series(dtype=float)).sum())
    total_elab_bk = float(docs_summary.get("dias_elaboracao_BK", pd.Series(dtype=float)).sum())
    total = float(docs_summary.get("dias_total", pd.Series(dtype=float)).sum())
    outros = max(0.0, total - (total_analise_cliente + total_revisao_bk + total_elab_bk))

    labels = ["Elaboração (BK)", "Análise (Cliente)", "Revisão (BK)", "Outros"]
    values = [total_elab_bk, total_analise_cliente, total_revisao_bk, outros]

    fig = plt.figure()
    plt.bar(labels, values, color="#0f172a")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Dias")
    plt.title("Tempo total por etapa x responsável (dias corridos)")
    return _fig_to_b64(fig)


def make_rows_docs(docs_summary: pd.DataFrame, top_n: int = 12) -> str:
    """Linhas HTML da tabela de documentos (gargalos)."""
    if docs_summary is None or docs_summary.empty:
        return '<tr><td colspan="6" class="muted">Sem dados de documentos.</td></tr>'

    df = docs_summary.copy()
    # ranking por gargalo (análise cliente + revisão BK)
    df["_score"] = df.get("dias_analise_CLIENTE", 0).fillna(0) + df.get("dias_revisao_BK", 0).fillna(0)
    df = df.sort_values("_score", ascending=False).head(top_n)

    rows = []
    for _, r in df.iterrows():
        rows.append(
            "<tr>"
            f"<td>{r.get('doc_code','')}</td>"
            f"<td>{r.get('ultima_rev','')}</td>"
            f"<td>{(r.get('status_atual','') or '').upper()}</td>"
            f"<td>{r.get('dias_analise_CLIENTE','')}</td>"
            f"<td>{r.get('dias_revisao_BK','')}</td>"
            f"<td>{r.get('dias_total','')}</td>"
            "</tr>"
        )
    return "\n".join(rows)

def make_rows_projects(projects_overdue: pd.DataFrame) -> str:
    """Cria linhas HTML para projetos em atraso (tabela)."""
    def badge(x: str) -> str:
        return f'<span class="badge">{x}</span>'

    rows = []
    for _, r in projects_overdue.iterrows():
        rows.append(
            "<tr>"
            f"<td>{r.get('id','')}</td>"
            f"<td>{r.get('nome','')}</td>"
            f"<td>{r.get('cod_projeto','') or ''}</td>"
            f"<td>{badge(r.get('status','') or '')}</td>"
            f"<td>{r.get('data_inicio','') or ''}</td>"
            f"<td>{r.get('data_conclusao','') or ''}</td>"
            f"<td>{int(r.get('dias_atraso',0) or 0)}</td>"
            f"<td>{(r.get('atraso_responsabilidade') or '').upper()}</td>"
            "</tr>"
        )
    return "\n".join(rows) if rows else '<tr><td colspan="8" class="muted">Sem projetos em atraso.</td></tr>'


def make_rows_tasks(tasks_overdue: pd.DataFrame, project_name_by_id: Dict[int, str]) -> str:
    """Cria linhas HTML para tarefas em atraso."""
    def badge(x: str) -> str:
        return f'<span class="badge">{x}</span>'

    rows = []
    for _, r in tasks_overdue.iterrows():
        pid = r.get("project_id")
        pname = project_name_by_id.get(int(pid), "") if pid and str(pid).isdigit() else ""
        rows.append(
            "<tr>"
            f"<td>{r.get('id','')}</td>"
            f"<td>{pname}</td>"
            f"<td>{r.get('title','')}</td>"
            f"<td>{r.get('data_conclusao','') or ''}</td>"
            f"<td>{badge(r.get('status_tarefa','') or '')}</td>"
            f"<td>{r.get('id_responsavel','') or ''}</td>"
            f"<td>{(r.get('atraso_responsabilidade') or 'não informado').upper()}</td>"
            "</tr>"
        )
    return "\n".join(rows) if rows else '<tr><td colspan="7" class="muted">Sem tarefas vencidas.</td></tr>'


def build_report(template_html: str, projects: pd.DataFrame, tasks: pd.DataFrame, projects_overdue: pd.DataFrame, tasks_overdue: pd.DataFrame, docs_summary: pd.DataFrame | None = None) -> str:
    """
    Gera o HTML final do relatório.
    - template_html: conteúdo do arquivo HTML com placeholders
    - projects, tasks, projects_overdue, tasks_overdue: dataframes
    """
    status_b64 = chart_projects_status(projects)
    overdue_b64 = chart_tasks_overdue(tasks)
    docs_b64 = chart_docs_leadtime(docs_summary if docs_summary is not None else pd.DataFrame())
    docs_rows = make_rows_docs(docs_summary if docs_summary is not None else pd.DataFrame())


    # mapa id -> nome do projeto (usado nas tarefas)
    proj_map = {int(r["id"]): str(r.get("nome") or "") for _, r in projects.iterrows() if str(r.get("id","")).isdigit()}

    # Logo (SVG) — tenta carregar assets/logo.svg e converter para data-uri
    logo_img = ""
    try:
        logo_path = Path("assets/logo.svg")
        if logo_path.exists():
            svg_text = logo_path.read_text(encoding="utf-8")
            svg_b64 = base64.b64encode(svg_text.encode("utf-8")).decode("utf-8")
            # injeta a tag <img> (a altura definida para manter layout)
            logo_img = f'<img src="data:image/svg+xml;base64,{svg_b64}" style="height:48px; display:block;" />'
    except Exception:
        logo_img = ""

    ctx = {
        "generated_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "kpi_projetos_total": len(projects),
        "kpi_projetos_atrasados": len(projects_overdue),
        "kpi_tarefas_total": len(tasks),
        "kpi_tarefas_atrasadas": len(tasks_overdue),
        # converte charts base64 para tags <img> usadas pelo template
        "chart_projects_status_svg": f'<img src="data:image/png;base64,{status_b64}" alt="Projetos por status" />',
        "chart_tasks_overdue_svg": f'<img src="data:image/png;base64,{overdue_b64}" alt="Tarefas vencidas" />',
        "chart_docs_leadtime_svg": f'<img src="data:image/png;base64,{docs_b64}" alt="Documentos - lead time" />',
        "docs_rows": docs_rows,
        "projects_rows": make_rows_projects(projects_overdue),
        "tasks_rows": make_rows_tasks(tasks_overdue, proj_map),
        # footer / logo
        "logo_img": logo_img,
        "created_by_text": "Criado pela BK Engenharia e Tecnologia",
    }

    return render_html(template_html, ctx)