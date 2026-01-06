# reports/render_controle_projetos.py
# Gerador de HTML do relatório de Controle de Projetos
# Observações:
# - Agora injeta a logo (assets/logo.svg) como data-uri base64 no template através do placeholder {{logo_img}}.
# - Adiciona o texto "Criado pela BK Engenharia e Tecnologia" no rodapé via {{created_by_text}}.
from __future__ import annotations

import base64
from datetime import datetime, date
from io import BytesIO
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def chart_projects_status(projects: pd.DataFrame) -> str:
    if projects.empty:
        fig = plt.figure()
        plt.text(0.5, 0.5, "Sem dados", ha="center", va="center")
        plt.axis("off")
        return _fig_to_b64(fig)
    s = projects["status"].fillna("sem status").astype(str).str.lower()
    counts = s.value_counts().sort_values(ascending=False)
    fig = plt.figure()
    plt.bar(counts.index, counts.values)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Qtd")
    plt.title("Projetos por status")
    return _fig_to_b64(fig)


def chart_tasks_overdue(tasks: pd.DataFrame) -> str:
    if tasks.empty:
        fig = plt.figure()
        plt.text(0.5, 0.5, "Sem dados", ha="center", va="center")
        plt.axis("off")
        return _fig_to_b64(fig)

    today = date.today()
    t = tasks.copy()
    # campo/data podem variar; tentamos um mapeamento defensivo
    if "data_conclusao" in t.columns:
        t["due"] = pd.to_datetime(t["data_conclusao"], errors="coerce").dt.date
    elif "due_date" in t.columns:
        t["due"] = pd.to_datetime(t["due_date"], errors="coerce").dt.date
    else:
        t["due"] = pd.NaT

    overdue = t[(t["due"].notna()) & (t["due"] < today) & (t.get("status_tarefa", "").fillna("").str.lower() != "concluida")]
    if overdue.empty:
        fig = plt.figure()
        plt.text(0.5, 0.5, "Sem tarefas vencidas", ha="center", va="center")
        plt.axis("off")
        return _fig_to_b64(fig)

    grp = overdue.get("atraso_responsabilidade", pd.Series()).fillna("não informado").astype(str).str.upper().value_counts()
    fig = plt.figure()
    plt.bar(grp.index, grp.values)
    plt.ylabel("Qtd")
    plt.title("Tarefas vencidas por responsabilidade")
    return _fig_to_b64(fig)


def render_html(template_html: str, context: Dict[str, Any]) -> str:
    out = template_html
    for k, v in context.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


def make_rows_projects(projects_overdue: pd.DataFrame) -> str:
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


def _load_logo_datauri() -> str:
    """
    Carrega assets/logo.svg e retorna um <img> HTML (data-uri) para injeção direta no template.
    Se não houver o arquivo, retorna string vazia.
    """
    p = Path("assets/logo.svg")
    if not p.exists():
        return ""
    svg = p.read_text(encoding="utf-8")
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    datauri = f"data:image/svg+xml;base64,{b64}"
    # retorno em HTML (img tag) para facilitar o template
    return f'<img src="{datauri}" style="height:42px; display:inline-block; vertical-align:middle; margin-right:12px;" alt="BK logo"/>'

def build_report(template_html: str, projects: pd.DataFrame, tasks: pd.DataFrame, projects_overdue: pd.DataFrame, tasks_overdue: pd.DataFrame) -> str:
    """
    Recebe:
    - template_html: o conteúdo do arquivo HTML (string) com placeholders
    - projects, tasks, projects_overdue, tasks_overdue: DataFrames com os dados
    Retorna o HTML final com gráficos embutidos e logo.
    """
    status_b64 = chart_projects_status(projects)
    overdue_b64 = chart_tasks_overdue(tasks)

    proj_map = {int(r["id"]): str(r.get("nome") or "") for _, r in projects.iterrows() if str(r.get("id","")).isdigit()}

    ctx = {
        "generated_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "kpi_projetos_total": len(projects),
        "kpi_projetos_atrasados": len(projects_overdue),
        "kpi_tarefas_total": len(tasks),
        "kpi_tarefas_atrasadas": len(tasks_overdue),
        # Inserimos SVGs em <img> para template
        "chart_projects_status_svg": f'<img src="data:image/png;base64,{status_b64}" alt="projs status" />',
        "chart_tasks_overdue_svg": f'<img src="data:image/png;base64,{overdue_b64}" alt="tarefas vencidas" />',
        "projects_rows": make_rows_projects(projects_overdue),
        "tasks_rows": make_rows_tasks(tasks_overdue, proj_map),
        # Logo e texto de rodapé
        "logo_img": _load_logo_datauri(),
        "created_by_text": "Criado pela BK Engenharia e Tecnologia",
    }
    return render_html(template_html, ctx)
