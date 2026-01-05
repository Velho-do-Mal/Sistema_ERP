from __future__ import annotations

import base64
from datetime import datetime, date
from io import BytesIO
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt


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
    # usa campo atraso_responsabilidade; se vazio, agrupa como "não informado"
    if tasks.empty:
        fig = plt.figure()
        plt.text(0.5, 0.5, "Sem dados", ha="center", va="center")
        plt.axis("off")
        return _fig_to_b64(fig)

    today = date.today()
    t = tasks.copy()
    t["due"] = pd.to_datetime(t["data_conclusao"], errors="coerce").dt.date
    overdue = t[(t["due"].notna()) & (t["due"] < today) & (t["status_tarefa"].fillna("").str.lower() != "concluida")]
    if overdue.empty:
        fig = plt.figure()
        plt.text(0.5, 0.5, "Sem tarefas vencidas", ha="center", va="center")
        plt.axis("off")
        return _fig_to_b64(fig)

    grp = overdue["atraso_responsabilidade"].fillna("não informado").astype(str).str.upper().value_counts()
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


def build_report(template_html: str, projects: pd.DataFrame, tasks: pd.DataFrame, projects_overdue: pd.DataFrame, tasks_overdue: pd.DataFrame) -> str:
    status_b64 = chart_projects_status(projects)
    overdue_b64 = chart_tasks_overdue(tasks)

    proj_map = {int(r["id"]): str(r.get("nome") or "") for _, r in projects.iterrows() if str(r.get("id","")).isdigit()}

    ctx = {
        "generated_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "kpi_projetos_total": len(projects),
        "kpi_projetos_atrasados": len(projects_overdue),
        "kpi_tarefas_total": len(tasks),
        "kpi_tarefas_atrasadas": len(tasks_overdue),
        "chart_projects_status_b64": status_b64,
        "chart_tasks_overdue_b64": overdue_b64,
        "projects_rows": make_rows_projects(projects_overdue),
        "tasks_rows": make_rows_tasks(tasks_overdue, proj_map),
    }
    return render_html(template_html, ctx)