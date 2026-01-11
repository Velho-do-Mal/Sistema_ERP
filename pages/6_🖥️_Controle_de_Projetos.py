# pages/6_🖥️_Controle_de_Projetos.py
# -*- coding: utf-8 -*-
"""
BK_ERP - 🖥️ Controle de Projetos (centralizador)

Atualizações:
- list_projects(engine) foi tornado defensivo: detecta colunas reais na tabela
  e monta a SQL dinamicamente para evitar "no such column".
- Usa engine para pd.read_sql, SessionLocal para DML.
- Mantém renderer compatível e footer/rodapé.
"""


from datetime import date
import datetime as dt
from pathlib import Path
from typing import Optional, List

import pandas as pd
import streamlit as st
from sqlalchemy import text, inspect

# camada compartilhada
from bk_erp_shared.erp_db import ensure_erp_tables, get_finance_db
from bk_erp_shared.auth import login_and_guard

import bk_finance  # utilitários do financeiro (format, etc.)


# -------------------------
# Helpers para montar SQL defensivo
# -------------------------

def _safe_date(value, fallback=None):
    """Converte value em datetime.date (aceita date/datetime/pandas.Timestamp/str)."""
    if fallback is None:
        fallback = dt.date.today()
    if value is None or value == "":
        return fallback
    try:
        # pandas.NaT / nan
        if pd.isna(value):
            return fallback
    except Exception:
        pass
    try:
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime().date()
    except Exception:
        pass
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return fallback
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return dt.datetime.strptime(s, fmt).date()
            except Exception:
                pass
    return fallback



def _pick_column(cols: List[str], candidates: List[str]) -> Optional[str]:
    """Retorna o primeiro nome de coluna presente em cols a partir de candidates."""
    for c in candidates:
        if c in cols:
            return c
        # tentar variações case-insensitive
        for col in cols:
            if col.lower() == c.lower():
                return col
    return None


# -------------------------
# Tabelas/CRUDs locais
# -------------------------
def _ensure_tasks_table(SessionLocal) -> None:
    ddl_pg = """
    CREATE TABLE IF NOT EXISTS project_tasks (
        id SERIAL PRIMARY KEY,
        project_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        due_date DATE,
        is_done BOOLEAN DEFAULT FALSE,
        assigned_to TEXT,
        delay_responsibility TEXT DEFAULT 'N/A',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    ddl_sqlite = """
    CREATE TABLE IF NOT EXISTS project_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        due_date DATE,
        is_done BOOLEAN DEFAULT 0,
        assigned_to TEXT,
        delay_responsibility TEXT DEFAULT 'N/A',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    with SessionLocal() as conn:
        try:
            conn.execute(text(ddl_pg))
        except Exception:
            conn.execute(text(ddl_sqlite))


def list_projects(engine) -> pd.DataFrame:
    """
    Lista projetos de forma defensiva:
    - Detecta colunas existentes na tabela `projects` através do inspector.
    - Monta expressões SQL conforme colunas encontradas para evitar SELECT de colunas inexistentes.
    - Retorna DataFrame com colunas: id, nome, status, planned_end_date, actual_end_date, progress_pct, delay_responsibility
    """
    insp = inspect(engine)
    try:
        cols_meta = insp.get_columns("projects")
        cols = [c["name"] for c in cols_meta]
    except Exception:
        # se inspector falhar, tenta uma query simples para obter algumas colunas (fallback)
        cols = []
    # candidatos para cada campo lógico
    planned_candidates = ["planned_end_date", "planned_end", "dataInicio", "data"]
    actual_candidates = ["actual_end_date", "actual_end", "data_conclusao", "data"]
    progress_candidates = ["progress_pct", "progress", "pct", "percent"]
    delay_candidates = ["delay_responsibility", "atraso_responsibility", "delay_resp", "atraso_responsibility"]

    planned_col = _pick_column(cols, planned_candidates)
    actual_col = _pick_column(cols, actual_candidates)
    progress_col = _pick_column(cols, progress_candidates)
    delay_col = _pick_column(cols, delay_candidates)

    select_exprs = [
        "id",
        "COALESCE(nome,'(sem nome)') AS nome",
        "COALESCE(status,'') AS status",
    ]

    if planned_col:
        select_exprs.append(f"COALESCE({planned_col}, '') AS planned_end_date")
    else:
        select_exprs.append("'' AS planned_end_date")

    if actual_col:
        select_exprs.append(f"COALESCE({actual_col}, '') AS actual_end_date")
    else:
        select_exprs.append("'' AS actual_end_date")

    if progress_col:
        select_exprs.append(f"COALESCE({progress_col},0) AS progress_pct")
    else:
        select_exprs.append("0 AS progress_pct")

    if delay_col:
        select_exprs.append(f"COALESCE({delay_col},'N/A') AS delay_responsibility")
    else:
        select_exprs.append("'N/A' AS delay_responsibility")

    sql = "SELECT\n    " + ",\n    ".join(select_exprs) + "\nFROM projects\nORDER BY id DESC"
    # Usa engine (SQL string) — pandas aceita string + engine
    df = pd.read_sql(sql, engine)
    return df


def update_project(SessionLocal, project_id: int, status: str, progress_pct: int,
                   planned_end_date: date, actual_end_date: date, delay_responsibility: str) -> None:
    with SessionLocal() as conn:
        conn.execute(
            text(
                """
                UPDATE projects
                SET status=:status,
                    progress_pct=:progress_pct,
                    planned_end_date=:planned_end_date,
                    actual_end_date=:actual_end_date,
                    delay_responsibility=:delay_responsibility
                WHERE id=:id
                """
            ),
            {
                "id": int(project_id),
                "status": status,
                "progress_pct": int(progress_pct),
                "planned_end_date": planned_end_date,
                "actual_end_date": actual_end_date,
                "delay_responsibility": delay_responsibility or "N/A",
            },
        )


def list_tasks(engine, SessionLocal, project_id: int) -> pd.DataFrame:
    """
    Lista tarefas de um projeto. Garante criação da tabela e usa engine para pd.read_sql.
    """
    _ensure_tasks_table(SessionLocal)
    sql = """
        SELECT id, title, due_date, COALESCE(is_done,FALSE) AS is_done,
               COALESCE(assigned_to,'') AS assigned_to,
               COALESCE(delay_responsibility,'N/A') AS delay_responsibility
        FROM project_tasks
        WHERE project_id=:pid
        ORDER BY COALESCE(is_done,FALSE) ASC, due_date, id DESC
    """
    df = pd.read_sql(sql, engine, params={"pid": int(project_id)})
    return df


def add_task(SessionLocal, project_id: int, title: str, due_date: date,
             delay_responsibility: str = "N/A", assigned_to: str = "") -> None:
    _ensure_tasks_table(SessionLocal)
    with SessionLocal() as conn:
        conn.execute(
            text(
                """
                INSERT INTO project_tasks (project_id, title, due_date, is_done, assigned_to, delay_responsibility)
                VALUES (:pid, :title, :due, FALSE, :assigned_to, :resp)
                """
            ),
            {"pid": int(project_id), "title": title, "due": due_date, "assigned_to": assigned_to, "resp": delay_responsibility or "N/A"},
        )


def mark_done(SessionLocal, task_id: int) -> None:
    _ensure_tasks_table(SessionLocal)
    with SessionLocal() as conn:
        conn.execute(text("UPDATE project_tasks SET is_done=TRUE WHERE id=:id"), {"id": int(task_id)})


# -------------------------
# Renderer
# -------------------------
def _get_renderer(engine, SessionLocal):
    try:
        from reports.render_controle_projetos import build_report as rc_build_report  # type: ignore

        def _wrapper(project_id: Optional[int] = None) -> str:
            projects = list_projects(engine)
            if project_id is not None:
                projects = projects[projects["id"] == int(project_id)]

            tasks_frames = []
            for pid in projects["id"].tolist():
                t = list_tasks(engine, SessionLocal, int(pid))
                if not t.empty:
                    t = t.copy()
                    t["project_id"] = int(pid)
                    if "due_date" in t.columns and "data_conclusao" not in t.columns:
                        t["data_conclusao"] = t["due_date"]
                    if "status_tarefa" not in t.columns:
                        t["status_tarefa"] = t["is_done"].map(lambda v: "Concluída" if bool(v) else "Aberta")
                    tasks_frames.append(t)
            tasks_df = pd.concat(tasks_frames, ignore_index=True) if tasks_frames else pd.DataFrame(columns=["id","title","due_date","is_done","assigned_to","delay_responsibility","project_id","data_conclusao","status_tarefa"])

            today = date.today()
            try:
                projects_local = projects.copy()
                projects_local["planned_end_date_dt"] = pd.to_datetime(projects_local.get("planned_end_date", pd.NaT), errors="coerce").dt.date
                projects_overdue = projects_local[projects_local["planned_end_date_dt"].notna() & (projects_local["planned_end_date_dt"] < today)]
            except Exception:
                projects_overdue = projects.iloc[0:0]

            try:
                tasks_local = tasks_df.copy()
                tasks_local["data_conclusao_dt"] = pd.to_datetime(tasks_local.get("data_conclusao", pd.NaT), errors="coerce").dt.date
                tasks_overdue = tasks_local[tasks_local["data_conclusao_dt"].notna() & (tasks_local["data_conclusao_dt"] < today) & (tasks_local["status_tarefa"].str.lower() != "concluida")]
            except Exception:
                tasks_overdue = tasks_df.iloc[0:0]

            tpl_path = Path("reports/templates/controle_projetos_BK.html")
            template_html = tpl_path.read_text(encoding="utf-8") if tpl_path.exists() else "<html><body>{{content}}</body></html>"

            html = rc_build_report(template_html, projects, tasks_df, projects_overdue, tasks_overdue)
            return html

        return _wrapper

    except Exception:
        def _fallback(project_id: Optional[int] = None):
            tpl_path = Path("reports/templates/controle_projetos_BK.html")
            tpl = tpl_path.read_text(encoding="utf-8") if tpl_path.exists() else "<html><body>{{content}}</body></html>"
            df = list_projects(engine)
            if project_id is not None:
                df = df[df["id"] == int(project_id)]
            rows = "".join([f"<tr><td>{int(r.id)}</td><td>{r.nome}</td><td>{r.status}</td></tr>" for r in df.itertuples()])
            content = f"<h2>Controle de Projetos</h2><table class='table'><thead><tr><th>ID</th><th>Projeto</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>"
            return tpl.replace("{{content}}", content)
        return _fallback


# -------------------------
# Página Principal
# -------------------------
def main():
    st.set_page_config(page_title="Controle de Projetos", page_icon="🖥️", layout="wide")
    ensure_erp_tables()

    engine, SessionLocal = get_finance_db()

    login_and_guard(SessionLocal)

    st.title("🖥️ Controle de Projetos")

    df = list_projects(engine)
    if df.empty:
        st.info("Nenhum projeto cadastrado. Cadastre um projeto em **Gestão de Projetos**.")
        return

    left, right = st.columns([1.2, 2.0], gap="large")

    with left:
        st.subheader("Projetos")
        project_opts = {f"#{int(r.id)} - {r.nome}": int(r.id) for r in df.itertuples()}
        sel_label = st.selectbox("Selecione", list(project_opts.keys()), index=0)
        pid = project_opts[sel_label]
        row = df[df["id"] == pid].iloc[0]

        status = st.text_input("Status", value=str(row.get("status", "") or ""))
        progress = st.slider("Progresso (%)", 0, 100, int(row.get("progress_pct") or 0))

        planned_default = _safe_date(row.get("planned_end_date"), dt.date.today())
        actual_default = _safe_date(row.get("actual_end_date"), planned_default)

        planned = st.date_input("Entrega prevista", value=planned_default)
        actual = st.date_input("Entrega real (se concluído)", value=actual_default)

        resp_opts = ["N/A", "CLIENTE", "BK"]
        resp_raw = str(row.get("delay_responsibility") or "N/A").strip().upper()
        if resp_raw.startswith("CLI"):
            resp_raw = "CLIENTE"
        elif resp_raw.startswith("BK"):
            resp_raw = "BK"
        else:
            resp_raw = "N/A"
        resp = st.selectbox("Responsável pelo atraso", resp_opts, index=resp_opts.index(resp_raw))

        if st.button("Salvar projeto", type="primary", width='stretch'):
            update_project(SessionLocal, pid, status, progress, planned, actual, resp)
            st.success("Projeto atualizado.")
            st.rerun()

        st.divider()
        st.subheader("Relatório")
        build_report = _get_renderer(engine, SessionLocal)
        html = build_report(project_id=pid)
        st.download_button("📄 Baixar relatório (HTML)", data=html.encode("utf-8"), file_name=f"controle_projetos_{pid}.html", mime="text/html", width='stretch')

    with right:
        st.subheader("Tarefas do Projeto")
        tasks = list_tasks(engine, SessionLocal, pid)
        if tasks.empty:
            st.info("Nenhuma tarefa cadastrada.")
        else:
            st.dataframe(tasks, width='stretch', hide_index=True)

        with st.expander("➕ Nova tarefa", expanded=True):
            title = st.text_input("Título", key="t_title")
            due = st.date_input("Prazo", value=date.today(), key="t_due")
            t_resp = st.selectbox("Responsável pelo atraso (se vencer)", ["N/A", "CLIENTE", "BK"], index=0, key="t_resp")
            assigned_to = st.text_input("Responsável (nome/e-mail)", value="", key="t_assigned")
            if st.button("Adicionar tarefa", key="t_add"):
                if not title.strip():
                    st.warning("Informe um título.")
                else:
                    add_task(SessionLocal, pid, title.strip(), due, t_resp, assigned_to.strip())
                    st.success("Tarefa criada.")
                    st.rerun()

        if not tasks.empty:
            open_tasks = tasks[tasks["is_done"] == False]
            if not open_tasks.empty:
                st.divider()
                st.subheader("Marcar como concluída")
                opt = {f"#{int(r.id)} - {r.title}": int(r.id) for r in open_tasks.itertuples()}
                tsel = st.selectbox("Selecione a tarefa", list(opt.keys()), key="t_sel")
                if st.button("Concluir", width='stretch', key="t_done"):
                    mark_done(SessionLocal, opt[tsel])
                    st.success("Concluída.")
                    st.rerun()


if __name__ == "__main__":
    main()
