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

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional, List, Tuple

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


# -------------------------
# Tarefas/Documentos (tabela estilo Excel)
# -------------------------
DOC_STATUS_OPTIONS = [
    "Em andamento - BK",
    "Em análise - Cliente",
    "Em revisão - BK",
    "Aprovado - Cliente",
]

REV_OPTIONS = [f"R0{c}" for c in list("ABCDEFGHIJKLMNO")]

def _status_to_responsible(status: str) -> str:
    s = (status or "").lower()
    if "cliente" in s:
        return "CLIENTE"
    if "bk" in s:
        return "BK"
    return "N/A"

def _next_revision(rev: str) -> str:
    rev = (rev or "R0A").strip().upper()
    if not rev.startswith("R0") or len(rev) < 3:
        return "R0A"
    letter = rev[2]
    letters = list("ABCDEFGHIJKLMNO")
    try:
        i = letters.index(letter)
    except Exception:
        return "R0A"
    if i >= len(letters) - 1:
        return f"R0{letters[-1]}"
    return f"R0{letters[i+1]}"

def _ensure_doc_tables(engine, SessionLocal) -> None:
    """Cria/ajusta tabelas necessárias para a tabela de Documentos do Controle de Projetos."""
    dialect = engine.dialect.name
    if dialect == "sqlite":
        id_col = "INTEGER PRIMARY KEY AUTOINCREMENT"
        ts = "TEXT"
    else:
        id_col = "SERIAL PRIMARY KEY"
        ts = "TIMESTAMP"
    ddl = f"""
    CREATE TABLE IF NOT EXISTS project_doc_tasks (
        id {id_col},
        project_id INTEGER NOT NULL,
        service_id INTEGER,
        service_name TEXT,
        doc_name TEXT,
        doc_number TEXT,
        revision TEXT DEFAULT 'R0A',
        start_date DATE,
        end_date DATE,
        status TEXT DEFAULT 'Em andamento - BK',
        responsible TEXT DEFAULT 'BK',
        created_at {ts} DEFAULT CURRENT_TIMESTAMP
    );
    """
    with SessionLocal() as session:
        session.execute(text(ddl))
        session.commit()

    # Migra colunas se a tabela já existia com esquema antigo
    try:
        cols = [c["name"] for c in inspect(engine).get_columns("project_doc_tasks")]
    except Exception:
        cols = []
    alter_stmts = []
    if "doc_name" not in cols:
        alter_stmts.append("ALTER TABLE project_doc_tasks ADD COLUMN doc_name TEXT;")
    if "doc_number" not in cols:
        alter_stmts.append("ALTER TABLE project_doc_tasks ADD COLUMN doc_number TEXT;")
    if "responsible" not in cols:
        alter_stmts.append("ALTER TABLE project_doc_tasks ADD COLUMN responsible TEXT;")
    if "service_name" not in cols:
        alter_stmts.append("ALTER TABLE project_doc_tasks ADD COLUMN service_name TEXT;")
    if alter_stmts:
        with SessionLocal() as session:
            for stmt in alter_stmts:
                try:
                    session.execute(text(stmt))
                except Exception:
                    pass
            session.commit()

def _list_services(engine) -> pd.DataFrame:
    """Lista serviços cadastrados (product_services)."""
    try:
        sql = text("""
            SELECT id, name, code, type, active
            FROM product_services
            WHERE COALESCE(active, 1) = 1
            ORDER BY name ASC
        """)
        return pd.read_sql(sql, engine)
    except Exception:
        return pd.DataFrame(columns=["id","name","code","type","active"])

def _list_doc_tasks(engine, project_id: int) -> pd.DataFrame:
    try:
        sql = text("""
            SELECT id, project_id, service_id, service_name, doc_name, doc_number,
                   revision, start_date, end_date, status, responsible
            FROM project_doc_tasks
            WHERE project_id = :pid
            ORDER BY id ASC
        """)
        return pd.read_sql(sql, engine, params={"pid": int(project_id)})
    except Exception:
        return pd.DataFrame(columns=[
            "id","project_id","service_id","service_name","doc_name","doc_number",
            "revision","start_date","end_date","status","responsible"
        ])

def _save_doc_tasks(SessionLocal, project_id: int, edited: pd.DataFrame, original: pd.DataFrame, svc_name_to_id: dict) -> Tuple[int,int,int]:
    inserted = updated = deleted = 0
    df = edited.copy()

    # detect delete
    ids_to_delete = []
    if "Excluir" in df.columns:
        ids_to_delete = [int(x) for x in df.loc[df["Excluir"] == True, "id"].dropna().tolist()]

    # Apply revision auto increment: "Em análise" -> "Em revisão"
    try:
        old_by_id = {int(r["id"]): r for _, r in original.dropna(subset=["id"]).iterrows()}
    except Exception:
        old_by_id = {}
    for idx,row in df.iterrows():
        rid = row.get("id")
        if pd.isna(rid):
            continue
        rid = int(rid)
        old = old_by_id.get(rid, {})
        old_status = str(old.get("status") or "")
        new_status = str(row.get("Status") or row.get("status") or "")
        if old_status == "Em análise - Cliente" and new_status == "Em revisão - BK":
            df.at[idx, "Revisão"] = _next_revision(str(old.get("revision") or row.get("Revisão") or "R0A"))

    # Fill responsible
    if "Status" in df.columns:
        df["Responsável"] = df["Status"].apply(lambda s: _status_to_responsible(str(s)))
    elif "status" in df.columns:
        df["responsible"] = df["status"].apply(lambda s: _status_to_responsible(str(s)))

    # Remove deletions from df to save
    if ids_to_delete:
        df_keep = df[~df["id"].isin(ids_to_delete)]
    else:
        df_keep = df

    with SessionLocal() as session:
        # deletes
        for did in ids_to_delete:
            session.execute(text("DELETE FROM project_doc_tasks WHERE id = :id"), {"id": int(did)})
        deleted = len(ids_to_delete)

        # inserts/updates
        for _, r in df_keep.iterrows():
            rid = r.get("id")
            service_name = str(r.get("Serviço") or r.get("service_name") or "").strip()
            service_id = svc_name_to_id.get(service_name) if service_name else None
            payload = {
                "project_id": int(project_id),
                "service_id": int(service_id) if service_id is not None else None,
                "service_name": service_name or None,
                "doc_name": str(r.get("Nome do documento") or r.get("doc_name") or "").strip() or None,
                "doc_number": str(r.get("Nº do documento") or r.get("doc_number") or "").strip() or None,
                "revision": str(r.get("Revisão") or r.get("revision") or "R0A").strip() or "R0A",
                "start_date": r.get("Data de início") or r.get("start_date"),
                "end_date": r.get("Data de conclusão") or r.get("end_date"),
                "status": str(r.get("Status") or r.get("status") or DOC_STATUS_OPTIONS[0]),
                "responsible": str(r.get("Responsável") or r.get("responsible") or _status_to_responsible(str(r.get("Status") or ""))),
            }
            if pd.isna(rid):
                session.execute(text("""
                    INSERT INTO project_doc_tasks (project_id, service_id, service_name, doc_name, doc_number, revision, start_date, end_date, status, responsible)
                    VALUES (:project_id, :service_id, :service_name, :doc_name, :doc_number, :revision, :start_date, :end_date, :status, :responsible)
                """), payload)
                inserted += 1
            else:
                payload["id"] = int(rid)
                session.execute(text("""
                    UPDATE project_doc_tasks
                    SET service_id=:service_id,
                        service_name=:service_name,
                        doc_name=:doc_name,
                        doc_number=:doc_number,
                        revision=:revision,
                        start_date=:start_date,
                        end_date=:end_date,
                        status=:status,
                        responsible=:responsible
                    WHERE id=:id
                """), payload)
                updated += 1
        session.commit()

    return inserted, updated, deleted


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

        # Status (combobox)


        status_opts = ["em_andamento", "em_aprovacao", "em_analise", "em_revisao", "aprovado", "concluido", "encerrado", "rascunho"]


        cur_status = str(row.get("status", "") or "em_andamento")


        if cur_status not in status_opts:


            status_opts = [cur_status] + status_opts


        status = st.selectbox("Status", options=status_opts, index=status_opts.index(cur_status))
        progress = st.slider("Progresso (%)", 0, 100, int(row.get("progress_pct") or 0))

        planned_default = row.get("planned_end_date")
        if pd.isna(planned_default) or planned_default is None or planned_default == "":
            planned_default = date.today()
        actual_default = row.get("actual_end_date")
        if pd.isna(actual_default) or actual_default is None or actual_default == "":
            actual_default = planned_default

        planned = st.date_input("Entrega prevista", value=planned_default)
        actual = st.date_input("Entrega real (se concluído)", value=actual_default)

        resp = st.selectbox("Responsável pelo atraso", ["N/A", "CLIENTE", "BK"], index=["N/A", "CLIENTE", "BK"].index(str(row.get("delay_responsibility") or "N/A")))

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
        st.subheader("Tarefas / Documentos (tabela estilo Excel)")

        _ensure_doc_tables(engine, SessionLocal)

        svc_df = _list_services(engine)
        svc_df["name"] = svc_df["name"].fillna("").astype(str)
        service_names = [s for s in svc_df["name"].tolist() if s.strip()]
        svc_name_to_id = {str(r.name): int(r.id) for r in svc_df.itertuples() if str(getattr(r, "name", "")).strip()}

        col_logo_bk, col_logo_cliente = st.columns(2)
        with col_logo_bk:
            st.file_uploader("Logo BK (PNG/JPG)", type=["png", "jpg", "jpeg"], key="logo_bk_upload")
        with col_logo_cliente:
            st.file_uploader("Logo Cliente (PNG/JPG)", type=["png", "jpg", "jpeg"], key="logo_cliente_upload")

        df_db = _list_doc_tasks(engine, pid)

        if df_db.empty:
            df_view = pd.DataFrame(columns=[
                "id",
                "Serviço",
                "Nome do documento",
                "Nº do documento",
                "Revisão",
                "Data de início",
                "Data de conclusão",
                "Status",
                "Responsável",
                "Excluir",
            ])
        else:
            df_view = pd.DataFrame({
                "id": df_db.get("id"),
                "Serviço": df_db.get("service_name"),
                "Nome do documento": df_db.get("doc_name"),
                "Nº do documento": df_db.get("doc_number"),
                "Revisão": df_db.get("revision").fillna("R0A"),
                "Data de início": df_db.get("start_date"),
                "Data de conclusão": df_db.get("end_date"),
                "Status": df_db.get("status").fillna(DOC_STATUS_OPTIONS[0]),
                "Responsável": df_db.get("responsible").fillna("BK"),
                "Excluir": False,
            })

        if df_view.empty:
            df_view.loc[0] = [None, "", "", "", "R0A", None, None, DOC_STATUS_OPTIONS[0], "BK", False]

        edited = st.data_editor(
            df_view,
            hide_index=True,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "id": st.column_config.NumberColumn("ID", disabled=True),
                "Serviço": st.column_config.SelectboxColumn("Serviço", options=service_names, required=False),
                "Revisão": st.column_config.SelectboxColumn("Revisão", options=REV_OPTIONS, required=False),
                "Status": st.column_config.SelectboxColumn("Status", options=DOC_STATUS_OPTIONS, required=True),
                "Responsável": st.column_config.TextColumn("Responsável", disabled=True),
                "Data de início": st.column_config.DateColumn("Data de início"),
                "Data de conclusão": st.column_config.DateColumn("Data de conclusão"),
                "Excluir": st.column_config.CheckboxColumn("Excluir"),
            },
        )

        if st.button("💾 Salvar tabela", width="stretch", key="doc_save"):
            ins, upd, dele = _save_doc_tasks(SessionLocal, pid, edited, df_db, svc_name_to_id)
            st.success(f"Tabela salva. Inseridos: {ins} | Atualizados: {upd} | Excluídos: {dele}")
            st.rerun()

        try:
            rev_series = edited["Revisão"].fillna("R0A").astype(str)
            total_revs = sum((REV_OPTIONS.index(r) + 1) if r in REV_OPTIONS else 1 for r in rev_series)
            st.caption(f"Total de revisões (estimado): {total_revs}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
