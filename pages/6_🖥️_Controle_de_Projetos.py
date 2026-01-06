# pages/6_🖥️_Controle_de_Projetos.py
# Página: Controle de Projetos (centralizador)
# Observações:
# - Corrigi a inicialização do DB/Session: agora usa bk_erp_shared.erp_db.get_finance_db()
#   (essa função retorna (SessionLocal, engine) e isola a dependência do módulo de financeiro).
# - Usa bk_erp_shared.auth.login_and_guard para o login (coerente com o restante do app).
# - Mantive a lógica original de projeto/tarefas/relatório.
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import text

# Import local compartilhado
from bk_erp_shared.erp_db import ensure_erp_tables, get_finance_db
from bk_erp_shared.auth import login_and_guard

# Observação: mantive a importação de bk_finance caso outras funções do módulo sejam usadas
# mas não precisamos chamar get_finance_db diretamente do bk_finance.
import bk_finance


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


def list_projects(SessionLocal) -> pd.DataFrame:
    with SessionLocal() as conn:
        sql = """
            SELECT
                id,
                COALESCE(nome,'(sem nome)') AS nome,
                COALESCE(status,'') AS status,
                planned_end_date,
                actual_end_date,
                COALESCE(progress_pct,0) AS progress_pct,
                COALESCE(delay_responsibility,'N/A') AS delay_responsibility
            FROM projects
            ORDER BY id DESC
        """
        return pd.read_sql(text(sql), conn)


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


def list_tasks(SessionLocal, project_id: int) -> pd.DataFrame:
    _ensure_tasks_table(SessionLocal)
    with SessionLocal() as conn:
        try:
            sql = """
                SELECT id, title, due_date, COALESCE(is_done,FALSE) AS is_done,
                       COALESCE(assigned_to,'') AS assigned_to,
                       COALESCE(delay_responsibility,'N/A') AS delay_responsibility
                FROM project_tasks
                WHERE project_id=:pid
                ORDER BY COALESCE(is_done,FALSE) ASC, due_date NULLS LAST, id DESC
            """
            return pd.read_sql(text(sql), conn, params={"pid": int(project_id)})
        except Exception:
            sql = """
                SELECT id, title, due_date, COALESCE(is_done,0) AS is_done,
                       COALESCE(assigned_to,'') AS assigned_to,
                       COALESCE(delay_responsibility,'N/A') AS delay_responsibility
                FROM project_tasks
                WHERE project_id=:pid
                ORDER BY COALESCE(is_done,0) ASC, due_date, id DESC
            """
            return pd.read_sql(text(sql), conn, params={"pid": int(project_id)})


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


def _get_renderer():
    # Import lazy e seguro: não quebra se houver arquivo antigo ou erro no renderizador.
    try:
        from reports.render_controle_projetos import build_report  # type: ignore
        return build_report
    except Exception:
        # fallback mínimo idêntico ao original: constrói um HTML simples (compatível)
        def _fallback(SessionLocal, project_id=None):
            tpl_path = Path("reports/templates/controle_projetos_BK.html")
            tpl = tpl_path.read_text(encoding="utf-8") if tpl_path.exists() else "<html><body>{{content}}</body></html>"
            df = list_projects(SessionLocal)
            if project_id:
                df = df[df["id"] == project_id]
            rows = "".join([f"<tr><td>{int(r.id)}</td><td>{r.nome}</td><td>{r.status}</td></tr>" for r in df.itertuples()])
            content = f"<h2>Controle de Projetos</h2><table class='table'><thead><tr><th>ID</th><th>Projeto</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>"
            return tpl.replace("{{content}}", content)
        return _fallback


def main():
    # Page config
    st.set_page_config(page_title="Controle de Projetos", page_icon="🖥️", layout="wide")
    ensure_erp_tables()

    # <-- CORREÇÃO PRINCIPAL -->
    # Obtém SessionLocal e engine pela função compartilhada
    SessionLocal, _engine = get_finance_db()
    # Usa o wrapper de autenticação do pacote compartilhado para manter comportamento uniforme
    login_and_guard(SessionLocal)
    # ----------------------------->

    st.title("🖥️ Controle de Projetos")
    df = list_projects(SessionLocal)
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

        planned_default = row.get("planned_end_date")
        if pd.isna(planned_default) or planned_default is None:
            planned_default = date.today()
        actual_default = row.get("actual_end_date")
        if pd.isna(actual_default) or actual_default is None:
            actual_default = planned_default

        planned = st.date_input("Entrega prevista", value=planned_default)
        actual = st.date_input("Entrega real (se concluído)", value=actual_default)

        resp = st.selectbox("Responsável pelo atraso", ["N/A", "CLIENTE", "BK"], index=["N/A", "CLIENTE", "BK"].index(str(row.get("delay_responsibility") or "N/A")))

        if st.button("Salvar projeto", type="primary", use_container_width=True):
            update_project(SessionLocal, pid, status, progress, planned, actual, resp)
            st.success("Projeto atualizado.")
            st.rerun()

        st.divider()
        st.subheader("Relatório")
        build_report = _get_renderer()
        html = build_report(SessionLocal, project_id=pid)
        st.download_button("📄 Baixar relatório (HTML)", data=html.encode("utf-8"), file_name=f"controle_projetos_{pid}.html", mime="text/html", use_container_width=True)

    with right:
        st.subheader("Tarefas do Projeto")
        tasks = list_tasks(SessionLocal, pid)
        if tasks.empty:
            st.info("Nenhuma tarefa cadastrada.")
        else:
            st.dataframe(tasks, use_container_width=True, hide_index=True)

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
                if st.button("Concluir", use_container_width=True, key="t_done"):
                    mark_done(SessionLocal, opt[tsel])
                    st.success("Concluída.")
                    st.rerun()


if __name__ == "__main__":
    main()
