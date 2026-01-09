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
from typing import Optional, List

import pandas as pd
import streamlit as st
from sqlalchemy import text, inspect

# camada compartilhada
from bk_erp_shared.erp_db import ensure_erp_tables, get_finance_db
from bk_erp_shared.auth import login_and_guard, current_user

import bk_finance  # utilitários do financeiro (format, etc.)

# -------------------------
# Controle de Documentos - Eventos de Status (Lead Time)
# -------------------------
def _ensure_doc_events_table(SessionLocal) -> None:
    ddl_pg = '''
    CREATE TABLE IF NOT EXISTS doc_status_events (
        id SERIAL PRIMARY KEY,
        project_id INTEGER NOT NULL,
        doc_code TEXT NOT NULL,
        revision TEXT DEFAULT '',
        status TEXT NOT NULL,
        responsible TEXT DEFAULT 'BK',
        note TEXT DEFAULT '',
        user_email TEXT DEFAULT '',
        entered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    '''
    ddl_sqlite = '''
    CREATE TABLE IF NOT EXISTS doc_status_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        doc_code TEXT NOT NULL,
        revision TEXT DEFAULT '',
        status TEXT NOT NULL,
        responsible TEXT DEFAULT 'BK',
        note TEXT DEFAULT '',
        user_email TEXT DEFAULT '',
        entered_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    '''
    with SessionLocal() as conn:
        try:
            conn.execute(text(ddl_pg))
        except Exception:
            conn.execute(text(ddl_sqlite))


def _insert_doc_event(SessionLocal, project_id: int, doc_code: str, revision: str, status: str,
                      responsible: str, note: str, user_email: str) -> None:
    _ensure_doc_events_table(SessionLocal)
    with SessionLocal() as conn:
        conn.execute(
            text(
                '''
                INSERT INTO doc_status_events (project_id, doc_code, revision, status, responsible, note, user_email, entered_at)
                VALUES (:pid, :doc, :rev, :status, :resp, :note, :email, CURRENT_TIMESTAMP)
                '''
            ),
            {
                "pid": int(project_id),
                "doc": doc_code.strip(),
                "rev": (revision or "").strip(),
                "status": status.strip().lower(),
                "resp": (responsible or "BK").strip().upper(),
                "note": (note or "").strip(),
                "email": (user_email or "").strip(),
            },
        )


def _load_doc_events(engine, project_id: int) -> pd.DataFrame:
    sql = '''
    SELECT id, project_id, doc_code, revision, status, responsible, note, user_email, entered_at
    FROM doc_status_events
    WHERE project_id = :pid
    ORDER BY doc_code, entered_at, id
    '''
    df = pd.read_sql(sql, engine, params={"pid": int(project_id)})
    df["entered_at"] = pd.to_datetime(df["entered_at"], errors="coerce")
    return df



def _compute_doc_lead_times(events: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula lead time (em dias corridos) por documento, somando:
      - por etapa: elaboracao / analise / revisao
      - por responsável: BK / CLIENTE
      - por combinação etapa x responsável (ex.: revisao_BK, analise_CLIENTE)
    """
    cols = [
        "doc_code", "ultima_rev", "status_atual",
        "dias_elaboracao", "dias_analise", "dias_revisao", "dias_total",
        "dias_BK", "dias_CLIENTE",
        "dias_elaboracao_BK", "dias_elaboracao_CLIENTE",
        "dias_analise_BK", "dias_analise_CLIENTE",
        "dias_revisao_BK", "dias_revisao_CLIENTE",
    ]

    if events.empty:
        return pd.DataFrame(columns=cols)

    # normaliza colunas esperadas
    e = events.copy()
    e["entered_at"] = pd.to_datetime(e["entered_at"], errors="coerce")
    e = e.sort_values(["doc_code", "entered_at", "id"], ascending=[True, True, True])

    etapas = ["elaboracao", "analise", "revisao"]
    responsaveis = ["BK", "CLIENTE"]

    out_rows = []
    for doc_code, g in e.groupby("doc_code", sort=True):
        g = g.reset_index(drop=True)

        dias_por_etapa = {k: 0.0 for k in etapas}
        dias_por_resp = {k: 0.0 for k in responsaveis}
        dias_combo = {(et, rp): 0.0 for et in etapas for rp in responsaveis}

        for i in range(len(g)):
            cur = g.iloc[i]
            nxt = g.iloc[i + 1] if i + 1 < len(g) else None

            t0 = cur.get("entered_at")
            if pd.isna(t0):
                continue

            # fim do intervalo: próximo evento ou agora
            t1 = nxt.get("entered_at") if nxt is not None else pd.Timestamp.now(tz=None)
            if pd.isna(t1):
                t1 = pd.Timestamp.now(tz=None)
            if t1 < t0:
                continue

            delta_days = (t1 - t0).total_seconds() / 86400.0

            stt = str(cur.get("status") or "").strip().lower()
            # normaliza sinônimos comuns
            if stt in ("elaboração", "elaboracao", "elaboração interna"):
                stt = "elaboracao"
            elif stt in ("análise", "analise", "análise cliente", "analise cliente"):
                stt = "analise"
            elif stt in ("revisão", "revisao", "revisão interna"):
                stt = "revisao"

            resp = str(cur.get("responsible") or "").strip().upper()
            if resp not in responsaveis:
                resp = "BK"

            if stt in dias_por_etapa:
                dias_por_etapa[stt] += delta_days
                dias_combo[(stt, resp)] += delta_days

            if resp in dias_por_resp:
                dias_por_resp[resp] += delta_days

        last = g.iloc[-1]
        status_atual = str(last.get("status") or "").strip().lower()
        ultima_rev = str(last.get("revision") or "")

        dias_total = sum(dias_por_etapa.values())

        out_rows.append({
            "doc_code": doc_code,
            "ultima_rev": ultima_rev,
            "status_atual": status_atual,
            "dias_elaboracao": round(dias_por_etapa["elaboracao"], 2),
            "dias_analise": round(dias_por_etapa["analise"], 2),
            "dias_revisao": round(dias_por_etapa["revisao"], 2),
            "dias_total": round(dias_total, 2),
            "dias_BK": round(dias_por_resp["BK"], 2),
            "dias_CLIENTE": round(dias_por_resp["CLIENTE"], 2),

            "dias_elaboracao_BK": round(dias_combo[("elaboracao", "BK")], 2),
            "dias_elaboracao_CLIENTE": round(dias_combo[("elaboracao", "CLIENTE")], 2),

            "dias_analise_BK": round(dias_combo[("analise", "BK")], 2),
            "dias_analise_CLIENTE": round(dias_combo[("analise", "CLIENTE")], 2),

            "dias_revisao_BK": round(dias_combo[("revisao", "BK")], 2),
            "dias_revisao_CLIENTE": round(dias_combo[("revisao", "CLIENTE")], 2),
        })

    df = pd.DataFrame(out_rows, columns=cols)
    # documentos mais demorados primeiro
    return df.sort_values(["dias_total", "doc_code"], ascending=[False, True])



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

            # documentos (lead time) - resumo para o relatório
            docs_summary = pd.DataFrame()
            if project_id is not None:
                try:
                    ev = _load_doc_events(engine, int(project_id))
                    if not ev.empty:
                        docs_summary = _compute_doc_lead_times(ev)
                except Exception:
                    docs_summary = pd.DataFrame()

            tpl_path = Path("reports/templates/controle_projetos_BK.html")
            template_html = tpl_path.read_text(encoding="utf-8") if tpl_path.exists() else "<html><body>{{content}}</body></html>"

            html = rc_build_report(template_html, projects, tasks_df, projects_overdue, tasks_overdue, docs_summary=docs_summary)
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

        planned_default = row.get("planned_end_date")
        actual_default = row.get("actual_end_date")

        def _coerce_date(v, fallback: date) -> date:
            """Converte valores diversos (date/datetime/Timestamp/str/NaT) para date."""
            try:
                if v is None:
                    return fallback
                if isinstance(v, date):
                    return v
                # pandas Timestamp / datetime
                if hasattr(v, "to_pydatetime"):
                    return v.to_pydatetime().date()
                s = str(v).strip()
                if s == "" or s.lower() in ("none", "nat", "nan"):
                    return fallback
                parsed = pd.to_datetime(s, errors="coerce")
                if pd.isna(parsed):
                    return fallback
                return parsed.date()
            except Exception:
                return fallback

        planned_default = _coerce_date(planned_default, date.today())
        actual_default = _coerce_date(actual_default, planned_default)

        planned = st.date_input("Entrega prevista", value=planned_default)
        actual = st.date_input("Entrega real (se concluído)", value=actual_default)

        _opts_resp = ["N/A", "CLIENTE", "BK"]
        _cur_resp = str(row.get("delay_responsibility") or "N/A").upper()
        if _cur_resp not in _opts_resp:
            _cur_resp = "N/A"
        resp = st.selectbox("Responsável pelo atraso", _opts_resp, index=_opts_resp.index(_cur_resp))

        if st.button("Salvar projeto", type="primary", use_container_width=True):
            update_project(SessionLocal, pid, status, progress, planned, actual, resp)
            st.success("Projeto atualizado.")
            st.rerun()

        st.divider()
        st.subheader("Relatório")
        build_report = _get_renderer(engine, SessionLocal)
        html = build_report(project_id=pid)
        st.download_button("📄 Baixar relatório (HTML)", data=html.encode("utf-8"), file_name=f"controle_projetos_{pid}.html", mime="text/html", use_container_width=True)

    with right:
        st.subheader("Tarefas do Projeto")
        tasks = list_tasks(engine, SessionLocal, pid)
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
        st.divider()
        st.subheader("📄 Controle de documentos (lead time)")

        st.caption("Registre cada mudança de status do documento. O sistema soma o tempo (dias corridos) em cada etapa até a próxima movimentação.")
        colA, colB, colC = st.columns([1.2, 0.8, 1.0])
        with colA:
            doc_code = st.text_input("Documento (código)", value="", key="doc_evt_code")
            revision = st.text_input("Revisão", value="", key="doc_evt_rev", placeholder="Ex.: R00, R01...")
        with colB:
            status_doc = st.selectbox("Status", ["elaboracao", "analise", "revisao", "aprovado"], index=0, key="doc_evt_status")
            # Sugestão automática: análise geralmente fica com o CLIENTE; revisão/elaboração com a BK.
            prev_status = st.session_state.get("_doc_evt_prev_status")
            if prev_status != status_doc:
                st.session_state["_doc_evt_prev_status"] = status_doc
                st.session_state["doc_evt_resp"] = "CLIENTE" if status_doc == "analise" else "BK"

            _resp_opts = ["BK", "CLIENTE"]
            _cur_resp = st.session_state.get("doc_evt_resp", "BK")
            if _cur_resp not in _resp_opts:
                _cur_resp = "BK"
            responsible = st.selectbox("Responsável pela etapa", _resp_opts, index=_resp_opts.index(_cur_resp), key="doc_evt_resp")
        with colC:
            note = st.text_input("Observação (opcional)", value="", key="doc_evt_note")

        if st.button("Registrar movimentação", type="primary", use_container_width=True, key="doc_evt_add"):
            if not doc_code.strip():
                st.warning("Informe o código do documento.")
            else:
                try:
                    _insert_doc_event(SessionLocal, pid, doc_code, revision, status_doc, responsible, note, current_user().get("email"))
                    st.success("Movimentação registrada.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao registrar: {e}")

        events = _load_doc_events(engine, pid)
        if events.empty:
            st.info("Nenhuma movimentação registrada para este projeto.")
        else:
            st.markdown("##### Resumo (dias por etapa e responsável)")
            summary = _compute_doc_lead_times(events)
            st.dataframe(summary, use_container_width=True, hide_index=True)

            # --- Gráficos (lead time por etapa x responsável)
            try:
                import plotly.express as px

                st.markdown("##### Gráficos (tempo por etapa x responsável)")

                total_analise_cliente = float(summary.get("dias_analise_CLIENTE", pd.Series(dtype=float)).sum())
                total_revisao_bk = float(summary.get("dias_revisao_BK", pd.Series(dtype=float)).sum())
                total_elab_bk = float(summary.get("dias_elaboracao_BK", pd.Series(dtype=float)).sum())
                total = float(summary.get("dias_total", pd.Series(dtype=float)).sum())

                outros = max(0.0, total - (total_analise_cliente + total_revisao_bk + total_elab_bk))

                agg = pd.DataFrame([
                    {"Etapa": "Elaboração (BK)", "Dias": total_elab_bk},
                    {"Etapa": "Análise (Cliente)", "Dias": total_analise_cliente},
                    {"Etapa": "Revisão (BK)", "Dias": total_revisao_bk},
                    {"Etapa": "Outros", "Dias": outros},
                ])

                fig_total = px.bar(agg, x="Etapa", y="Dias", text_auto=True, title="Tempo total (dias corridos)")
                st.plotly_chart(fig_total, use_container_width=True)

                c1, c2 = st.columns(2, gap="large")
                with c1:
                    top_a = summary.sort_values("dias_analise_CLIENTE", ascending=False).head(10)
                    fig_a = px.bar(top_a, x="doc_code", y="dias_analise_CLIENTE", text_auto=True,
                                   title="Top 10 documentos - Análise com Cliente (dias)")
                    st.plotly_chart(fig_a, use_container_width=True)
                with c2:
                    top_r = summary.sort_values("dias_revisao_BK", ascending=False).head(10)
                    fig_r = px.bar(top_r, x="doc_code", y="dias_revisao_BK", text_auto=True,
                                   title="Top 10 documentos - Revisão com BK (dias)")
                    st.plotly_chart(fig_r, use_container_width=True)

            except Exception as _e:
                st.info("Não foi possível gerar gráficos automaticamente para este relatório (dependência de Plotly).")


            with st.expander("Ver eventos (linha do tempo)", expanded=False):
                events_disp = events.copy()
                events_disp["entered_at"] = events_disp["entered_at"].dt.strftime("%d/%m/%Y %H:%M")
                st.dataframe(events_disp[["doc_code","revision","status","responsible","entered_at","user_email","note"]], use_container_width=True, hide_index=True)

            # download CSV do resumo
            csv = summary.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Baixar resumo (CSV)", data=csv, file_name=f"lead_time_documentos_projeto_{pid}.csv", mime="text/csv", use_container_width=True)



if __name__ == "__main__":
    main()
