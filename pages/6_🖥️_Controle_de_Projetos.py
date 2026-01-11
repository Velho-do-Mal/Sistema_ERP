# pages/6_🖥️_Controle_de_Projetos.py
# -*- coding: utf-8 -*-
"""
BK_ERP - 🖥️ Controle de Projetos (BK x Cliente)

IMPORTANTE (conforme alinhado com você):
- Este módulo é INDEPENDENTE do módulo Gestão de Projetos/EAP.
- O único vínculo é: a lista de projetos vem do cadastro de projetos (tabela `projects`).
- Aqui nós controlamos DOCUMENTOS/TAREFAS (tabela estilo Excel), logos e relatório HTML.

Principais features nesta página:
- Seleção do projeto (projects)
- Formulário de dados do relatório (cliente, nº projeto etc) + upload de logos
- Tabela estilo Excel (st.data_editor) com colunas:
  ID, Serviço (do cadastro), Nome do documento, Nº do documento, Revisão,
  Data de início, Data de conclusão, Status, Responsável, Observação
- Botão "Salvar tabela" (insert/update/delete)
- Botão "Baixar relatório (HTML)" com cabeçalho (logos, BK, cliente, projeto) + tabela + gráficos

Correções importantes:
- login_and_guard(SessionLocal) (o auth.py exige SessionLocal)
- SQL parametrizada para Postgres/Neon usando sqlalchemy.text() (evita erro ":pid")
"""

import base64
import datetime as _dt
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from bk_erp_shared.erp_db import get_finance_db, ensure_erp_tables
from bk_erp_shared.auth import login_and_guard
from bk_erp_shared.sales import list_product_services



def _rerun():
    """Compat: Streamlit novo usa st.rerun(); versões antigas usam _rerun()."""
    try:
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            _rerun()
    except Exception:
        _rerun()


# --------------------------
# Utilidades
# --------------------------
def _safe_date(v) -> Optional[_dt.date]:
    """Converte date/datetime/pandas Timestamp/str em datetime.date (ou None)."""
    if v is None:
        return None
    # pandas NaT
    try:
        import pandas as _pd
        if _pd.isna(v):
            return None
    except Exception:
        pass

    if isinstance(v, _dt.date) and not isinstance(v, _dt.datetime):
        return v
    if isinstance(v, _dt.datetime):
        return v.date()

    # pandas.Timestamp
    try:
        import pandas as _pd
        if isinstance(v, _pd.Timestamp):
            if _pd.isna(v):
                return None
            return v.to_pydatetime().date()
    except Exception:
        pass

    # string ISO
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return _dt.datetime.strptime(s, fmt).date()
            except Exception:
                continue
        # último fallback: fromisoformat
        try:
            return _dt.date.fromisoformat(s)
        except Exception:
            return None

    return None


def _bytes_to_data_uri(content: bytes, mime: str) -> str:
    b64 = base64.b64encode(content).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _guess_mime(filename: str) -> str:
    fn = (filename or "").lower()
    if fn.endswith(".png"):
        return "image/png"
    if fn.endswith(".jpg") or fn.endswith(".jpeg"):
        return "image/jpeg"
    if fn.endswith(".webp"):
        return "image/webp"
    if fn.endswith(".svg"):
        return "image/svg+xml"
    return "application/octet-stream"


# --------------------------
# Banco / Tabelas deste módulo
# --------------------------
def _ensure_control_tables(engine, SessionLocal) -> None:
    """Cria / migra tabelas necessárias para este módulo (PG e SQLite)."""
    # Meta do relatório por projeto
    ddl_meta_pg = """
    CREATE TABLE IF NOT EXISTS project_control_meta (
        project_id INTEGER PRIMARY KEY,
        client_name TEXT,
        project_name TEXT,
        project_number TEXT,
        logo_bk BYTEA,
        logo_bk_mime TEXT,
        logo_client BYTEA,
        logo_client_mime TEXT,
        updated_at TIMESTAMP DEFAULT NOW()
    );
    """
    ddl_meta_sqlite = """
    CREATE TABLE IF NOT EXISTS project_control_meta (
        project_id INTEGER PRIMARY KEY,
        client_name TEXT,
        project_name TEXT,
        project_number TEXT,
        logo_bk BLOB,
        logo_bk_mime TEXT,
        logo_client BLOB,
        logo_client_mime TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Tabela principal (documentos/tarefas)
    ddl_tasks_pg = """
    CREATE TABLE IF NOT EXISTS project_doc_tasks (
        id SERIAL PRIMARY KEY,
        project_id INTEGER NOT NULL,
        service_id INTEGER NULL,
        service_label TEXT NULL,
        doc_name TEXT,
        doc_number TEXT,
        revision TEXT,
        status TEXT,
        responsible TEXT,
        start_date DATE,
        end_date DATE,
        notes TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    ddl_tasks_sqlite = """
    CREATE TABLE IF NOT EXISTS project_doc_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        service_id INTEGER,
        service_label TEXT,
        doc_name TEXT,
        doc_number TEXT,
        revision TEXT,
        status TEXT,
        responsible TEXT,
        start_date TEXT,
        end_date TEXT,
        notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """

    with SessionLocal() as conn:
        # meta
        try:
            conn.execute(text(ddl_meta_pg))
        except Exception:
            conn.execute(text(ddl_meta_sqlite))

        # tasks
        try:
            conn.execute(text(ddl_tasks_pg))
        except Exception:
            conn.execute(text(ddl_tasks_sqlite))

        # migração defensiva (se já existia versão antiga)
        # Adiciona colunas faltantes sem quebrar
        _maybe_add_column(conn, engine, "project_doc_tasks", "service_id", "INTEGER")
        _maybe_add_column(conn, engine, "project_doc_tasks", "service_label", "TEXT")
        _maybe_add_column(conn, engine, "project_doc_tasks", "doc_number", "TEXT")
        _maybe_add_column(conn, engine, "project_doc_tasks", "responsible", "TEXT")
        _maybe_add_column(conn, engine, "project_doc_tasks", "notes", "TEXT")
        _maybe_add_column(conn, engine, "project_doc_tasks", "status", "TEXT")
        _maybe_add_column(conn, engine, "project_doc_tasks", "start_date", "DATE")
        _maybe_add_column(conn, engine, "project_doc_tasks", "end_date", "DATE")

        try:
            conn.commit()
        except Exception:
            pass


def _maybe_add_column(conn, engine, table: str, col: str, coltype: str) -> None:
    """Adiciona coluna se não existir (PG/SQLite)."""
    try:
        cols = _get_table_columns(engine, table)
        if col in cols:
            return
        # Postgres
        try:
            conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {col} {coltype};'))
        except Exception:
            # SQLite
            conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {col} {coltype};'))
    except Exception:
        # não bloquear a app
        return


def _get_table_columns(engine, table: str) -> set:
    with engine.connect() as c:
        try:
            rows = c.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = :t
            """), {"t": table}).fetchall()
            if rows:
                return {r[0] for r in rows}
        except Exception:
            pass
        try:
            rows = c.execute(text(f"PRAGMA table_info({table});")).fetchall()
            return {r[1] for r in rows}
        except Exception:
            return set()


# --------------------------
# Consultas
# --------------------------
def _list_projects(engine) -> pd.DataFrame:
    # tabela projects (vem do ERP)
    sql = text("""
        SELECT id,
               COALESCE(nome,'') AS nome,
               COALESCE(status,'') AS status
        FROM projects
        ORDER BY id DESC
    """)
    with engine.connect() as conn:
        return pd.read_sql_query(sql, conn)


def _load_project(engine, project_id: int) -> Dict[str, Any]:
    sql = text("""
        SELECT id, COALESCE(nome,'') AS nome, COALESCE(status,'') AS status
        FROM projects
        WHERE id = :pid
        LIMIT 1
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params={"pid": int(project_id)})
    if df.empty:
        return {}
    return dict(df.iloc[0].to_dict())


def _load_meta(engine, project_id: int) -> Dict[str, Any]:
    sql = text("""
        SELECT project_id, client_name, project_name, project_number,
               logo_bk, logo_bk_mime, logo_client, logo_client_mime
        FROM project_control_meta
        WHERE project_id = :pid
        LIMIT 1
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params={"pid": int(project_id)})
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return row


def _upsert_meta(SessionLocal, project_id: int, payload: Dict[str, Any]) -> None:
    with SessionLocal() as conn:
        # UPSERT compatível (PG/SQLite) via tentativa
        try:
            conn.execute(
                text("""
                    INSERT INTO project_control_meta
                    (project_id, client_name, project_name, project_number,
                     logo_bk, logo_bk_mime, logo_client, logo_client_mime, updated_at)
                    VALUES
                    (:project_id, :client_name, :project_name, :project_number,
                     :logo_bk, :logo_bk_mime, :logo_client, :logo_client_mime, NOW())
                    ON CONFLICT (project_id) DO UPDATE SET
                        client_name=EXCLUDED.client_name,
                        project_name=EXCLUDED.project_name,
                        project_number=EXCLUDED.project_number,
                        logo_bk=EXCLUDED.logo_bk,
                        logo_bk_mime=EXCLUDED.logo_bk_mime,
                        logo_client=EXCLUDED.logo_client,
                        logo_client_mime=EXCLUDED.logo_client_mime,
                        updated_at=NOW()
                """),
                {"project_id": int(project_id), **payload},
            )
        except Exception:
            # SQLite fallback
            conn.execute(
                text("""
                    INSERT INTO project_control_meta
                    (project_id, client_name, project_name, project_number,
                     logo_bk, logo_bk_mime, logo_client, logo_client_mime, updated_at)
                    VALUES
                    (:project_id, :client_name, :project_name, :project_number,
                     :logo_bk, :logo_bk_mime, :logo_client, :logo_client_mime, CURRENT_TIMESTAMP)
                    ON CONFLICT(project_id) DO UPDATE SET
                        client_name=excluded.client_name,
                        project_name=excluded.project_name,
                        project_number=excluded.project_number,
                        logo_bk=excluded.logo_bk,
                        logo_bk_mime=excluded.logo_bk_mime,
                        logo_client=excluded.logo_client,
                        logo_client_mime=excluded.logo_client_mime,
                        updated_at=CURRENT_TIMESTAMP
                """),
                {"project_id": int(project_id), **payload},
            )
        try:
            conn.commit()
        except Exception:
            pass


def _list_doc_tasks(engine, project_id: int) -> pd.DataFrame:
    sql = text("""
        SELECT id,
               COALESCE(service_id, NULL) AS service_id,
               COALESCE(service_label,'') AS service_label,
               COALESCE(doc_name,'') AS doc_name,
               COALESCE(doc_number,'') AS doc_number,
               COALESCE(revision,'') AS revision,
               COALESCE(status,'') AS status,
               COALESCE(responsible,'') AS responsible,
               start_date,
               end_date,
               COALESCE(notes,'') AS notes
        FROM project_doc_tasks
        WHERE project_id = :pid
        ORDER BY id DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params={"pid": int(project_id)})
    # Normaliza datas para date
    if "start_date" in df.columns:
        df["start_date"] = df["start_date"].apply(_safe_date)
    if "end_date" in df.columns:
        df["end_date"] = df["end_date"].apply(_safe_date)
    return df


def _save_doc_tasks(SessionLocal, project_id: int, original_df: pd.DataFrame, edited_df: pd.DataFrame,
                    service_map: Dict[str, int]) -> Tuple[int, int, int]:
    """Persistência simples: deleta linhas marcadas, depois upsert."""
    inserted = updated = deleted = 0

    # normaliza colunas
    df = edited_df.copy()
    if "Excluir" not in df.columns:
        df["Excluir"] = False

    # deletar
    to_delete = df[(df["Excluir"] == True) & (df["id"].notna())]["id"].tolist()
    if to_delete:
        with SessionLocal() as conn:
            for _id in to_delete:
                conn.execute(
                    text("DELETE FROM project_doc_tasks WHERE id=:id AND project_id=:pid"),
                    {"id": int(_id), "pid": int(project_id)},
                )
                deleted += 1
            try:
                conn.commit()
            except Exception:
                pass
        # remove do df antes de upsert
        df = df[~df["id"].isin(to_delete)].copy()

    # upsert
    with SessionLocal() as conn:
        for _, row in df.iterrows():
            rid = row.get("id")
            service_label = str(row.get("service_label") or "").strip()
            sid = row.get("service_id")
            if (sid is None or str(sid) == "" or str(sid) == "nan") and service_label in service_map:
                sid = service_map[service_label]
            try:
                sid_int = int(sid) if sid is not None and str(sid) not in ("", "nan") else None
            except Exception:
                sid_int = None

            payload = {
                "pid": int(project_id),
                "service_id": sid_int,
                "service_label": service_label,
                "doc_name": str(row.get("doc_name") or "").strip(),
                "doc_number": str(row.get("doc_number") or "").strip(),
                "revision": str(row.get("revision") or "").strip(),
                "status": str(row.get("status") or "").strip(),
                "responsible": str(row.get("responsible") or "").strip(),
                "start_date": _safe_date(row.get("start_date")),
                "end_date": _safe_date(row.get("end_date")),
                "notes": str(row.get("notes") or "").strip(),
            }

            if rid is None or str(rid) in ("", "nan"):
                conn.execute(
                    text("""
                        INSERT INTO project_doc_tasks
                        (project_id, service_id, service_label, doc_name, doc_number, revision,
                         status, responsible, start_date, end_date, notes)
                        VALUES
                        (:pid, :service_id, :service_label, :doc_name, :doc_number, :revision,
                         :status, :responsible, :start_date, :end_date, :notes)
                    """),
                    payload,
                )
                inserted += 1
            else:
                conn.execute(
                    text("""
                        UPDATE project_doc_tasks
                        SET service_id=:service_id,
                            service_label=:service_label,
                            doc_name=:doc_name,
                            doc_number=:doc_number,
                            revision=:revision,
                            status=:status,
                            responsible=:responsible,
                            start_date=:start_date,
                            end_date=:end_date,
                            notes=:notes
                        WHERE id=:id AND project_id=:pid
                    """),
                    {"id": int(rid), **payload},
                )
                updated += 1

        try:
            conn.commit()
        except Exception:
            pass

    return inserted, updated, deleted


# --------------------------
# Relatório HTML
# --------------------------
def _build_report_html(meta: Dict[str, Any], proj: Dict[str, Any], df: pd.DataFrame) -> str:
    client = str(meta.get("client_name") or "").strip() or "Cliente"
    project_name = str(meta.get("project_name") or "").strip() or str(proj.get("nome") or "Projeto")
    project_number = str(meta.get("project_number") or "").strip()

    # logos (data URI)
    logo_left = ""
    if meta.get("logo_bk") is not None:
        try:
            logo_left = _bytes_to_data_uri(meta["logo_bk"], meta.get("logo_bk_mime") or "image/png")
        except Exception:
            logo_left = ""
    logo_right = ""
    if meta.get("logo_client") is not None:
        try:
            logo_right = _bytes_to_data_uri(meta["logo_client"], meta.get("logo_client_mime") or "image/png")
        except Exception:
            logo_right = ""

    # tabela HTML (bordas finas cinza claro)
    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    cols = ["id", "service_label", "doc_name", "doc_number", "revision", "status", "responsible", "start_date", "end_date", "notes"]
    df2 = df.copy()
    for c in cols:
        if c not in df2.columns:
            df2[c] = ""
    df2["start_date"] = df2["start_date"].apply(lambda x: _safe_date(x))
    df2["end_date"] = df2["end_date"].apply(lambda x: _safe_date(x))

    rows_html = []
    for _, r in df2.iterrows():
        rows_html.append(
            "<tr>"
            f"<td>{esc(str(r['id']))}</td>"
            f"<td>{esc(str(r['service_label']))}</td>"
            f"<td>{esc(str(r['doc_name']))}</td>"
            f"<td>{esc(str(r['doc_number']))}</td>"
            f"<td>{esc(str(r['revision']))}</td>"
            f"<td>{esc(str(r['status']))}</td>"
            f"<td>{esc(str(r['responsible']))}</td>"
            f"<td>{esc(str(_safe_date(r['start_date']) or ''))}</td>"
            f"<td>{esc(str(_safe_date(r['end_date']) or ''))}</td>"
            f"<td>{esc(str(r['notes']))}</td>"
            "</tr>"
        )
    table_html = f"""
    <table class="bk-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>Serviço</th>
          <th>Nome do documento</th>
          <th>Nº do documento</th>
          <th>Revisão</th>
          <th>Status</th>
          <th>Responsável</th>
          <th>Início</th>
          <th>Conclusão</th>
          <th>Observação</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html) if rows_html else '<tr><td colspan="10" style="text-align:center;color:#64748B;">Sem registros</td></tr>'}
      </tbody>
    </table>
    """

    # gráficos simples (dias total + revisões)
    dfm = df2.copy()
    dfm["dias_total"] = dfm.apply(
        lambda r: (r["end_date"] - r["start_date"]).days if _safe_date(r["start_date"]) and _safe_date(r["end_date"]) else None,
        axis=1
    )
    # revision letter -> numero
    def rev_to_n(v: str) -> int:
        s = (v or "").strip().upper()
        # R0A, R0B ...
        m = None
        import re as _re
        m = _re.search(r'R0([A-Z])', s)
        if not m:
            return 0
        return (ord(m.group(1)) - ord('A')) + 1

    dfm["rev_n"] = dfm["revision"].apply(rev_to_n)
    chart1 = ""
    chart2 = ""
    try:
        dft = dfm.dropna(subset=["dias_total"])
        if not dft.empty:
            fig1 = px.bar(dft, x="doc_name", y="dias_total", title="Dias totais por documento")
            chart1 = fig1.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception:
        chart1 = ""
    try:
        dfr = dfm.copy()
        if not dfr.empty:
            fig2 = px.bar(dfr, x="doc_name", y="rev_n", title="Nº de revisão (estimado) por documento")
            chart2 = fig2.to_html(full_html=False, include_plotlyjs=False)
    except Exception:
        chart2 = ""

    # layout do relatório
    html = f"""
    <!doctype html>
    <html lang="pt-br">
    <head>
      <meta charset="utf-8"/>
      <title>Relatório - Controle de Projetos</title>
      <style>
        body {{
          font-family: Arial, Helvetica, sans-serif;
          color: #0F172A;
          margin: 24px;
        }}
        .hdr {{
          display: grid;
          grid-template-columns: 180px 1fr 180px;
          align-items: center;
          gap: 12px;
          margin-bottom: 14px;
        }}
        .hdr .logo {{
          height: 70px;
          display:flex;
          align-items:center;
          justify-content:center;
        }}
        .hdr .logo img {{
          max-height: 70px;
          max-width: 170px;
          object-fit: contain;
        }}
        .hdr .center {{
          text-align: center;
        }}
        .hdr .center .title {{
          font-size: 22px;
          font-weight: 800;
        }}
        .hdr .center .sub {{
          margin-top: 6px;
          color: #334155;
          font-size: 13px;
        }}
        .hdr .center .sub2 {{
          margin-top: 4px;
          color: #64748B;
          font-size: 12px;
        }}
        .bk-table {{
          width: 100%;
          border-collapse: collapse;
          margin-top: 10px;
        }}
        .bk-table th, .bk-table td {{
          border: 1px solid #D1D5DB;
          padding: 6px 8px;
          font-size: 12px;
        }}
        .bk-table th {{
          background: #F1F5F9;
          text-align: left;
          font-weight: 700;
        }}
        .section {{
          margin-top: 14px;
        }}
      </style>
    </head>
    <body>
      <div class="hdr">
        <div class="logo">{f'<img src="{logo_left}" />' if logo_left else ''}</div>
        <div class="center">
          <div class="title">BK Engenharia e Tecnologia</div>
          <div class="sub">Cliente: <b>{client}</b></div>
          <div class="sub2">Projeto: <b>{project_name}</b>{f' • Nº {project_number}' if project_number else ''}</div>
        </div>
        <div class="logo">{f'<img src="{logo_right}" />' if logo_right else ''}</div>
      </div>

      <div class="section">
        {table_html}
      </div>

      <div class="section">
        {chart1 if chart1 else ''}
      </div>
      <div class="section">
        {chart2 if chart2 else ''}
      </div>
    </body>
    </html>
    """
    return html


# --------------------------
# UI
# --------------------------
def main():
    # DB / Auth
    engine, SessionLocal = get_finance_db()
    ensure_erp_tables(engine, SessionLocal)
    _ensure_control_tables(engine, SessionLocal)

    login_and_guard(SessionLocal)

    st.title("🖥️ Controle de Projetos")
    st.caption("Documentos / tarefas do projeto (BK x Cliente) com tabela estilo Excel e relatório.")

    # Lista de projetos
    projects_df = _list_projects(engine)
    if projects_df.empty:
        st.info("Nenhum projeto cadastrado ainda. Cadastre um projeto no módulo Gestão de Projetos.")
        return

    # Seleção do projeto
    proj_opts = {f"#{int(r.id)} - {r.nome}": int(r.id) for r in projects_df.itertuples()}
    proj_label = st.selectbox("Projetos", list(proj_opts.keys()))
    pid = proj_opts.get(proj_label)
    if not pid:
        st.stop()

    proj = _load_project(engine, pid) or {}
    meta = _load_meta(engine, pid) or {}

    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.subheader("Projeto")

        status_opts = ["em_planejamento", "em_andamento", "em_aprovacao", "encerrado"]
        cur_status = (proj.get("status") or "").strip()
        if cur_status and cur_status not in status_opts:
            status_opts = [cur_status] + [s for s in status_opts if s != cur_status]
        status_sel = st.selectbox("Status", status_opts, index=max(0, status_opts.index(cur_status)) if cur_status in status_opts else 0)

        # Campos de prazo (se existirem na tabela; aqui apenas informativo)
        # Mantemos simples para não conflitar com EAP
        st.divider()
        st.subheader("Relatório (HTML)")
        st.caption("Os dados do relatório e as logos ficam salvos por projeto.")

        with st.form("meta_form"):
            client_name = st.text_input("Cliente", value=str(meta.get("client_name") or ""))
            project_name = st.text_input("Nome do projeto", value=str(meta.get("project_name") or proj.get("nome") or ""))
            project_number = st.text_input("Nº do projeto", value=str(meta.get("project_number") or ""))

            colA, colB = st.columns(2)
            with colA:
                logo_bk_file = st.file_uploader("Logo BK (PNG/JPG)", type=["png", "jpg", "jpeg", "webp"], key="logo_bk_upl")
            with colB:
                logo_client_file = st.file_uploader("Logo Cliente (PNG/JPG)", type=["png", "jpg", "jpeg", "webp"], key="logo_client_upl")

            saved = st.form_submit_button("Salvar dados do relatório")

        if saved:
            payload = {
                "client_name": client_name.strip(),
                "project_name": project_name.strip(),
                "project_number": project_number.strip(),
                "logo_bk": meta.get("logo_bk"),
                "logo_bk_mime": meta.get("logo_bk_mime"),
                "logo_client": meta.get("logo_client"),
                "logo_client_mime": meta.get("logo_client_mime"),
            }
            if logo_bk_file is not None:
                payload["logo_bk"] = logo_bk_file.getvalue()
                payload["logo_bk_mime"] = _guess_mime(logo_bk_file.name)
            if logo_client_file is not None:
                payload["logo_client"] = logo_client_file.getvalue()
                payload["logo_client_mime"] = _guess_mime(logo_client_file.name)

            _upsert_meta(SessionLocal, pid, payload)
            st.success("Dados do relatório salvos!")

            # atualizar meta em memória
            meta = _load_meta(engine, pid) or {}

        # Atualiza status do projeto (apenas tabela projects)
        if st.button("Salvar status do projeto"):
            with SessionLocal() as conn:
                conn.execute(text("UPDATE projects SET status=:s WHERE id=:id"), {"s": status_sel, "id": int(pid)})
                try:
                    conn.commit()
                except Exception:
                    pass
            st.success("Status atualizado.")

        # botão relatório
        df_now = _list_doc_tasks(engine, pid)
        html = _build_report_html(meta, proj, df_now)
        st.download_button(
            "📄 Baixar relatório (HTML)",
            data=html.encode("utf-8"),
            file_name=f"relatorio_controle_projetos_{pid}.html",
            mime="text/html",
            width="stretch"
        )

    with right:
        st.subheader("Tarefas / Documentos (tabela estilo Excel)")
        st.caption("Preencha diretamente na tabela. Use a coluna 'Excluir' para remover linhas.")

        # Carrega serviços do cadastro
        with SessionLocal() as conn:
            services = list_product_services(conn)
        services = services[services["active"] == True] if "active" in services.columns else services
        services = services.sort_values("name") if "name" in services.columns else services
        service_labels = []
        service_map = {}
        for _, r in services.iterrows():
            label = str(r.get("name") or "").strip()
            if not label:
                continue
            service_labels.append(label)
            try:
                service_map[label] = int(r.get("id"))
            except Exception:
                pass
        if not service_labels:
            service_labels = ["(cadastre serviços em Cadastros > Serviços)"]

        df = _list_doc_tasks(engine, pid)
        if df.empty:
            df = pd.DataFrame(columns=[
                "id", "service_label", "doc_name", "doc_number", "revision", "status",
                "responsible", "start_date", "end_date", "notes"
            ])

        # coluna excluir
        if "Excluir" not in df.columns:
            df["Excluir"] = False

        status_options = ["Em andamento - BK", "Em análise - Cliente", "Em revisão - BK", "Aprovado - Cliente"]
        resp_options = ["BK", "CLIENTE", "N/A"]

        edited = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,  # mantém compat com streamlit atual (warning OK)
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", disabled=True),
                "service_label": st.column_config.SelectboxColumn("Serviço", options=service_labels, required=False),
                "doc_name": st.column_config.TextColumn("Nome do documento", required=False),
                "doc_number": st.column_config.TextColumn("Nº do documento", required=False),
                "revision": st.column_config.TextColumn("Revisão", required=False),
                "status": st.column_config.SelectboxColumn("Status", options=status_options, required=False),
                "responsible": st.column_config.SelectboxColumn("Responsável", options=resp_options, required=False),
                "start_date": st.column_config.DateColumn("Data de início", required=False),
                "end_date": st.column_config.DateColumn("Data de conclusão", required=False),
                "notes": st.column_config.TextColumn("Observação", required=False),
                "Excluir": st.column_config.CheckboxColumn("Excluir"),
                "service_id": st.column_config.NumberColumn("service_id", disabled=True, help="interno", width="small"),
            },
            key=f"doc_editor_{pid}",
        )

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("💾 Salvar tabela", width="stretch"):
                ins, upd, dele = _save_doc_tasks(SessionLocal, pid, df, edited, service_map)
                st.success(f"Salvo! Inseridos: {ins} • Atualizados: {upd} • Excluídos: {dele}")
                _rerun()
        with col2:
            if st.button("🔄 Recarregar", width="stretch"):
                _rerun()

        # Prévia rápida de métricas
        with col3:
            st.caption("Resumo rápido")
            dfm = edited.copy()
            dfm["start_date"] = dfm["start_date"].apply(_safe_date)
            dfm["end_date"] = dfm["end_date"].apply(_safe_date)
            total_docs = len(dfm[dfm.get("doc_name").notna()]) if "doc_name" in dfm.columns else len(dfm)
            aprovados = len(dfm[dfm["status"] == "Aprovado - Cliente"]) if "status" in dfm.columns else 0
            st.write(f"Documentos/tarefas: **{total_docs}** • Aprovados: **{aprovados}**")

        # Gráficos no app (não mexe no relatório)
        st.divider()
        st.subheader("Gráficos (visão rápida)")
        try:
            dfp = edited.copy()
            dfp["start_date"] = dfp["start_date"].apply(_safe_date)
            dfp["end_date"] = dfp["end_date"].apply(_safe_date)
            dfp["dias_total"] = dfp.apply(
                lambda r: (r["end_date"] - r["start_date"]).days
                if r.get("start_date") and r.get("end_date") else None,
                axis=1
            )
            dft = dfp.dropna(subset=["dias_total"])
            if not dft.empty:
                st.plotly_chart(px.bar(dft, x="doc_name", y="dias_total", title="Dias totais por documento"), width="stretch")
            else:
                st.info("Preencha datas de início e conclusão para ver o gráfico de dias totais.")
        except Exception as e:
            st.warning("Não foi possível gerar os gráficos agora (dados incompletos).")


if __name__ == "__main__":
    main()
