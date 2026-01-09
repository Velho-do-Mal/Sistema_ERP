# pages/6_🖥️_Controle_de_Projetos.py
# -*- coding: utf-8 -*-
"""BK_ERP - 🖥️ Controle de Projetos

Requisitos atendidos (Controle de Projetos):
- Tabela estilo Excel (st.data_editor) para controle de tarefas/documentos do projeto.
- Status e medição de tempo acumulado por etapa e por responsável:
  - Em andamento - BK  -> conta como ELABORAÇÃO (BK)
  - Em análise - Cliente -> conta como ANÁLISE (CLIENTE) a partir da Data de conclusão (entrega ao cliente)
  - Em revisão - BK -> conta como REVISÃO (BK) a partir da Data de início (revisão)
  - Aprovado - Cliente -> para de contar
- Revisões automáticas:
  - Ao criar uma linha (início do trabalho) -> revisão inicia em R0A
  - Sempre que status mudar para "Em revisão - BK" (vindo do Cliente) -> incrementa automaticamente (R0B, R0C, ...)
- Relatórios com gráficos + export (CSV e HTML com logos).

Obs.: Este módulo mantém o painel de projeto e o relatório geral existente.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from typing import Optional, List, Dict, Tuple

import base64
import pandas as pd
import streamlit as st
from sqlalchemy import text, inspect

# shared
from bk_erp_shared.erp_db import ensure_erp_tables, get_finance_db
from bk_erp_shared.auth import login_and_guard
from bk_erp_shared.theme import apply_theme

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="BK_ERP - Controle de Projetos", layout="wide")

# -------------------------
# Utilidades
# -------------------------
STATUS_OPTIONS = [
    "Em andamento - BK",
    "Em análise - Cliente",
    "Em revisão - BK",
    "Aprovado - Cliente",
]

def _to_date(v, fallback: Optional[date] = None) -> date:
    """Converte entrada para date (defensivo para Streamlit)."""
    if fallback is None:
        fallback = date.today()
    if v is None:
        return fallback
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    # pandas Timestamp / NaT
    try:
        import pandas as _pd
        if isinstance(v, _pd.Timestamp):
            if _pd.isna(v):
                return fallback
            return v.to_pydatetime().date()
    except Exception:
        pass
    s = str(v).strip()
    if not s or s.lower() in ("nat", "none"):
        return fallback
    # ISO yyyy-mm-dd or yyyy/mm/dd
    try:
        s2 = s.replace("/", "-")
        return datetime.fromisoformat(s2).date()
    except Exception:
        return fallback


def _pick_column(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
        for col in cols:
            if col.lower() == c.lower():
                return col
    return None


def _sql_table_exists(insp, table: str) -> bool:
    try:
        return insp.has_table(table)
    except Exception:
        try:
            return table in insp.get_table_names()
        except Exception:
            return False


def _ensure_doc_tables(SessionLocal) -> None:
    """Cria tabelas (Postgres/SQLite) com commit (evita ProgrammingError no Cloud)."""
    # id
    ddl_pg = """
    CREATE TABLE IF NOT EXISTS project_doc_tasks (
        id SERIAL PRIMARY KEY,
        project_id INTEGER NOT NULL,
        service_id INTEGER NULL,
        service_name TEXT,
        complemento TEXT,
        project_number TEXT,
        start_date DATE,
        delivery_date DATE,
        status TEXT DEFAULT 'Em andamento - BK',
        revision_code TEXT DEFAULT 'R0A',
        observation TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    ddl_sqlite = """
    CREATE TABLE IF NOT EXISTS project_doc_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        service_id INTEGER,
        service_name TEXT,
        complemento TEXT,
        project_number TEXT,
        start_date DATE,
        delivery_date DATE,
        status TEXT,
        revision_code TEXT,
        observation TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    ddl_events_pg = """
    CREATE TABLE IF NOT EXISTS doc_status_events (
        id SERIAL PRIMARY KEY,
        doc_task_id INTEGER NOT NULL,
        project_id INTEGER NOT NULL,
        event_date DATE NOT NULL,
        status TEXT NOT NULL,
        responsible TEXT NOT NULL,
        revision_code TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    ddl_events_sqlite = """
    CREATE TABLE IF NOT EXISTS doc_status_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_task_id INTEGER NOT NULL,
        project_id INTEGER NOT NULL,
        event_date DATE NOT NULL,
        status TEXT NOT NULL,
        responsible TEXT NOT NULL,
        revision_code TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    with SessionLocal() as session:
        try:
            session.execute(text(ddl_pg))
        except Exception:
            session.execute(text(ddl_sqlite))
        try:
            session.execute(text(ddl_events_pg))
        except Exception:
            session.execute(text(ddl_events_sqlite))
        session.commit()


def _status_to_responsible(status: str) -> str:
    if status == "Em análise - Cliente" or status == "Aprovado - Cliente":
        return "CLIENTE"
    return "BK"


def _status_kind(status: str) -> str:
    # para cálculo por etapa
    s = (status or "").lower()
    if "análise" in s or "analise" in s:
        return "analise"
    if "revis" in s:
        return "revisao"
    if "aprov" in s:
        return "aprovado"
    return "elaboracao"


def _next_revision(code: Optional[str]) -> str:
    """R0A -> R0B -> ... -> R0O (cap)."""
    if not code:
        return "R0A"
    code = str(code).strip().upper()
    if not code.startswith("R0") or len(code) < 3:
        return "R0A"
    letter = code[2]
    if not ("A" <= letter <= "Z"):
        return "R0A"
    nxt = chr(min(ord(letter) + 1, ord("O")))  # cap em O
    return f"R0{nxt}"


def _revision_count_from_code(code: Optional[str]) -> int:
    """R0A=1, R0B=2 ... R0O=15"""
    if not code:
        return 0
    code = str(code).strip().upper()
    if not code.startswith("R0") or len(code) < 3:
        return 0
    letter = code[2]
    if not ("A" <= letter <= "Z"):
        return 0
    return (ord(letter) - ord("A")) + 1


def _list_services(engine) -> pd.DataFrame:
    """Lista serviços/produtos cadastrados (tabela product_services)."""
    insp = inspect(engine)
    if not _sql_table_exists(insp, "product_services"):
        return pd.DataFrame(columns=["id", "name", "type", "active"])
    sql = """
        SELECT id, COALESCE(name,'') AS name, COALESCE(type,'') AS type, COALESCE(active,TRUE) AS active
        FROM product_services
        WHERE COALESCE(active,TRUE) = TRUE
        ORDER BY COALESCE(name,'') ASC
    """
    try:
        return pd.read_sql(sql, engine)
    except Exception:
        return pd.DataFrame(columns=["id", "name", "type", "active"])


def _list_doc_tasks(engine, SessionLocal, project_id: int) -> pd.DataFrame:
    _ensure_doc_tables(SessionLocal)
    sql = """
        SELECT id, service_id, COALESCE(service_name,'') AS service_name,
               COALESCE(complemento,'') AS complemento,
               COALESCE(project_number,'') AS project_number,
               start_date, delivery_date,
               COALESCE(status,'Em andamento - BK') AS status,
               COALESCE(revision_code,'R0A') AS revision_code,
               COALESCE(observation,'') AS observation
        FROM project_doc_tasks
        WHERE project_id=:pid
        ORDER BY id ASC
    """
    df = pd.read_sql(sql, engine, params={"pid": int(project_id)})
    # normalizar datas para date
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce").dt.date
    if "delivery_date" in df.columns:
        df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce").dt.date
    return df


def _get_doc_task_by_id(SessionLocal, doc_id: int) -> Optional[Dict[str, object]]:
    with SessionLocal() as session:
        r = session.execute(
            text("""
                SELECT id, status, revision_code, start_date, delivery_date
                FROM project_doc_tasks WHERE id=:id
            """),
            {"id": int(doc_id)},
        ).mappings().first()
        return dict(r) if r else None


def _insert_event(SessionLocal, project_id: int, doc_task_id: int, status: str,
                  event_dt: date, revision_code: Optional[str]) -> None:
    responsible = _status_to_responsible(status)
    with SessionLocal() as session:
        session.execute(
            text("""
                INSERT INTO doc_status_events (doc_task_id, project_id, event_date, status, responsible, revision_code)
                VALUES (:doc, :pid, :dt, :st, :resp, :rev)
            """),
            {"doc": int(doc_task_id), "pid": int(project_id), "dt": event_dt.isoformat(), "st": status, "resp": responsible, "rev": revision_code},
        )
        session.commit()


def _upsert_doc_tasks(SessionLocal, project_id: int, rows: pd.DataFrame, services_map: Dict[int, str],
                      project_number: str) -> Tuple[int, int]:
    """Upsert + eventos. Retorna (inseridos, atualizados)."""
    inserted = 0
    updated = 0

    for _, r in rows.iterrows():
        rid = r.get("id", None)
        delete_flag = bool(r.get("Excluir", False))

        service_id = r.get("service_id", None)
        try:
            service_id_int = int(service_id) if service_id not in (None, "", pd.NA) and not pd.isna(service_id) else None
        except Exception:
            service_id_int = None
        service_name = ""
        if service_id_int is not None:
            service_name = services_map.get(service_id_int, "")
        if not service_name:
            service_name = str(r.get("service_name", "") or "").strip()

        complemento = str(r.get("complemento", "") or "").strip()
        start_dt = _to_date(r.get("start_date", None), fallback=date.today())
        delivery_dt_val = r.get("delivery_date", None)
        delivery_dt = _to_date(delivery_dt_val, fallback=start_dt)

        status = str(r.get("status", "Em andamento - BK") or "Em andamento - BK").strip()
        if status not in STATUS_OPTIONS:
            status = "Em andamento - BK"

        observation = str(r.get("observation", "") or "").strip()

        # --------- DELETE ---------
        if delete_flag and rid not in (None, "", pd.NA) and not pd.isna(rid):
            with SessionLocal() as session:
                session.execute(text("DELETE FROM project_doc_tasks WHERE id=:id"), {"id": int(rid)})
                session.execute(text("DELETE FROM doc_status_events WHERE doc_task_id=:id"), {"id": int(rid)})
                session.commit()
            continue

        # --------- INSERT ---------
        if rid in (None, "", pd.NA) or pd.isna(rid):
            # regra: ao iniciar, revisão é R0A
            revision_code = "R0A"
            with SessionLocal() as session:
                res = session.execute(
                    text("""
                        INSERT INTO project_doc_tasks
                        (project_id, service_id, service_name, complemento, project_number,
                         start_date, delivery_date, status, revision_code, observation, updated_at)
                        VALUES (:pid, :sid, :sname, :comp, :pnum,
                                :sd, :dd, :st, :rev, :obs, CURRENT_TIMESTAMP)
                    """),
                    {
                        "pid": int(project_id),
                        "sid": service_id_int,
                        "sname": service_name,
                        "comp": complemento,
                        "pnum": project_number,
                        "sd": start_dt.isoformat(),
                        "dd": delivery_dt.isoformat(),
                        "st": status,
                        "rev": revision_code,
                        "obs": observation,
                    },
                )
                # obter id inserido
                new_id = None
                try:
                    new_id = res.scalar()
                except Exception:
                    pass
                if new_id is None:
                    # fallback sqlite
                    new_id = session.execute(text("SELECT last_insert_rowid()")).scalar()
                session.commit()

            inserted += 1
            # evento inicial: conta como elaboração BK (em andamento) ou revisão BK se começar já em revisão
            event_date = start_dt if status in ("Em andamento - BK", "Em revisão - BK") else delivery_dt
            _insert_event(SessionLocal, project_id, int(new_id), status, event_date, revision_code)
            continue

        # --------- UPDATE ---------
        rid_int = int(rid)
        prev = _get_doc_task_by_id(SessionLocal, rid_int) or {}
        prev_status = str(prev.get("status", "") or "")
        prev_rev = str(prev.get("revision_code", "") or "") or "R0A"

        revision_code = str(r.get("revision_code", "") or prev_rev or "R0A").strip().upper()
        if not revision_code:
            revision_code = "R0A"

        status_changed = (status != prev_status)

        # regra de revisão automática:
        # sempre que entrar em "Em revisão - BK" vindo do cliente (ex.: de "Em análise - Cliente"), incrementa.
        if status_changed and status == "Em revisão - BK" and prev_status == "Em análise - Cliente":
            # Cliente devolveu -> nova revisão (R0A -> R0B -> ...)
            revision_code = _next_revision(prev_rev)

        with SessionLocal() as session:
            session.execute(
                text("""
                    UPDATE project_doc_tasks SET
                        service_id=:sid,
                        service_name=:sname,
                        complemento=:comp,
                        project_number=:pnum,
                        start_date=:sd,
                        delivery_date=:dd,
                        status=:st,
                        revision_code=:rev,
                        observation=:obs,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=:id
                """),
                {
                    "sid": service_id_int,
                    "sname": service_name,
                    "comp": complemento,
                    "pnum": project_number,
                    "sd": start_dt.isoformat(),
                    "dd": delivery_dt.isoformat(),
                    "st": status,
                    "rev": revision_code,
                    "obs": observation,
                    "id": rid_int,
                },
            )
            session.commit()

        updated += 1

        # evento se mudou status (para calcular tempos acumulados corretamente)
        if status_changed:
            if status in ("Em andamento - BK", "Em revisão - BK"):
                event_date = start_dt
            else:
                event_date = delivery_dt
            _insert_event(SessionLocal, project_id, rid_int, status, event_date, revision_code)

    return inserted, updated


def _compute_doc_metrics(engine, project_id: int) -> pd.DataFrame:
    """Retorna DF por doc_task_id com tempos BK/Cliente e revisões."""
    sql = """
        SELECT doc_task_id, event_date, status, responsible, COALESCE(revision_code,'') AS revision_code
        FROM doc_status_events
        WHERE project_id=:pid
        ORDER BY doc_task_id ASC, event_date ASC, id ASC
    """
    try:
        ev = pd.read_sql(sql, engine, params={"pid": int(project_id)})
    except Exception:
        return pd.DataFrame()

    if ev.empty:
        return pd.DataFrame()

    ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce").dt.date
    today = date.today()

    out_rows = []
    for doc_id, grp in ev.groupby("doc_task_id", sort=True):
        g = grp.sort_values(["event_date"]).reset_index(drop=True)
        dias_elab_bk = 0
        dias_rev_bk = 0
        dias_anal_cli = 0
        # contar revisões pelo maior revision_code encontrado (R0A..)
        max_rev_code = ""
        for rv in g["revision_code"].tolist():
            if rv and rv > max_rev_code:
                max_rev_code = rv

        for i in range(len(g)):
            stt = str(g.loc[i, "status"])
            kind = _status_kind(stt)
            resp = str(g.loc[i, "responsible"]).upper()
            start_dt = g.loc[i, "event_date"]
            if pd.isna(start_dt) or start_dt is None:
                continue
            if i < len(g) - 1:
                end_dt = g.loc[i + 1, "event_date"]
                if pd.isna(end_dt) or end_dt is None:
                    end_dt = today
            else:
                # último: se aprovado, não conta; senão conta até hoje
                if kind == "aprovado":
                    end_dt = start_dt
                else:
                    end_dt = today

            delta = max(0, (end_dt - start_dt).days)
            if resp == "BK":
                if kind == "revisao":
                    dias_rev_bk += delta
                else:
                    # em andamento conta como elaboração
                    dias_elab_bk += delta
            else:
                if kind == "analise":
                    dias_anal_cli += delta

        out_rows.append(
            {
                "doc_task_id": int(doc_id),
                "dias_elaboracao_BK": int(dias_elab_bk),
                "dias_revisao_BK": int(dias_rev_bk),
                "dias_analise_CLIENTE": int(dias_anal_cli),
                "revision_code": max_rev_code,
                "revisoes_qtd": _revision_count_from_code(max_rev_code),
                "dias_total": int(dias_elab_bk + dias_rev_bk + dias_anal_cli),
            }
        )
    return pd.DataFrame(out_rows)


def _chart_png_b64(df: pd.DataFrame, x: str, y: str, title: str) -> str:
    """Gera gráfico simples (matplotlib) e retorna base64 PNG."""
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # sem cores fixas: matplotlib usa padrão
        ax.bar(df[x].astype(str).tolist(), df[y].astype(float).tolist())
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        bio = BytesIO()
        fig.savefig(bio, format="png", dpi=160)
        plt.close(fig)
        return base64.b64encode(bio.getvalue()).decode("utf-8")
    except Exception:
        return ""


def _html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _build_doc_report_html(project_name: str, client_name: str, logo_bk_b64: str, logo_cli_b64: str,
                           doc_table: pd.DataFrame, metrics: pd.DataFrame) -> str:
    # merge
    rep = doc_table.merge(metrics, left_on="id", right_on="doc_task_id", how="left")
    rep = rep.fillna({"dias_elaboracao_BK":0,"dias_revisao_BK":0,"dias_analise_CLIENTE":0,"revisoes_qtd":0,"dias_total":0,"revision_code":"R0A"})
    # charts
    top = rep.sort_values("dias_total", ascending=False).head(10)
    top_rev = rep.sort_values("dias_revisao_BK", ascending=False).head(10)
    top_ana = rep.sort_values("dias_analise_CLIENTE", ascending=False).head(10)
    top_revcount = rep.sort_values("revisoes_qtd", ascending=False).head(10)

    charts = []
    for dfx, y, ttl in [
        (top, "dias_total", "Top 10 - Tempo total (BK+Cliente)"),
        (top_ana, "dias_analise_CLIENTE", "Top 10 - Análise (Cliente)"),
        (top_rev, "dias_revisao_BK", "Top 10 - Revisão (BK)"),
        (top_revcount, "revisoes_qtd", "Top 10 - Qtde de revisões"),
    ]:
        if dfx.empty:
            charts.append("")
            continue
        dfx = dfx.copy()
        dfx["label"] = dfx.apply(lambda r: f"#{int(r['id'])} {str(r.get('service_name',''))}".strip(), axis=1)
        b64 = _chart_png_b64(dfx[["label", y]], "label", y, ttl)
        charts.append(f'<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;" />' if b64 else "")

    # table html
    cols = ["id","service_name","complemento","project_number","start_date","delivery_date","status","revision_code","revisoes_qtd",
            "dias_elaboracao_BK","dias_revisao_BK","dias_analise_CLIENTE","dias_total","observation"]
    for c in cols:
        if c not in rep.columns:
            rep[c] = ""
    th = "".join([f"<th>{_html_escape(c)}</th>" for c in cols])
    rows_html = []
    for _, r in rep[cols].iterrows():
        tds = "".join([f"<td>{_html_escape(str(r.get(c,'')))}</td>" for c in cols])
        rows_html.append(f"<tr>{tds}</tr>")
    table_html = f"""
    <table>
      <thead><tr>{th}</tr></thead>
      <tbody>{''.join(rows_html) if rows_html else '<tr><td colspan="14">Sem dados</td></tr>'}</tbody>
    </table>
    """

    logo_bk = f'<img src="data:image/png;base64,{logo_bk_b64}" style="height:52px" />' if logo_bk_b64 else ""
    logo_cli = f'<img src="data:image/png;base64,{logo_cli_b64}" style="height:52px" />' if logo_cli_b64 else ""

    total_revs = int(rep["revisoes_qtd"].sum()) if not rep.empty else 0

    return f"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8"/>
<title>Controle de Projetos - { _html_escape(project_name) }</title>
<style>
body{{font-family: Arial, sans-serif; margin:24px;}}
.header{{display:flex;align-items:center;justify-content:space-between;gap:16px;}}
h1{{margin:0;}}
.badge{{display:inline-block;padding:4px 10px;border-radius:12px;background:#eef;border:1px solid #ccd;}}
.grid{{display:grid;grid-template-columns:1fr;gap:14px;margin-top:14px;}}
.card{{border:1px solid #ddd;border-radius:12px;padding:12px;}}
table{{border-collapse:collapse;width:100%;font-size:12px;}}
th,td{{border:1px solid #ddd;padding:6px;vertical-align:top;}}
th{{background:#f7f7f7;}}
</style>
</head>
<body>
<div class="header">
  <div>{logo_bk}</div>
  <div style="text-align:center">
    <h1>Controle de Projetos</h1>
    <div class="badge"><b>Projeto:</b> {_html_escape(project_name)} | <b>Cliente:</b> {_html_escape(client_name)}</div>
    <div class="badge"><b>Total de revisões (projeto):</b> {total_revs}</div>
  </div>
  <div>{logo_cli}</div>
</div>

<div class="grid">
  <div class="card">
    <h2>Gráficos</h2>
    {charts[0]}<br/>
    {charts[1]}<br/>
    {charts[2]}<br/>
    {charts[3]}<br/>
  </div>

  <div class="card">
    <h2>Tabela completa</h2>
    {table_html}
  </div>
</div>
</body>
</html>
"""


# -------------------------
# Página
# -------------------------
def main() -> None:
    ensure_erp_tables()
    engine, SessionLocal = get_finance_db()
    login_and_guard(SessionLocal)
    apply_theme()

    st.markdown('<div class="bk-card"><div class="bk-title">🖥️ Controle de Projetos</div><div class="bk-subtitle">Acompanhe status, prazos e tempos (BK x Cliente) com revisões automáticas.</div></div>', unsafe_allow_html=True)

    projects = _list_projects_defensive(engine)
    if projects.empty:
        st.info("Nenhum projeto cadastrado.")
        return

    # select project
    proj_opts = {f"#{int(r.id)} - {r.nome}": int(r.id) for r in projects.itertuples()}
    sel = st.selectbox("Projetos", list(proj_opts.keys()))
    pid = proj_opts[sel]
    row = projects[projects["id"] == int(pid)].iloc[0].to_dict()

    # split layout
    left, right = st.columns([0.42, 0.58], gap="large")

    with left:
        st.subheader("Projeto")
        status = st.text_input("Status", value=str(row.get("status", "") or ""))
        progress = st.slider("Progresso (%)", 0, 100, int(row.get("progress_pct") or 0))

        planned_default = _to_date(row.get("planned_end_date", None), fallback=date.today())
        actual_default = _to_date(row.get("actual_end_date", None), fallback=planned_default)

        planned = st.date_input("Entrega prevista", value=planned_default)
        actual = st.date_input("Entrega real (se concluído)", value=actual_default)

        resp_list = ["N/A", "CLIENTE", "BK"]
        resp_val = str(row.get("delay_responsibility") or "N/A").upper()
        if resp_val not in resp_list:
            resp_val = "N/A"
        resp = st.selectbox("Responsável pelo atraso", resp_list, index=resp_list.index(resp_val))

        if st.button("Salvar projeto", type="primary", use_container_width=True):
            _update_project(SessionLocal, pid, status, progress, planned, actual, resp)
            st.success("Projeto atualizado.")
            st.rerun()

        st.divider()
        st.subheader("Relatório geral (HTML)")
        build_report = _get_renderer(engine, SessionLocal)
        html = build_report(project_id=pid)
        st.download_button("📄 Baixar relatório (HTML)", data=html.encode("utf-8"),
                           file_name=f"controle_projetos_{pid}.html", mime="text/html", use_container_width=True)

    with right:
        st.subheader("Tarefas / Documentos (tabela estilo Excel)")
        # header/logo
        c1, c2 = st.columns(2)
        with c1:
            logo_bk = st.file_uploader("Logo BK (PNG/JPG)", type=["png","jpg","jpeg"], key="logo_bk")
        with c2:
            logo_cli = st.file_uploader("Logo Cliente (PNG/JPG)", type=["png","jpg","jpeg"], key="logo_cli")

        logo_bk_b64 = base64.b64encode(logo_bk.getvalue()).decode("utf-8") if logo_bk else ""
        logo_cli_b64 = base64.b64encode(logo_cli.getvalue()).decode("utf-8") if logo_cli else ""

        # services
        services_df = _list_services(engine)
        services_map = {int(r.id): str(r.name) for r in services_df.itertuples()} if not services_df.empty else {}
        service_labels = {f"{int(r.id)} - {r.name}": int(r.id) for r in services_df.itertuples()} if not services_df.empty else {}
        # data
        doc_df = _list_doc_tasks(engine, SessionLocal, pid)
        if doc_df.empty:
            doc_df = pd.DataFrame(columns=[
                "id","service_id","service_name","complemento","project_number","start_date","delivery_date","status","revision_code","observation"
            ])

        # add helper column
        doc_edit = doc_df.copy()
        doc_edit["Excluir"] = False

        # editor column config
        col_cfg = {
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "service_id": st.column_config.SelectboxColumn(
                "Tarefa (Serviços)",
                options=list(services_map.keys()) if services_map else [],
                help="Selecione pelo ID do serviço/produto cadastrado (pesquise digitando).",
                required=False,
            ),
            "service_name": st.column_config.TextColumn("Tarefa (descrição)", help="Preenchido automaticamente pelo serviço; pode ajustar manualmente.", required=False),
            "complemento": st.column_config.TextColumn("Complemento", required=False),
            "project_number": st.column_config.TextColumn("Nº do projeto", disabled=True),
            "start_date": st.column_config.DateColumn("Data de início", required=True),
            "delivery_date": st.column_config.DateColumn("Data de conclusão (entrega p/ análise)", required=True),
            "status": st.column_config.SelectboxColumn("Status", options=STATUS_OPTIONS, required=True),
            "revision_code": st.column_config.TextColumn("Nº Revisão", disabled=True, help="R0A no início. Incrementa automaticamente quando volta para revisão."),
            "observation": st.column_config.TextColumn("Observação", required=False),
            "Excluir": st.column_config.CheckboxColumn("Excluir", help="Marque e clique em Salvar.", default=False),
        }

        # fill project number
        proj_number = str(row.get("cod_projeto") or row.get("project_code") or row.get("project_number") or f"{pid}")
        doc_edit["project_number"] = proj_number
        if "status" in doc_edit.columns and doc_edit["status"].isna().any():
            doc_edit["status"] = doc_edit["status"].fillna("Em andamento - BK")
        if "revision_code" in doc_edit.columns and doc_edit["revision_code"].isna().any():
            doc_edit["revision_code"] = doc_edit["revision_code"].fillna("R0A")

        edited = st.data_editor(
            doc_edit,
            num_rows="dynamic",
            use_container_width=True,
            column_config=col_cfg,
            hide_index=True,
            key="doc_editor",
        )

        # pós-processamento: preencher service_name ao selecionar service_id
        edited_df = pd.DataFrame(edited)
        if not edited_df.empty and services_map:
            def _fill_name(rowx):
                sid = rowx.get("service_id", None)
                try:
                    sid_i = int(sid) if sid not in (None,"") and not pd.isna(sid) else None
                except Exception:
                    sid_i = None
                if sid_i is not None:
                    rowx["service_name"] = services_map.get(sid_i, rowx.get("service_name",""))
                return rowx
            edited_df = edited_df.apply(_fill_name, axis=1)

        if st.button("💾 Salvar tabela", type="primary", use_container_width=True):
            ins, upd = _upsert_doc_tasks(SessionLocal, pid, edited_df, services_map, proj_number)
            st.success(f"Salvo. Inseridos: {ins} | Atualizados: {upd}.")
            st.rerun()

        st.divider()
        st.subheader("Métricas (BK x Cliente)")
        metrics = _compute_doc_metrics(engine, pid)
        if metrics.empty:
            st.info("Sem histórico ainda. Dica: ao mudar o status, o sistema registra eventos e passa a calcular os tempos.")
        else:
            # join para mostrar por tarefa
            view = doc_df.merge(metrics, left_on="id", right_on="doc_task_id", how="left")
            view = view.fillna({"dias_elaboracao_BK":0,"dias_revisao_BK":0,"dias_analise_CLIENTE":0,"revisoes_qtd":0,"dias_total":0,"revision_code":"R0A"})
            total_revs = int(view["revisoes_qtd"].sum()) if not view.empty else 0
            st.metric("Total de revisões (projeto)", total_revs)

            # charts
            top_total = view.sort_values("dias_total", ascending=False).head(10)
            if not top_total.empty:
                st.write("Top 10 - Tempo total")
                chart_df = top_total[["service_name","dias_total"]].copy()
                chart_df = chart_df.set_index("service_name")
                st.bar_chart(chart_df)

            c3, c4 = st.columns(2)
            with c3:
                top_cli = view.sort_values("dias_analise_CLIENTE", ascending=False).head(10)
                if not top_cli.empty:
                    st.write("Top 10 - Análise (Cliente)")
                    st.bar_chart(top_cli.set_index("service_name")[["dias_analise_CLIENTE"]])
            with c4:
                top_bk = view.sort_values("dias_revisao_BK", ascending=False).head(10)
                if not top_bk.empty:
                    st.write("Top 10 - Revisão (BK)")
                    st.bar_chart(top_bk.set_index("service_name")[["dias_revisao_BK"]])

            st.write("Tabela detalhada")
            st.dataframe(view[[
                "id","service_name","status","revision_code","revisoes_qtd",
                "dias_elaboracao_BK","dias_revisao_BK","dias_analise_CLIENTE","dias_total"
            ]], use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Exportação (com logos)")
            # nomes
            project_name = str(row.get("nome") or "")
            client_name = str(row.get("client_name") or row.get("cliente") or "")
            html_doc = _build_doc_report_html(project_name, client_name, logo_bk_b64, logo_cli_b64, doc_df, metrics)

            st.download_button("⬇️ Baixar relatório DOC (HTML)", data=html_doc.encode("utf-8"),
                               file_name=f"controle_documentos_projeto_{pid}.html", mime="text/html", use_container_width=True)
            csv_bytes = view.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Baixar CSV (tabela+tempos)", data=csv_bytes,
                               file_name=f"controle_documentos_projeto_{pid}.csv", mime="text/csv", use_container_width=True)


# -------------------------
# Projetos: leitura/atualização (defensivo)
# -------------------------
def _list_projects_defensive(engine) -> pd.DataFrame:
    insp = inspect(engine)
    try:
        cols_meta = insp.get_columns("projects")
        cols = [c["name"] for c in cols_meta]
    except Exception:
        cols = []

    # campos lógicos
    planned_col = _pick_column(cols, ["planned_end_date", "planned_end", "dataInicio", "data"])
    actual_col = _pick_column(cols, ["actual_end_date", "actual_end", "data_conclusao"])
    progress_col = _pick_column(cols, ["progress_pct", "progresso", "progresso_pct"])
    delay_col = _pick_column(cols, ["delay_responsibility", "atraso_responsabilidade", "responsavel_atraso"])
    code_col = _pick_column(cols, ["cod_projeto", "project_code", "project_number", "numero_projeto"])
    client_name_col = _pick_column(cols, ["client_name", "cliente", "nome_cliente"])

    sel = ["id", "nome", "status"]
    if code_col:
        sel.append(f"{code_col} AS cod_projeto")
    else:
        sel.append("NULL AS cod_projeto")
    if client_name_col:
        sel.append(f"{client_name_col} AS client_name")
    else:
        sel.append("NULL AS client_name")
    if planned_col:
        sel.append(f"{planned_col} AS planned_end_date")
    else:
        sel.append("NULL AS planned_end_date")
    if actual_col:
        sel.append(f"{actual_col} AS actual_end_date")
    else:
        sel.append("NULL AS actual_end_date")
    if progress_col:
        sel.append(f"{progress_col} AS progress_pct")
    else:
        sel.append("0 AS progress_pct")
    if delay_col:
        sel.append(f"{delay_col} AS delay_responsibility")
    else:
        sel.append("'N/A' AS delay_responsibility")

    sql = f"SELECT {', '.join(sel)} FROM projects ORDER BY id DESC"
    try:
        df = pd.read_sql(sql, engine)
    except Exception:
        df = pd.DataFrame(columns=["id","nome","status","cod_projeto","client_name","planned_end_date","actual_end_date","progress_pct","delay_responsibility"])
    return df


def _update_project(SessionLocal, project_id: int, status: str, progress_pct: int,
                    planned_end: date, actual_end: date, delay_resp: str) -> None:
    with SessionLocal() as session:
        # tenta colunas comuns
        # status, progress_pct, planned_end_date, actual_end_date, delay_responsibility
        # (se alguma não existir, o update pode falhar; nesse caso, ignore silenciosamente)
        try:
            session.execute(
                text("""
                    UPDATE projects SET
                        status=:st,
                        progress_pct=:pp,
                        planned_end_date=:pl,
                        actual_end_date=:ac,
                        delay_responsibility=:dr
                    WHERE id=:id
                """),
                {"st": status, "pp": int(progress_pct), "pl": planned_end.isoformat(), "ac": actual_end.isoformat(), "dr": delay_resp, "id": int(project_id)},
            )
            session.commit()
        except Exception:
            # fallback: atualiza só status (mínimo)
            try:
                session.execute(text("UPDATE projects SET status=:st WHERE id=:id"), {"st": status, "id": int(project_id)})
                session.commit()
            except Exception:
                session.rollback()


def _get_renderer(engine, SessionLocal):
    """Tenta usar renderer existente (mantém compatibilidade)."""
    try:
        from reports.render_controle_projetos import build_report as rc_build_report  # type: ignore

        def _wrapper(project_id: Optional[int] = None) -> str:
            projects = _list_projects_defensive(engine)
            if project_id is not None:
                projects = projects[projects["id"] == int(project_id)]
            # tasks "antigas" não são mais o foco; mas mantemos vazio para compatibilidade
            tasks_df = pd.DataFrame(columns=["id","title","due_date","is_done","assigned_to","delay_responsibility","project_id","data_conclusao","status_tarefa"])
            today = date.today()
            projects_local = projects.copy()
            projects_local["planned_end_date_dt"] = pd.to_datetime(projects_local.get("planned_end_date", pd.NaT), errors="coerce").dt.date
            projects_overdue = projects_local[projects_local["planned_end_date_dt"].notna() & (projects_local["planned_end_date_dt"] < today)]
            tasks_overdue = tasks_df.iloc[0:0]
            return rc_build_report(projects, tasks_df, projects_overdue, tasks_overdue)

        return _wrapper
    except Exception:
        def _fallback(project_id: Optional[int] = None) -> str:
            return "<html><body><h1>Relatório indisponível</h1></body></html>"
        return _fallback


if __name__ == "__main__":
    main()
