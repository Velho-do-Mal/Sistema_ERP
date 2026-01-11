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


def _ensure_doc_tables(engine, SessionLocal) -> None:
    """Cria/atualiza tabelas usadas no Controle de Projetos.

    Importante: faz commit dentro da mesma sessão (no Postgres/Neon isso evita erros de
    'ProgrammingError' por DDL não persistido entre sessões).
    """
    dialect = (getattr(getattr(engine, "dialect", None), "name", "") or "").lower()

    ddl_pg = """
    CREATE TABLE IF NOT EXISTS project_doc_tasks (
        id SERIAL PRIMARY KEY,
        project_id INTEGER NOT NULL,
        service_id INTEGER NULL,
        service_name TEXT,
        doc_name TEXT,
        doc_number TEXT,
        complemento TEXT,
        project_number TEXT,
        start_date DATE,
        delivery_date DATE,
        status TEXT,
        revision_code TEXT,
        observation TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    ddl_sqlite = """
    CREATE TABLE IF NOT EXISTS project_doc_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        service_id INTEGER NULL,
        service_name TEXT,
        doc_name TEXT,
        doc_number TEXT,
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

    # DDL + migrações (tudo com commit)
    try:
        with SessionLocal() as session:
            # Cria tabelas (tenta Postgres, senão SQLite)
            try:
                session.execute(text(ddl_pg))
            except Exception:
                session.execute(text(ddl_sqlite))

            try:
                session.execute(text(ddl_events_pg))
            except Exception:
                session.execute(text(ddl_events_sqlite))

            # Migração defensiva: adiciona colunas novas (doc_name / doc_number)
            if dialect == "postgresql":
                session.execute(text("ALTER TABLE project_doc_tasks ADD COLUMN IF NOT EXISTS doc_name TEXT"))
                session.execute(text("ALTER TABLE project_doc_tasks ADD COLUMN IF NOT EXISTS doc_number TEXT"))
            elif dialect == "sqlite":
                cols = []
                try:
                    cols = [r[1] for r in session.execute(text("PRAGMA table_info(project_doc_tasks)")).fetchall()]
                except Exception:
                    cols = []
                if "doc_name" not in cols:
                    session.execute(text("ALTER TABLE project_doc_tasks ADD COLUMN doc_name TEXT"))
                if "doc_number" not in cols:
                    session.execute(text("ALTER TABLE project_doc_tasks ADD COLUMN doc_number TEXT"))

            # Migração defensiva: alinha nomes antigos (inicio/conclusao/observacao) com os atuais (start_date/delivery_date/observation)
            # e garante colunas usadas pela app (project_number/start_date/delivery_date/observation).
            if dialect == "postgresql":
                session.execute(text("ALTER TABLE project_doc_tasks ADD COLUMN IF NOT EXISTS project_number TEXT"))
                session.execute(text("ALTER TABLE project_doc_tasks ADD COLUMN IF NOT EXISTS start_date DATE"))
                session.execute(text("ALTER TABLE project_doc_tasks ADD COLUMN IF NOT EXISTS delivery_date DATE"))
                session.execute(text("ALTER TABLE project_doc_tasks ADD COLUMN IF NOT EXISTS observation TEXT"))
                # se vier de versões antigas
                session.execute(text("ALTER TABLE project_doc_tasks ADD COLUMN IF NOT EXISTS inicio DATE"))
                session.execute(text("ALTER TABLE project_doc_tasks ADD COLUMN IF NOT EXISTS conclusao DATE"))
                session.execute(text("ALTER TABLE project_doc_tasks ADD COLUMN IF NOT EXISTS observacao TEXT"))
                # copia valores antigos para os novos (quando os novos estiverem nulos)
                session.execute(text("UPDATE project_doc_tasks SET start_date = COALESCE(start_date, inicio)"))
                session.execute(text("UPDATE project_doc_tasks SET delivery_date = COALESCE(delivery_date, conclusao)"))
                session.execute(text("UPDATE project_doc_tasks SET observation = COALESCE(observation, observacao)"))
            elif dialect == "sqlite":
                cols = []
                try:
                    cols = [r[1] for r in session.execute(text("PRAGMA table_info(project_doc_tasks)")).fetchall()]
                except Exception:
                    cols = []
                for col_def in [
                    ("project_number", "TEXT"),
                    ("start_date", "DATE"),
                    ("delivery_date", "DATE"),
                    ("observation", "TEXT"),
                    ("inicio", "DATE"),
                    ("conclusao", "DATE"),
                    ("observacao", "TEXT"),
                ]:
                    cname, ctype = col_def
                    if cname not in cols:
                        session.execute(text(f"ALTER TABLE project_doc_tasks ADD COLUMN {cname} {ctype}"))
                # tenta copiar valores (SQLite aceita UPDATE com colunas existentes)
                try:
                    session.execute(text("UPDATE project_doc_tasks SET start_date = COALESCE(start_date, inicio)"))
                    session.execute(text("UPDATE project_doc_tasks SET delivery_date = COALESCE(delivery_date, conclusao)"))
                    session.execute(text("UPDATE project_doc_tasks SET observation = COALESCE(observation, observacao)"))
                except Exception:
                    pass

            session.commit()
    except Exception:
        # não bloqueia a app; logs no Cloud mostram detalhes
        pass
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
    """Lista linhas da tabela estilo Excel. Usa text()+conn para evitar erro de paramstyle no Cloud."""
    _ensure_doc_tables(engine, SessionLocal)
    sql = text("""
        SELECT id, service_id, COALESCE(service_name,'') AS service_name,
               COALESCE(doc_name,'') AS doc_name,
               COALESCE(doc_number,'') AS doc_number,
               COALESCE(complemento,'') AS complemento,
               COALESCE(project_number,'') AS project_number,
               start_date, delivery_date,
               COALESCE(status,'Em andamento - BK') AS status,
               COALESCE(revision_code,'R0A') AS revision_code,
               COALESCE(observation,'') AS observation
        FROM project_doc_tasks
        WHERE project_id=:pid
        ORDER BY id ASC
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(sql, conn, params={"pid": int(project_id)})
    except Exception:
        # Se ainda não existir (ou migração atrasada), tenta garantir tabelas e retornar vazio ao invés de quebrar a página
        try:
            _ensure_doc_tables(engine, SessionLocal)
            with engine.connect() as conn:
                df = pd.read_sql_query(sql, conn, params={"pid": int(project_id)})
        except Exception:
            return pd.DataFrame(columns=['id','service_id','service_name','doc_name','doc_number','complemento','project_number','start_date','delivery_date','status','revision_code','observation'])

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
                      project_number: str) -> Tuple[int, int, int]:
    """Upsert + eventos. Retorna (inseridos, atualizados)."""
    inserted = 0
    updated = 0
    deleted = 0

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

        doc_name = str(r.get("doc_name", "") or "").strip()
        doc_number = str(r.get("doc_number", "") or "").strip()
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
                # apaga eventos primeiro (evita violação de FK)
                session.execute(text("DELETE FROM doc_status_events WHERE doc_task_id=:id"), {"id": int(rid)})
                session.execute(text("DELETE FROM project_doc_tasks WHERE id=:id"), {"id": int(rid)})
                session.commit()
            deleted += 1
            continue

        # --------- INSERT ---------
        if rid in (None, "", pd.NA) or pd.isna(rid):
            # regra: ao iniciar, revisão é R0A
            revision_code = "R0A"
            with SessionLocal() as session:
                res = session.execute(
                    text("""
                        INSERT INTO project_doc_tasks
                        (project_id, service_id, service_name, doc_name, doc_number, complemento, project_number,
                         start_date, delivery_date, status, revision_code, observation, updated_at)
                        VALUES (:pid, :sid, :sname, :dname, :dnum, :comp, :pnum,
                                :sd, :dd, :st, :rev, :obs, CURRENT_TIMESTAMP)
                    """),
                    {
                        "pid": int(project_id),
                        "sid": service_id_int,
                        "sname": service_name,
                        "dname": doc_name,
                        "dnum": doc_number,
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
                        doc_name=:dname,
                        doc_number=:dnum,
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

    return inserted, updated, deleted
def _compute_doc_metrics(engine, project_id: int) -> pd.DataFrame:
    """Retorna DF por doc_task_id com tempos BK/Cliente e revisões."""
    sql = text("""
        SELECT doc_task_id, event_date, status, responsible, COALESCE(revision_code,'') AS revision_code
        FROM doc_status_events
        WHERE project_id=:pid
        ORDER BY doc_task_id ASC, event_date ASC, id ASC
    """)
    try:
        with engine.connect() as conn:
            ev = pd.read_sql_query(sql, conn, params={"pid": int(project_id)})
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
    rep = rep.fillna({
        "dias_elaboracao_BK": 0,
        "dias_revisao_BK": 0,
        "dias_analise_CLIENTE": 0,
        "revisoes_qtd": 0,
        "dias_total": 0,
        "revision_code": "R0A"
    })

    # charts
    top = rep.sort_values("dias_total", ascending=False).head(10)
    top_rev = rep.sort_values("dias_revisao_BK", ascending=False).head(10)
    top_ana = rep.sort_values("dias_analise_CLIENTE", ascending=False).head(10)
    top_revcount = rep.sort_values("revisoes_qtd", ascending=False).head(10)

    charts = []
    for dfx, y, ttl in [
        (top, "dias_total", "Top 10 - Tempo total (BK + Cliente)"),
        (top_ana, "dias_analise_CLIENTE", "Top 10 - Análise (Cliente)"),
        (top_rev, "dias_revisao_BK", "Top 10 - Revisão (BK)"),
        (top_revcount, "revisoes_qtd", "Top 10 - Nº de revisões"),
    ]:
        if dfx.empty:
            charts.append("")
            continue
        dfx = dfx.copy()
        dfx["label"] = dfx.apply(lambda r: f"#{int(r['id'])} {str(r.get('service_name',''))}".strip(), axis=1)
        b64 = _chart_png_b64(dfx[["label", y]], "label", y, ttl)
        charts.append(f'<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;" />' if b64 else "")

    # table html
    cols = [
        "service_name", "complemento", "project_number",
        "start_date", "delivery_date",
        "status", "revision_code", "revisoes_qtd",
        "dias_elaboracao_BK", "dias_revisao_BK", "dias_analise_CLIENTE", "dias_total",
        "observation"
    ]
    for c in cols:
        if c not in rep.columns:
            rep[c] = ""
    th = "".join([f"<th>{_html_escape(c)}</th>" for c in cols])
    rows_html = []
    for _, r in rep[cols].iterrows():
        tds = "".join([f"<td>{_html_escape(r.get(c,''))}</td>" for c in cols])
        rows_html.append(f"<tr>{tds}</tr>")
    table_html = f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"

    def _img(b64: str) -> str:
        if not b64:
            return ""
        return f'<img src="data:image/png;base64,{b64}" style="max-height:80px;max-width:160px;object-fit:contain;" />'

    return f"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8" />
<title>Controle de Projetos - Relatório</title>
<style>
    body {{
        font-family: Arial, Helvetica, sans-serif;
        margin: 24px;
        color: #111827;
    }}
    .header {{
        display: grid;
        grid-template-columns: 180px 1fr 180px;
        align-items: center;
        gap: 12px;
        padding: 12px 0 18px 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 18px;
    }}
    .title {{
        text-align: center;
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 0.2px;
    }}
    .subtitle {{
        text-align: center;
        margin-top: 6px;
        font-size: 14px;
        color: #374151;
    }}
    .meta {{
        margin: 10px 0 14px 0;
        font-size: 13px;
        color: #374151;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }}
    th, td {{
        border: 1px solid #d1d5db; /* cinza claro */
        padding: 6px 8px;
        vertical-align: top;
    }}
    th {{
        background: #f3f4f6;
        text-transform: none;
        font-weight: 700;
    }}
    .section {{
        margin-top: 18px;
    }}
    .section h2 {{
        font-size: 16px;
        margin: 0 0 10px 0;
    }}
    .charts img {{
        margin: 10px 0;
    }}
</style>
</head>
<body>

<div class="header">
  <div style="text-align:left;">{_img(logo_bk_b64)}</div>
  <div>
    <div class="title">BK Engenharia e Tecnologia</div>
    <div class="subtitle">Relatório - Controle de Projetos</div>
  </div>
  <div style="text-align:right;">{_img(logo_cli_b64)}</div>
</div>

<div class="meta">
  <div><b>Cliente:</b> {_html_escape(client_name)}</div>
  <div><b>Projeto:</b> {_html_escape(project_name)}</div>
</div>

<div class="section">
  <h2>Tarefas / Documentos</h2>
  {table_html}
</div>

<div class="section charts">
  <h2>Gráficos</h2>
  {charts[0]}
  {charts[1]}
  {charts[2]}
  {charts[3]}
</div>

</body>
</html>"""

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

        status_opts = ["em_andamento","em_aprovacao","em_analise","em_revisao","aprovado","concluido","encerrado","rascunho"]
        cur_status = str(row.get("status", "") or "em_andamento").strip()
        if cur_status not in status_opts:
            status_opts = [cur_status] + status_opts
        status = st.selectbox("Status", options=status_opts, index=status_opts.index(cur_status))
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

        if st.button("Salvar projeto", type="primary", width='stretch'):
            _update_project(SessionLocal, pid, status, progress, planned, actual, resp)
            st.success("Projeto atualizado.")
            st.rerun()

        st.divider()
        st.subheader("Relatório geral (HTML)")

        # Usa os logos enviados no lado direito (keys: logo_bk / logo_cli)
        def _upload_to_b64(upl):
            if upl is None:
                return ""
            try:
                data = upl.getvalue()
                mime = getattr(upl, "type", "") or ""
            except Exception:
                data = None
                mime = ""
            if not data:
                return ""
            b64 = base64.b64encode(data).decode("utf-8")
            if not mime:
                mime = "image/png"
            return f"data:{mime};base64,{b64}"

        def _fmt_days(v):
            try:
                return int(v)
            except Exception:
                return 0

        def _chart_png_b64(df, x_col, y_col, title):
            # gera um gráfico simples em PNG e retorna data-uri
            try:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(8, 3))
                ax = fig.add_subplot(111)
                ax.bar(df[x_col].astype(str).tolist(), df[y_col].astype(float).tolist())
                ax.set_title(title)
                ax.set_ylabel("Dias")
                ax.tick_params(axis='x', labelrotation=45)
                fig.tight_layout()
                bio = BytesIO()
                fig.savefig(bio, format="png", dpi=150)
                plt.close(fig)
                b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
                return f"data:image/png;base64,{b64}"
            except Exception:
                return ""

        # Monta HTML do relatório com tabela + gráficos (padrão BK)
        row_report = _load_project(engine, pid) or {}
        proj_nome = str(row_report.get("nome") or row_report.get("project_name") or f"Projeto {pid}")
        cliente = str(row_report.get("client_name") or row_report.get("cliente") or "Cliente")
        proj_code = str(row_report.get("cod_projeto") or row_report.get("project_code") or row_report.get("project_number") or pid)

        doc_df_r = _list_doc_tasks(engine, SessionLocal, pid)
        view_r = _compute_doc_metrics(engine, pid)
        if not doc_df_r.empty:
            # normaliza datas
            for c in ["start_date","delivery_date"]:
                if c in doc_df_r.columns:
                    doc_df_r[c] = pd.to_datetime(doc_df_r[c], errors="coerce").dt.date

        logo_bk_uri = _upload_to_b64(st.session_state.get("logo_bk"))
        logo_cli_uri = _upload_to_b64(st.session_state.get("logo_cli"))

        # tabela de saída (todas as colunas pedidas)
        table_cols = ["service_name","doc_name","doc_number","complemento","project_number","start_date","delivery_date","status","revision_code","observation","responsavel"]
        df_out = doc_df_r.copy()
        for c in table_cols:
            if c not in df_out.columns:
                df_out[c] = ""
        df_out = df_out[table_cols].copy()
        df_out = df_out.rename(columns={
            "service_name":"Tarefa (Serviço)",
            "complemento":"Complemento",
            "project_number":"Nº do projeto",
            "start_date":"Data de início",
            "delivery_date":"Data de conclusão",
            "status":"Status",
            "revision_code":"Nº Revisão",
            "observation":"Observação",
        })

        # Gráficos (Top 10)
        chart_total_uri = ""
        chart_cli_uri = ""
        chart_bk_uri = ""
        if not view_r.empty:
            v = view_r.copy()
            v["dias_total"] = v["dias_total"].apply(_fmt_days)
            v["dias_analise_CLIENTE"] = v["dias_analise_CLIENTE"].apply(_fmt_days)
            v["dias_revisao_BK"] = v["dias_revisao_BK"].apply(_fmt_days)

            top_total = v.sort_values("dias_total", ascending=False).head(10)
            top_cli = v.sort_values("dias_analise_CLIENTE", ascending=False).head(10)
            top_bk = v.sort_values("dias_revisao_BK", ascending=False).head(10)

            if not top_total.empty:
                chart_total_uri = _chart_png_b64(top_total, "service_name", "dias_total", "Top 10 - Tempo total (dias)")
            if not top_cli.empty:
                chart_cli_uri = _chart_png_b64(top_cli, "service_name", "dias_analise_CLIENTE", "Top 10 - Análise (Cliente) (dias)")
            if not top_bk.empty:
                chart_bk_uri = _chart_png_b64(top_bk, "service_name", "dias_revisao_BK", "Top 10 - Revisão (BK) (dias)")

        # HTML + CSS (bordas finas cinza claro)
        css = """
        <style>
          body { font-family: Arial, Helvetica, sans-serif; color: #111; margin: 24px; }
          .hdr { width: 100%; border: 1px solid #e0e0e0; border-radius: 10px; padding: 14px; }
          .hdr-grid { display: grid; grid-template-columns: 140px 1fr 140px; align-items: center; gap: 10px; }
          .hdr img { max-height: 64px; max-width: 120px; }
          .title { text-align: center; }
          .title h1 { margin: 0; font-size: 22px; }
          .title .sub { margin-top: 6px; font-size: 14px; color: #333; }
          .title .sub2 { margin-top: 2px; font-size: 13px; color: #444; }
          h2 { margin: 18px 0 10px; font-size: 18px; }
          table { width: 100%; border-collapse: collapse; }
          th, td { border: 1px solid #d8d8d8; padding: 6px 8px; font-size: 12px; }
          th { background: #f5f5f5; }
          .charts { margin-top: 14px; display: grid; grid-template-columns: 1fr; gap: 14px; }
          .chart { border: 1px solid #e0e0e0; border-radius: 10px; padding: 10px; }
          .chart img { width: 100%; height: auto; }
        </style>
        """

        left_logo_html = f'<img src="{logo_bk_uri}" />' if logo_bk_uri else ''
        right_logo_html = f'<img src="{logo_cli_uri}" />' if logo_cli_uri else ''

        html_parts = ['<html><head><meta charset="utf-8"/>', css, '</head><body>']
        html_parts.append('<div class="hdr">')
        html_parts.append('<div class="hdr-grid">')
        html_parts.append(f'<div style="text-align:left;">{left_logo_html}</div>')
        html_parts.append('<div class="title">'
                          '<h1>BK Engenharia e Tecnologia</h1>'
                          f'<div class="sub">Cliente: {cliente}</div>'
                          f'<div class="sub2">Projeto: {proj_nome} &nbsp;&nbsp;|&nbsp;&nbsp; Nº: {proj_code}</div>'
                          '</div>')
        html_parts.append(f'<div style="text-align:right;">{right_logo_html}</div>')
        html_parts.append('</div></div>')

        html_parts.append('<h2>Tarefas / Documentos</h2>')
        html_parts.append(df_out.to_html(index=False, escape=False))

        # gráficos
        html_parts.append('<div class="charts">')
        for uri in [chart_total_uri, chart_cli_uri, chart_bk_uri]:
            if uri:
                html_parts.append('<div class="chart">')
                html_parts.append(f'<img src="{uri}" />')
                html_parts.append('</div>')
        html_parts.append('</div>')

        html_parts.append('</body></html>')
        html_report = "\n".join(html_parts)

        st.download_button(
            "📄 Baixar relatório (HTML)",
            data=html_report.encode("utf-8"),
            file_name=f"controle_projetos_{pid}.html",
            mime="text/html",
            width='stretch',
        )
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
        services_name_to_id = {v: k for k, v in services_map.items()}
        service_names = sorted(list(services_name_to_id.keys()))
        service_labels = {f"{int(r.id)} - {r.name}": int(r.id) for r in services_df.itertuples()} if not services_df.empty else {}
        # data
        doc_df = _list_doc_tasks(engine, SessionLocal, pid)
        if doc_df.empty:
            doc_df = pd.DataFrame(columns=[
                "id","service_id","service_name","complemento","project_number","start_date","delivery_date","status","revision_code","observation"
            ])

        # add helper column
        doc_edit = doc_df.copy()
        # Coluna derivada (BK/CLIENTE) para ficar claro quem está com a tarefa
        doc_edit["responsavel"] = doc_edit.get("status", "").apply(lambda s: _status_to_responsible(str(s)))
        doc_edit["Excluir"] = False


# formulário rápido para adicionar uma linha (deixa claro onde informar o serviço)
with st.expander("➕ Adicionar tarefa/documento", expanded=False):
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        new_service = st.selectbox("Tarefa (Serviço)", service_names if 'service_names' in locals() else [], index=0 if (('service_names' in locals()) and service_names) else None)
    with c2:
        new_start = st.date_input("Data de início", value=date.today(), key="new_doc_start")
    with c3:
        new_delivery = st.date_input("Data de conclusão (entrega p/ análise)", value=date.today(), key="new_doc_delivery")
    new_doc_name = st.text_input("Nome do documento", key="new_doc_name")
    new_doc_number = st.text_input("Nº do documento", key="new_doc_number")
    new_comp = st.text_input("Complemento", key="new_doc_comp")
    new_obs = st.text_area("Observação", key="new_doc_obs")
    if st.button("Adicionar linha", width='stretch'):
        if not new_service:
            st.warning("Selecione um serviço.")
        else:
            sid = services_name_to_id.get(str(new_service), None) if 'services_name_to_id' in locals() else None
            df_new = pd.DataFrame([{
                "id": None,
                "service_id": sid,
                "service_name": str(new_service),
                "complemento": new_comp,
                "project_number": proj_number,
                "start_date": new_start,
                "delivery_date": new_delivery,
                "status": "Em andamento - BK",
                "revision_code": "R0A",
                "observation": new_obs,
                "Excluir": False,
            }])
            _upsert_doc_tasks(SessionLocal, pid, df_new, services_map, proj_number)
            st.success("Linha adicionada.")
            st.rerun()

        # editor column config
        col_cfg = {
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "service_name": st.column_config.SelectboxColumn(
                "Tarefa (Serviço)",
                options=service_names if 'service_names' in locals() else [],
                help="Selecione o serviço (digite para pesquisar).",
                required=True,
            ),
            "doc_name": st.column_config.TextColumn("Nome do documento", required=False),
            "doc_number": st.column_config.TextColumn("Nº do documento", required=False),
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
        st.caption("Tabela estilo Excel: edite direto nas células. A coluna **Tarefa (Serviço)** é um combobox (clique e digite para pesquisar).")

        doc_edit_view = doc_edit.copy()
        # esconder colunas técnicas
        if 'service_id' in doc_edit_view.columns:
            doc_edit_view = doc_edit_view.drop(columns=['service_id'])


        # Reordena colunas principais (o que o usuário pediu ver primeiro)
        wanted = [c for c in ["id","service_name","doc_name","doc_number","revision_code","start_date","delivery_date","status","responsavel"] if c in doc_edit_view.columns]
        rest = [c for c in doc_edit_view.columns if c not in wanted]
        doc_edit_view = doc_edit_view[wanted + rest]

        # tenta AgGrid (modo PowerApps) — se não existir, usa st.data_editor
        aggrid_available = False
        try:
            from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
            aggrid_available = True
        except Exception:
            aggrid_available = False

        use_powerapps = False
        if aggrid_available:
            use_powerapps = st.toggle("Modo PowerApps/Excel (AgGrid)", value=True)

        edited_df = pd.DataFrame()
        selected_rows = []

        if aggrid_available and use_powerapps:
            gb = GridOptionsBuilder.from_dataframe(doc_edit_view)
            gb.configure_default_column(editable=True, resizable=True, filter=True)
            gb.configure_selection("multiple", use_checkbox=True)

            # combobox (dropdown) para o serviço
            if 'service_name' in doc_edit_view.columns and service_names:
                gb.configure_column(
                    "service_name",
                    headerName="Tarefa (Serviço)",
                    editable=True,
                    cellEditor="agSelectCellEditor",
                    cellEditorParams={"values": service_names},
                    width=220,
                )

            gb.configure_column("project_number", headerName="Nº do projeto", editable=False, width=110)
            gb.configure_column("start_date", headerName="Data de início", width=130)
            gb.configure_column("delivery_date", headerName="Data de conclusão", width=170)
            gb.configure_column("status", headerName="Status", editable=True, width=160)
            gb.configure_column("revision_code", headerName="Nº Revisão", editable=False, width=90)
            gb.configure_column("Excluir", headerName="Excluir", editable=True, width=90)

            grid = AgGrid(
                doc_edit_view,
                gridOptions=gb.build(),
                update_mode=GridUpdateMode.MODEL_CHANGED,
                data_return_mode=DataReturnMode.AS_INPUT,
                fit_columns_on_grid_load=True,
                height=360,
                key="doc_aggrid",
            )
            edited_df = pd.DataFrame(grid.get("data", []))
            selected_rows = grid.get("selected_rows") or []
        else:
            edited = st.data_editor(
                doc_edit_view,
                num_rows="dynamic",
                width='stretch',
                column_config=col_cfg,
                hide_index=True,
                key="doc_editor",
            )
            edited_df = pd.DataFrame(edited)

        # pós-processamento: mapear service_name -> service_id
        if not edited_df.empty and services_name_to_id:
            edited_df['service_id'] = edited_df.get('service_name', '').map(lambda n: services_name_to_id.get(str(n), None))

        # excluir selecionados (AgGrid)
        if selected_rows and st.button("🗑️ Excluir selecionados", type="secondary", width='stretch'):
            sel_ids = {int(r["id"]) for r in selected_rows if r.get("id") is not None}
            df_del = edited_df.copy()
            if "Excluir" not in df_del.columns:
                df_del["Excluir"] = False
            df_del.loc[df_del["id"].isin(sel_ids), "Excluir"] = True
            ins, upd, dele = _upsert_doc_tasks(SessionLocal, pid, df_del, services_map, proj_number)
            st.success(f"Excluídos: {dele}.")
            st.rerun()

        if st.button("💾 Salvar tabela", type="primary", width='stretch'):
            ins, upd, dele = _upsert_doc_tasks(SessionLocal, pid, edited_df, services_map, proj_number)
            st.success(f"Salvo. Inseridos: {ins} | Atualizados: {upd} | Excluídos: {dele}.")
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
            ]], width='stretch', hide_index=True)

            st.divider()
            st.subheader("Exportação (com logos)")
            # nomes
            project_name = str(row.get("nome") or "")
            client_name = str(row.get("client_name") or row.get("cliente") or "")
            html_doc = _build_doc_report_html(project_name, client_name, logo_bk_b64, logo_cli_b64, doc_df, metrics)

            st.download_button("⬇️ Baixar relatório DOC (HTML)", data=html_doc.encode("utf-8"),
                               file_name=f"controle_documentos_projeto_{pid}.html", mime="text/html", width='stretch')
            csv_bytes = view.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Baixar CSV (tabela+tempos)", data=csv_bytes,
                               file_name=f"controle_documentos_projeto_{pid}.csv", mime="text/csv", width='stretch')


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




def _load_project(engine, project_id: int) -> dict:
    """Carrega dados básicos do projeto para cabeçalho/relatórios (defensivo)."""
    try:
        df = _list_projects_defensive(engine)
        if df is None or df.empty:
            return {}
        row = df[df["id"] == int(project_id)]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()
    except Exception:
        return {}
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
