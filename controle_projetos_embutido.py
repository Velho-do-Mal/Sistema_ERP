# pages/6_🖥️_Controle_de_Projetos.py
# -*- coding: utf-8 -*-
"""
BK_ERP • Controle de Projetos (BK x Cliente)
-------------------------------------------
Requisitos atendidos (resumo):
- Independente do módulo Gestão de Projetos: usa apenas a lista de projetos cadastrados (tabela `projects`).
- Formulário para informações do relatório (cliente, nome do projeto, nº do projeto, status do projeto, logos).
- Tabela estilo Excel (st.data_editor) com: ID, Serviço, Nome do documento, Nº do documento, Revisão, Responsável,
  Data de início, Data de conclusão, Status, Observação, Excluir.
- Contabiliza tempos BK x Cliente por histórico de status (múltiplas análises e revisões).
- Incrementa revisão automaticamente ao voltar para "Em revisão - BK" após "Em análise - Cliente" (R0A -> R0B -> ...).
- Relatório HTML: cabeçalho com logos + cliente/projeto + tabela completa + gráficos (Top 10 tempos).
- Compatível com Streamlit Cloud + Neon/Postgres (sem placeholders ":pid" quebrando no driver).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import text



# --- Compatibilidade de schema (migrações leves em runtime) -----------------
def _ensure_doc_tasks_schema(engine) -> None:
    """Garante colunas novas em `project_doc_tasks` sem exigir migração manual.

    - responsible_bk: responsável BK pelo documento (texto).
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE project_doc_tasks ADD COLUMN IF NOT EXISTS responsible_bk TEXT"))
    except Exception:
        # Se a tabela ainda não existir (primeira execução) ou banco não suportar IF NOT EXISTS,
        # deixamos o erro para o ponto de criação de tabelas. Assim evitamos quebrar a página.
        return

from bk_erp_shared.auth import login_and_guard
from bk_erp_shared.erp_db import ensure_erp_tables, get_finance_db
from bk_erp_shared.theme import apply_theme

# ------------------------------
# Constantes / opções
# ------------------------------
STATUS_OPTIONS = [
    "Em andamento - BK",     # conta BK
    "Em análise - Cliente",  # conta Cliente
    "Em revisão - BK",       # conta BK
    "Aprovado - Cliente",    # para de contar
]

RESPONSIBLE_OPTIONS = ["BK", "CLIENTE", "N/A"]

# ------------------------------
# Helpers
# ------------------------------
def _to_date(v, fallback: Optional[date] = None) -> Optional[date]:
    if v is None:
        return fallback
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
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
    try:
        return datetime.fromisoformat(s.replace("/", "-")).date()
    except Exception:
        return fallback


def _file_to_data_uri(uploaded) -> str:
    """Converte UploadedFile (PNG/JPG/SVG) para data-uri (string)."""
    if not uploaded:
        return ""
    name = (getattr(uploaded, "name", "") or "").lower()
    data = uploaded.getvalue()
    if name.endswith(".svg"):
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/svg+xml;base64,{b64}"
    # default png/jpg
    mime = "image/png"
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        mime = "image/jpeg"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _status_kind(status: str) -> str:
    s = (status or "").lower()
    if "análise" in s or "analise" in s:
        return "analise"
    if "revis" in s:
        return "revisao"
    if "aprov" in s:
        return "aprovado"
    return "bk"  # andamento/elaboração


def _responsible_from_status(status: str) -> str:
    k = _status_kind(status)
    if k == "analise":
        return "CLIENTE"
    if k == "aprovado":
        return "CLIENTE"
    return "BK"


def _next_revision(code: str) -> str:
    """R0A..R0Z + fallback."""
    c = (code or "").strip().upper()
    if not c.startswith("R0") or len(c) < 3:
        return "R0A"
    letter = c[2]
    if not ("A" <= letter <= "Z"):
        return "R0A"
    if letter == "Z":
        return "R0Z"
    return f"R0{chr(ord(letter) + 1)}"


# ------------------------------
# Banco / schema
# ------------------------------
def _ensure_control_tables(engine, SessionLocal) -> None:
    """Cria tabelas do Controle de Projetos + migrações leves."""
    dialect = (getattr(getattr(engine, "dialect", None), "name", "") or "").lower()

    ddl_meta_pg = """
    CREATE TABLE IF NOT EXISTS project_control_meta (
        project_id INTEGER PRIMARY KEY,
        client_name TEXT,
        project_name TEXT,
        project_number TEXT,
        project_status TEXT,
        logo_bk_uri TEXT,
        logo_client_uri TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    ddl_meta_sqlite = """
    CREATE TABLE IF NOT EXISTS project_control_meta (
        project_id INTEGER PRIMARY KEY,
        client_name TEXT,
        project_name TEXT,
        project_number TEXT,
        project_status TEXT,
        logo_bk_uri TEXT,
        logo_client_uri TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """

    ddl_tasks_pg = """
    CREATE TABLE IF NOT EXISTS project_doc_tasks (
        id SERIAL PRIMARY KEY,
        project_id INTEGER NOT NULL,
        service_id INTEGER NULL,
        service_name TEXT,
        doc_name TEXT,
        doc_number TEXT,
        revision_code TEXT,
        responsible TEXT,
        responsible_bk TEXT,
        inicio DATE,
        conclusao DATE,
        status TEXT,
        observacao TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    ddl_tasks_sqlite = """
    CREATE TABLE IF NOT EXISTS project_doc_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        service_id INTEGER NULL,
        service_name TEXT,
        doc_name TEXT,
        doc_number TEXT,
        revision_code TEXT,
        responsible TEXT,
        responsible_bk TEXT,
        inicio DATE,
        conclusao DATE,
        status TEXT,
        observacao TEXT,
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

    ddl_meta = ddl_meta_pg if "postgres" in dialect else ddl_meta_sqlite
    ddl_tasks = ddl_tasks_pg if "postgres" in dialect else ddl_tasks_sqlite
    ddl_events = ddl_events_pg if "postgres" in dialect else ddl_events_sqlite

    with SessionLocal() as session:
        session.execute(text(ddl_meta))
        session.execute(text(ddl_tasks))
        session.execute(text(ddl_events))
        session.commit()

        # migração defensiva: caso versões anteriores não tivessem 'responsible' ou 'doc_number'
        try:
            session.execute(text("ALTER TABLE project_doc_tasks ADD COLUMN responsible TEXT"))
            session.commit()
        except Exception:
            session.rollback()
        try:
            session.execute(text("ALTER TABLE project_doc_tasks ADD COLUMN responsible_bk TEXT"))
            session.commit()
        except Exception:
            session.rollback()
        try:
            session.execute(text("ALTER TABLE project_doc_tasks ADD COLUMN doc_number TEXT"))
            session.commit()
        except Exception:
            session.rollback()


# ------------------------------
# Queries
# ------------------------------
def _list_projects(engine) -> pd.DataFrame:
    sql = text("""
        SELECT id, nome, status, dataInicio, gerente
        FROM projects
        ORDER BY id DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn)
    return df


def _list_services(engine) -> pd.DataFrame:
    sql = text("""
        SELECT id, name
        FROM product_services
        WHERE COALESCE(type,'servico')='servico'
        ORDER BY name ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn)
    return df


def _load_meta(engine, project_id: int) -> Dict:
    sql = text("""
        SELECT project_id, client_name, project_name, project_number, project_status,
               logo_bk_uri, logo_client_uri
        FROM project_control_meta
        WHERE project_id = :pid
        LIMIT 1
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params={"pid": int(project_id)})
    if df.empty:
        return {}
    return dict(df.iloc[0].to_dict())


def _upsert_meta(SessionLocal, project_id: int, meta: Dict) -> None:
    with SessionLocal() as session:
        # tenta update
        upd = session.execute(
            text("""
                UPDATE project_control_meta
                SET client_name=:client_name,
                    project_name=:project_name,
                    project_number=:project_number,
                    project_status=:project_status,
                    logo_bk_uri=:logo_bk_uri,
                    logo_client_uri=:logo_client_uri,
                    updated_at=CURRENT_TIMESTAMP
                WHERE project_id=:pid
            """),
            {
                "pid": int(project_id),
                "client_name": meta.get("client_name"),
                "project_name": meta.get("project_name"),
                "project_number": meta.get("project_number"),
                "project_status": meta.get("project_status"),
                "logo_bk_uri": meta.get("logo_bk_uri"),
                "logo_client_uri": meta.get("logo_client_uri"),
            },
        )
        if upd.rowcount == 0:
            session.execute(
                text("""
                    INSERT INTO project_control_meta
                    (project_id, client_name, project_name, project_number, project_status, logo_bk_uri, logo_client_uri)
                    VALUES (:pid, :client_name, :project_name, :project_number, :project_status, :logo_bk_uri, :logo_client_uri)
                """),
                {
                    "pid": int(project_id),
                    "client_name": meta.get("client_name"),
                    "project_name": meta.get("project_name"),
                    "project_number": meta.get("project_number"),
                    "project_status": meta.get("project_status"),
                    "logo_bk_uri": meta.get("logo_bk_uri"),
                    "logo_client_uri": meta.get("logo_client_uri"),
                },
            )
        session.commit()


def _list_doc_tasks(engine, project_id: int) -> pd.DataFrame:
    sql = text("""
        SELECT id,
               COALESCE(service_name,'') AS service_name,
               COALESCE(doc_name,'') AS doc_name,
               COALESCE(doc_number,'') AS doc_number,
               COALESCE(revision_code,'') AS revision_code,
               COALESCE(responsible,'') AS responsible,
               COALESCE(responsible_bk,'') AS responsible_bk,
               inicio AS start_date,
               conclusao AS delivery_date,
               COALESCE(status,'Em andamento - BK') AS status,
               COALESCE(observacao,'') AS observation
        FROM project_doc_tasks
        WHERE project_id = :pid
        ORDER BY id DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params={"pid": int(project_id)})
    # normaliza datas
    for c in ["start_date", "delivery_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df


def _insert_event(SessionLocal, project_id: int, doc_task_id: int, status: str, revision_code: str,
                  start_date: Optional[date], delivery_date: Optional[date]) -> None:
    # regra de data do evento: análise/aprovado usa conclusao; BK usa inicio
    k = _status_kind(status)
    if k in ("analise", "aprovado"):
        ev_date = delivery_date or date.today()
    else:
        ev_date = start_date or date.today()
    responsible = _responsible_from_status(status)
    with SessionLocal() as session:
        session.execute(
            text("""
                INSERT INTO doc_status_events (doc_task_id, project_id, event_date, status, responsible, revision_code)
                VALUES (:did, :pid, :ev_date, :status, :resp, :rev)
            """),
            {
                "did": int(doc_task_id),
                "pid": int(project_id),
                "ev_date": ev_date,
                "status": status,
                "resp": responsible,
                "rev": revision_code or None,
            },
        )
        session.commit()


def _upsert_doc_tasks(SessionLocal, engine, project_id: int, edited: pd.DataFrame) -> Tuple[int, int, int]:
    """Salva tabela: insere/atualiza/exclui. Também registra eventos quando status muda.
    Retorna (inserted, updated, deleted).
    """
    inserted = updated = deleted = 0
    project_id = int(project_id)

    # mapa do status anterior para detectar transições
    old_df = _list_doc_tasks(engine, project_id)
    old_map = {int(r["id"]): r for _, r in old_df.iterrows()} if not old_df.empty else {}

    # garante colunas
    for col in ["Excluir", "status", "revision_code", "responsible", "responsible_bk"]:
        if col not in edited.columns:
            edited[col] = "" if col != "Excluir" else False

    with SessionLocal() as session:
        for _, row in edited.iterrows():
            rid = row.get("id")
            rid_int = int(rid) if (pd.notna(rid) and str(rid).strip() != "") else None

            # delete
            if bool(row.get("Excluir")) and rid_int:
                session.execute(
                    text("DELETE FROM project_doc_tasks WHERE id=:id AND project_id=:pid"),
                    {"id": rid_int, "pid": project_id},
                )
                deleted += 1
                continue

            service_name = str(row.get("service_name") or "").strip()
            doc_name = str(row.get("doc_name") or "").strip()
            doc_number = str(row.get("doc_number") or "").strip()
            status = str(row.get("status") or "Em andamento - BK").strip()
            observation = str(row.get("observation") or "").strip()

            start_date = _to_date(row.get("start_date"))
            delivery_date = _to_date(row.get("delivery_date"))
            responsible = str(row.get("responsible") or "").strip() or _responsible_from_status(status)
            responsible_bk = str(row.get("responsible_bk") or "").strip()

            # revisão: regra automática ao voltar para revisão após análise
            rev = str(row.get("revision_code") or "").strip().upper()
            if not rev:
                rev = "R0A"

            if rid_int and rid_int in old_map:
                old_status = str(old_map[rid_int].get("status") or "")
                old_rev = str(old_map[rid_int].get("revision_code") or "").strip().upper() or "R0A"
                if _status_kind(status) == "revisao" and _status_kind(old_status) == "analise":
                    # incrementa se usuário não incrementou
                    if rev == old_rev or not rev:
                        rev = _next_revision(old_rev)

            # insert or update
            if not rid_int:
                # insert
                res = session.execute(
                    text("""
                        INSERT INTO project_doc_tasks
                        (project_id, service_name, doc_name, doc_number, revision_code, responsible, responsible_bk,
                         inicio, conclusao, status, observacao)
                        VALUES (:pid, :service_name, :doc_name, :doc_number, :rev, :responsible, :responsible_bk,
                                :inicio, :conclusao, :status, :obs)
                        RETURNING id
                    """) if "postgres" in (getattr(getattr(engine, "dialect", None), "name", "") or "").lower()
                    else text("""
                        INSERT INTO project_doc_tasks
                        (project_id, service_name, doc_name, doc_number, revision_code, responsible, responsible_bk,
                         inicio, conclusao, status, observacao)
                        VALUES (:pid, :service_name, :doc_name, :doc_number, :rev, :responsible, :responsible_bk,
                                :inicio, :conclusao, :status, :obs)
                    """),
                    {
                        "pid": project_id,
                        "service_name": service_name,
                        "doc_name": doc_name,
                        "doc_number": doc_number,
                        "rev": rev,
                        "responsible": responsible,
                        "responsible_bk": responsible_bk,
                        "inicio": start_date,
                        "conclusao": delivery_date,
                        "status": status,
                        "obs": observation,
                    },
                )
                inserted += 1
                # pega id novo
                new_id = None
                try:
                    new_id = res.scalar()
                except Exception:
                    # sqlite: pega last_insert_rowid
                    try:
                        new_id = session.execute(text("SELECT last_insert_rowid()")).scalar()
                    except Exception:
                        new_id = None
                session.flush()
                session.commit()
                if new_id:
                    _insert_event(SessionLocal, project_id, int(new_id), status, rev, start_date, delivery_date)
                continue

            # update
            session.execute(
                text("""
                    UPDATE project_doc_tasks
                    SET service_name=:service_name,
                        doc_name=:doc_name,
                        doc_number=:doc_number,
                        revision_code=:rev,
                        responsible=:responsible,
                        responsible_bk=:responsible_bk,
                        inicio=:inicio,
                        conclusao=:conclusao,
                        status=:status,
                        observacao=:obs,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=:id AND project_id=:pid
                """),
                {
                    "id": rid_int,
                    "pid": project_id,
                    "service_name": service_name,
                    "doc_name": doc_name,
                    "doc_number": doc_number,
                    "rev": rev,
                    "responsible": responsible,
                    "responsible_bk": responsible_bk,
                    "inicio": start_date,
                    "conclusao": delivery_date,
                    "status": status,
                    "obs": observation,
                },
            )
            updated += 1

            # evento se mudou status (ou se mudou revisão em revisão)
            old = old_map.get(rid_int, {})
            old_status = str(old.get("status") or "")
            old_rev = str(old.get("revision_code") or "").strip().upper()
            if status != old_status or rev != old_rev:
                session.commit()
                _insert_event(SessionLocal, project_id, rid_int, status, rev, start_date, delivery_date)

        session.commit()

    return inserted, updated, deleted



def _compute_metrics(engine, project_id: int) -> pd.DataFrame:
    """Calcula tempos BK x Cliente e número de revisões (somente quando a letra da revisão sobe).

    - dias_BK / dias_CLIENTE: calculados a partir do histórico de eventos.
    - revisoes: conta somente incrementos alfabéticos do código de revisão (ex.: R0A -> R0B = 1).
      O primeiro código encontrado não conta como revisão (é a "base").
    """
    sql = text("""
        SELECT doc_task_id, event_date, status, responsible, COALESCE(revision_code,'') AS revision_code
        FROM doc_status_events
        WHERE project_id = :pid
        ORDER BY doc_task_id, event_date, id
    """)
    with engine.connect() as conn:
        ev = pd.read_sql_query(sql, conn, params={"pid": int(project_id)})

    if ev.empty:
        return pd.DataFrame(columns=["id", "dias_BK", "dias_CLIENTE", "revisoes"])

    ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce").dt.date
    today = date.today()

    def _rev_letter_ord(rev: str) -> Optional[int]:
        r = (rev or "").strip().upper()
        if len(r) >= 3 and r.startswith("R0"):
            ch = r[2]
            if "A" <= ch <= "Z":
                return ord(ch)
        return None

    rows: List[Dict] = []
    for doc_id, g in ev.groupby("doc_task_id"):
        g = g.sort_values(["event_date"])
        dias_bk = 0
        dias_cli = 0

        # revisões: só conta quando a letra sobe (R0A->R0B, R0B->R0C, ...)
        revisoes = 0
        last_letter: Optional[int] = None

        for i in range(len(g)):
            cur = g.iloc[i]
            cur_date = cur["event_date"] or today
            cur_status = str(cur.get("status") or "")
            cur_kind = _status_kind(cur_status)
            cur_resp = str(cur.get("responsible") or "")
            cur_rev = str(cur.get("revision_code") or "")

            cur_letter = _rev_letter_ord(cur_rev)
            if cur_letter is not None:
                if last_letter is None:
                    last_letter = cur_letter
                else:
                    if cur_letter > last_letter:
                        revisoes += 1
                        last_letter = cur_letter
                    elif cur_letter < last_letter:
                        # se voltar, rebaseia (não conta como revisão)
                        last_letter = cur_letter

            # calcula intervalo até próximo evento (ou até hoje, se não aprovado)
            if i < len(g) - 1:
                next_date = g.iloc[i + 1]["event_date"] or today
            else:
                if cur_kind == "aprovado":
                    next_date = cur_date
                else:
                    next_date = today

            delta = (next_date - cur_date).days
            if delta < 0:
                delta = 0

            if cur_resp.upper().startswith("CLI") or cur_kind == "analise":
                dias_cli += delta
            else:
                dias_bk += delta

        rows.append(
            {
                "id": int(doc_id),
                "dias_BK": int(dias_bk),
                "dias_CLIENTE": int(dias_cli),
                "revisoes": int(revisoes),
            }
        )

    return pd.DataFrame(rows)


def _chart_png_b64(df: pd.DataFrame, x: str, y: str, title: str) -> str:
    """Gera gráfico simples (matplotlib) em PNG base64 data-uri."""
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 3.2))
        ax = fig.add_subplot(111)
        ax.bar(df[x].astype(str).tolist(), df[y].astype(float).tolist())
        ax.set_title(title)
        ax.tick_params(axis='x', labelrotation=35)
        fig.tight_layout()
        bio = BytesIO()
        fig.savefig(bio, format="png", dpi=150)
        plt.close(fig)
        b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def _pie_chart_png_b64(labels: List[str], values: List[int], title: str) -> str:
    """Gera gráfico de pizza em PNG base64 (com cores diferentes por status)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        vals = [int(v) for v in values]
        if sum(vals) <= 0:
            return ""

        cmap = cm.get_cmap("tab20", max(len(labels), 3))
        colors = [cmap(i) for i in range(len(labels))]

        fig = plt.figure(figsize=(7.2, 3.6))
        ax = fig.add_subplot(111)
        wedges, texts, autotexts = ax.pie(
            vals,
            labels=None,
            autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
            startangle=90,
            colors=colors,
        )
        ax.axis("equal")
        ax.set_title(title)
        ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        fig.tight_layout()

        bio = BytesIO()
        fig.savefig(bio, format="png", dpi=150)
        plt.close(fig)
        b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def _bk_cliente_bar_png_b64(bk_days: int, client_days: int, title: str) -> str:
    """Gera gráfico de barras (BK x Cliente) com cores azul/vermelho e número em cima."""
    try:
        import matplotlib.pyplot as plt

        labels = ["BK", "Cliente"]
        vals = [int(bk_days), int(client_days)]
        colors = ["blue", "red"]

        fig = plt.figure(figsize=(6.2, 3.4))
        ax = fig.add_subplot(111)
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title)
        ax.set_ylabel("Dias")

        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom", fontsize=10)

        fig.tight_layout()
        bio = BytesIO()
        fig.savefig(bio, format="png", dpi=150)
        plt.close(fig)
        b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""

def _doc_time_grouped_bar_png_b64(dfm: pd.DataFrame, title: str) -> str:
    """Gráfico agrupado por documento: Dias BK (azul) x Dias Cliente (vermelho), com número em cima."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        if dfm is None or dfm.empty:
            return ""

        labels = dfm["doc_label"].astype(str).tolist()
        bk = dfm["dias_BK"].astype(int).tolist()
        cli = dfm["dias_CLIENTE"].astype(int).tolist()

        x = np.arange(len(labels))
        w = 0.38

        fig = plt.figure(figsize=(10, 4.2))
        ax = fig.add_subplot(111)

        bars1 = ax.bar(x - w/2, bk, width=w, color="blue", label="BK")
        bars2 = ax.bar(x + w/2, cli, width=w, color="red", label="Cliente")

        ax.set_title(title)
        ax.set_ylabel("Dias")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.legend()

        # números em cima das barras
        for b in list(bars1) + list(bars2):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h, f"{int(h)}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        bio = BytesIO()
        fig.savefig(bio, format="png", dpi=150)
        plt.close(fig)
        b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def _revisions_bar_png_b64(labels: list[str], values: list[int], title: str) -> str:
    """Gráfico de barras: revisões por documento, com número em cima."""
    try:
        import matplotlib.pyplot as plt

        if not labels or not values:
            return ""

        vals = [int(v) for v in values]

        fig = plt.figure(figsize=(10, 3.6))
        ax = fig.add_subplot(111)
        bars = ax.bar([str(x) for x in labels], vals)
        ax.set_title(title)
        ax.set_ylabel("Qtde de revisões")
        ax.tick_params(axis="x", labelrotation=35)

        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h, f"{int(h)}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        bio = BytesIO()
        fig.savefig(bio, format="png", dpi=150)
        plt.close(fig)
        b64 = base64.b64encode(bio.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""

def _build_report_html(meta: Dict, tasks_df: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    """Monta HTML do relatório (padrão BK: bordas finas cinza claro + gráficos)."""
    cliente = meta.get("client_name") or "Cliente"
    proj_nome = meta.get("project_name") or "Projeto"
    proj_num = meta.get("project_number") or ""
    proj_status = meta.get("project_status") or ""

    logo_bk_uri = meta.get("logo_bk_uri") or ""
    logo_cli_uri = meta.get("logo_client_uri") or ""

    # junta para exibir tempos no relatório
    out = tasks_df.copy()
    if not metrics_df.empty:
        out = out.merge(metrics_df, how="left", left_on="id", right_on="id")
    for c in ["dias_BK", "dias_CLIENTE", "revisoes"]:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(int)

    # gráficos Top 10
    chart_total = chart_cli = chart_bk = ""
    if not out.empty:
        out["dias_total"] = out.get("dias_BK", 0) + out.get("dias_CLIENTE", 0)
        top_total = out.sort_values("dias_total", ascending=False).head(10)
        top_cli = out.sort_values("dias_CLIENTE", ascending=False).head(10)
        top_bk = out.sort_values("dias_BK", ascending=False).head(10)
        if not top_total.empty:
            chart_total = _chart_png_b64(top_total, "doc_name", "dias_total", "Top 10 - Tempo total (dias)")
        if not top_cli.empty:
            chart_cli = _chart_png_b64(top_cli, "doc_name", "dias_CLIENTE", "Top 10 - Análise (Cliente) (dias)")
        if not top_bk.empty:
            chart_bk = _chart_png_b64(top_bk, "doc_name", "dias_BK", "Top 10 - Revisão/Elaboração (BK) (dias)")

    css = """
    <style>
      body { font-family: Arial, Helvetica, sans-serif; color: #111; margin: 24px; }
      .hdr { width: 100%; border: 1px solid #e0e0e0; border-radius: 10px; padding: 14px; }
      .hdr-grid { display: grid; grid-template-columns: 160px 1fr 160px; gap: 10px; align-items: center; }
      .hdr img { max-height: 72px; max-width: 160px; }
      .title { text-align: center; }
      .title h1 { margin: 0; font-size: 18px; }
      .sub { font-size: 12px; color: #333; margin-top: 4px; }
      .sub2 { font-size: 12px; color: #333; margin-top: 2px; }
      .badge { display:inline-block; padding: 2px 8px; border-radius: 10px; border:1px solid #e0e0e0; font-size:11px; color:#333; margin-top:6px; }
      table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 12px; }
      th, td { border: 1px solid #d9d9d9; padding: 6px 8px; }
      th { background: #f5f5f5; }
      .charts { margin-top: 14px; display: grid; grid-template-columns: 1fr; gap: 14px; }
      .chart { border: 1px solid #e0e0e0; border-radius: 10px; padding: 10px; }
      .chart img { width: 100%; height: auto; }
    </style>
    """

    left_logo_html = f'<img src="{logo_bk_uri}" />' if logo_bk_uri else ""
    right_logo_html = f'<img src="{logo_cli_uri}" />' if logo_cli_uri else ""

    hdr = (
        '<div class="hdr"><div class="hdr-grid">'
        f'<div style="text-align:left;">{left_logo_html}</div>'
        '<div class="title">'
        '<h1>BK Engenharia e Tecnologia</h1>'
        f'<div class="sub">Cliente: {cliente}</div>'
        f'<div class="sub2">Projeto: {proj_nome} &nbsp;&nbsp;|&nbsp;&nbsp; Nº: {proj_num}</div>'
        f'<div class="badge">Status: {proj_status or "N/D"}</div>'
        '</div>'
        f'<div style="text-align:right;">{right_logo_html}</div>'
        '</div></div>'
    )

    # tabela no relatório
    out_show = out.copy()

    desired = [
        "id",
        "service_name",
        "doc_name",
        "doc_number",
        "revision_code",
        "responsible",
        "responsible_bk",
        "start_date",
        "delivery_date",
        "status",
        "observation",
        # opcional: métricas calculadas (se quiser no relatório)
       # "dias_BK",
        #"dias_CLIENTE",
        # "revisoes",
    ]

    cols = [c for c in desired if c in out_show.columns]
    out_show = out_show[cols]

    out_show = out_show.rename(columns={
        "id": "ID",
        "service_name": "SERVIÇO",
        "doc_name": "NOME DO DOCUMENTO",
        "doc_number": "NUMERO DO DOCUMENTO",
        "revision_code": "REVISÃO",
        "responsible": "RESPONSÁVEL",
        "responsible_bk": "RESPONSÁVEL BK",
        "start_date": "DATA DE INÍCIO",
        "delivery_date": "DATA DE CONCLUSÃO",
        "status": "STATUS",
        "observation": "OBSERVAÇÃO",
        "dias_BK": "DIAS BK",
        "dias_CLIENTE": "DIAS CLIENTE",
        "revisoes": "REVISÕES",
    })

    html = ['<html><head><meta charset="utf-8"/>', css, '</head><body>']
    html.append(hdr)
    html.append('<h2>Tarefas / Documentos</h2>')
    html.append(out_show.to_html(index=False, escape=False))

    html.append('<div class="charts">')
    for uri in [chart_total, chart_cli, chart_bk]:
        if uri:
            html.append('<div class="chart">')
            html.append(f'<img src="{uri}" />')
            html.append('</div>')
    html.append('</div>')
    html.append('</body></html>')
    return "\n".join(html)
def _build_manager_report_html(meta: Dict, tasks_df: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    """Relatório gerencial:
    - Tabela: quantidade de documentos por status (cabeçalho = status; linha abaixo = quantidade).
    - Gráfico de pizza: percentual por status (cores diferentes).
    - Gráfico de barras: total de dias com BK vs Cliente (BK azul; Cliente vermelho; valor em cima).
    - Tabela: quantas vezes o documento foi revisado (somente quando letra sobe).
    - Botão para impressão (window.print).
    """
    cliente = meta.get("client_name") or "Cliente"
    proj_nome = meta.get("project_name") or "Projeto"
    proj_num = meta.get("project_number") or ""
    proj_status = meta.get("project_status") or ""

    logo_bk_uri = meta.get("logo_bk_uri") or ""
    logo_cli_uri = meta.get("logo_client_uri") or ""

    # status counts (preserva ordem conhecida, mas inclui outros se existirem)
    counts = {}
    if tasks_df is not None and not tasks_df.empty and "status" in tasks_df.columns:
        vc = tasks_df["status"].fillna("").astype(str).value_counts()
        for s in STATUS_OPTIONS:
            if s in vc.index:
                counts[s] = int(vc.loc[s])
        # statuses extras (se houver)
        for s in vc.index.tolist():
            if s and s not in counts:
                counts[s] = int(vc.loc[s])

    labels = list(counts.keys())
    values = list(counts.values())

    status_table_html = ""
    if labels:
        df_counts = pd.DataFrame([values], columns=labels)
        status_table_html = df_counts.to_html(index=False, escape=False)

    # gráficos
    pie_uri = _pie_chart_png_b64(labels, values, "Percentual de documentos por status") if labels else ""

    bk_total = int(metrics_df["dias_BK"].sum()) if metrics_df is not None and not metrics_df.empty and "dias_BK" in metrics_df.columns else 0
    cli_total = int(metrics_df["dias_CLIENTE"].sum()) if metrics_df is not None and not metrics_df.empty and "dias_CLIENTE" in metrics_df.columns else 0
    bk_cli_uri = _bk_cliente_bar_png_b64(bk_total, cli_total, "Total de dias com BK x Cliente")

    # gráfico: dias por documento (BK x Cliente)
    doc_time_uri = ""
    rev_chart_uri = ""
    if tasks_df is not None and not tasks_df.empty and metrics_df is not None and not metrics_df.empty:
        dfm = tasks_df.merge(metrics_df, on="id", how="left")
        # rótulo curto do documento
        if "doc_number" in dfm.columns and dfm["doc_number"].astype(str).str.strip().ne("").any():
            dfm["doc_label"] = dfm["doc_number"].fillna("").astype(str)
        else:
            dfm["doc_label"] = dfm.get("doc_name", "").fillna("").astype(str)

        dfm["dias_BK"] = pd.to_numeric(dfm.get("dias_BK", 0), errors="coerce").fillna(0).astype(int)
        dfm["dias_CLIENTE"] = pd.to_numeric(dfm.get("dias_CLIENTE", 0), errors="coerce").fillna(0).astype(int)
        dfm["revisoes"] = pd.to_numeric(dfm.get("revisoes", 0), errors="coerce").fillna(0).astype(int)

        # limita para não estourar largura no HTML
        dfm = dfm.sort_values(["dias_CLIENTE", "dias_BK"], ascending=False).head(20)
        doc_time_uri = _doc_time_grouped_bar_png_b64(dfm, "Dias por documento (BK x Cliente)")
        rev_chart_uri = _revisions_bar_png_b64(dfm["doc_label"].tolist(), dfm["revisoes"].tolist(), "Revisões por documento")

    # tabela de revisões por documento
    rev_table_html = "<p>Sem dados.</p>"
    if tasks_df is not None and not tasks_df.empty:
        df = tasks_df.copy()
        if metrics_df is not None and not metrics_df.empty:
            df = df.merge(metrics_df[["id", "revisoes"]], on="id", how="left")
        df["revisoes"] = df.get("revisoes", 0)
        df["revisoes"] = pd.to_numeric(df["revisoes"], errors="coerce").fillna(0).astype(int)

        show_cols = []
        for c in ["doc_name", "doc_number", "revision_code", "status", "revisoes"]:
            if c in df.columns:
                show_cols.append(c)
        if show_cols:
            df2 = df[show_cols].copy().rename(
                columns={
                    "doc_name": "NOME DO DOCUMENTO",
                    "doc_number": "NUMERO DO DOCUMENTO",
                    "revision_code": "REVISÃO ATUAL",
                    "status": "STATUS",
                    "revisoes": "QTDE DE REVISÕES",
                }
            )
            df2 = df2.sort_values("QTDE DE REVISÕES", ascending=False)
            rev_table_html = df2.to_html(index=False, escape=False)

    css = """
    <style>
      body { font-family: Arial, Helvetica, sans-serif; color: #111; margin: 24px; }
      .hdr { width: 100%; border: 1px solid #e0e0e0; border-radius: 10px; padding: 14px; }
      .hdr-grid { display: grid; grid-template-columns: 160px 1fr 160px; gap: 10px; align-items: center; }
      .hdr img { max-height: 72px; max-width: 160px; }
      .title { text-align: center; }
      .title h1 { margin: 0; font-size: 18px; }
      .sub { font-size: 12px; color: #333; margin-top: 4px; }
      .sub2 { font-size: 12px; color: #333; margin-top: 2px; }
      .badge { display:inline-block; padding: 2px 8px; border-radius: 10px; border:1px solid #e0e0e0; font-size:11px; color:#333; margin-top:6px; }
      .btnbar { margin: 12px 0 0 0; display:flex; justify-content:flex-end; }
      .printbtn { cursor:pointer; border: 1px solid #d9d9d9; padding: 8px 12px; border-radius: 10px; background:#fff; font-size: 12px; }
      table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 12px; }
      th, td { border: 1px solid #d9d9d9; padding: 6px 8px; }
      th { background: #f5f5f5; }
      .charts { margin-top: 14px; display: grid; grid-template-columns: 1fr; gap: 14px; }
      .chart { border: 1px solid #e0e0e0; border-radius: 10px; padding: 10px; }
      .chart img { width: 100%; height: auto; }
      @media print {
        .printbtn, .btnbar { display: none !important; }
        body { margin: 0; }
      }
    </style>
    """

    left_logo_html = f'<img src="{logo_bk_uri}" />' if logo_bk_uri else ""
    right_logo_html = f'<img src="{logo_cli_uri}" />' if logo_cli_uri else ""

    hdr = (
        '<div class="hdr"><div class="hdr-grid">'
        f'<div style="text-align:left;">{left_logo_html}</div>'
        '<div class="title">'
        '<h1>BK Engenharia e Tecnologia</h1>'
        f'<div class="sub">Cliente: {cliente}</div>'
        f'<div class="sub2">Projeto: {proj_nome} &nbsp;&nbsp;|&nbsp;&nbsp; Nº: {proj_num}</div>'
        f'<div class="badge">Status: {proj_status or "N/D"}</div>'
        '</div>'
        f'<div style="text-align:right;">{right_logo_html}</div>'
        '</div>'
        '<div class="btnbar"><button class="printbtn" onclick="window.print()">🖨️ Imprimir</button></div>'
        '</div>'
    )

    html = ['<html><head><meta charset="utf-8"/>', css, '</head><body>']
    html.append(hdr)

    html.append('<h2>Relatório gerencial</h2>')
    html.append('<h3>Quantidade de documentos por status</h3>')
    html.append(status_table_html or "<p>Sem documentos para contabilizar.</p>")

    html.append('<div class="charts">')
    if pie_uri:
        html.append('<div class="chart">')
        html.append(f'<img src="{pie_uri}" />')
        html.append('</div>')
    if bk_cli_uri:
        html.append('<div class="chart">')
        html.append(f'<img src="{bk_cli_uri}" />')
        html.append('</div>')
    if doc_time_uri:
        html.append('<div class="chart">')
        html.append(f'<img src="{doc_time_uri}" />')
        html.append('</div>')
    if rev_chart_uri:
        html.append('<div class="chart">')
        html.append(f'<img src="{rev_chart_uri}" />')
        html.append('</div>')
    html.append('</div>')

    html.append('<h3>Revisões por documento</h3>')
    html.append(rev_table_html)

    html.append('</body></html>')
    return "\n".join(html)




# ------------------------------
# UI
# ------------------------------

def render_controle_projetos(engine, project_id: Optional[int] = None) -> None:
    """Renderiza o Controle de Projetos dentro de outra página (ex.: Gestão de Projetos).

    - Se project_id for informado: usa esse projeto e não mostra seletor.
    - Se project_id for None: mostra seletor (modo standalone).
    """
    ensure_erp_tables()
    _ensure_doc_tasks_schema(engine)
    apply_theme()

    projects = _list_projects(engine)
    if projects.empty:
        st.info("Nenhum projeto cadastrado em Gestão de Projetos.")
        return

    pid: int
    if project_id is None:
        # seletor de projeto (modo standalone)
        proj_opts = {f"#{int(r.id)} - {r.nome}": int(r.id) for r in projects.itertuples()}
        sel_label = st.selectbox("Selecione o projeto", list(proj_opts.keys()))
        pid = int(proj_opts[sel_label])
    else:
        pid = int(project_id)
        if pid not in set(projects["id"].astype(int).tolist()):
            st.error(f"Projeto #{pid} não encontrado.")
            return

    # Defaults vindo do cadastro de projeto
    proj_row = projects[projects["id"].astype(int) == pid].iloc[0].to_dict()
    defaults = {
        "client_name": proj_row.get("client_name") or proj_row.get("cliente") or "Cliente",
        "project_name": proj_row.get("nome") or f"Projeto {pid}",
        "project_number": proj_row.get("cod_projeto") or proj_row.get("project_code") or str(pid),
        "project_status": proj_row.get("status") or "",
        "logo_bk_uri": "",
        "logo_client_uri": "",
    }

    # carrega meta persistida (se existir)
    meta_db = _load_meta(engine, pid) or {}
    meta = {**defaults, **meta_db}

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("📄 Informações do relatório")
        st.caption("Essas informações são usadas no cabeçalho do relatório e ficam salvas por projeto.")

        with st.form("form_meta", clear_on_submit=False):
            client_name = st.text_input("Cliente", value=str(meta.get("client_name") or ""))
            project_name = st.text_input("Nome do projeto", value=str(meta.get("project_name") or ""))
            project_number = st.text_input("Nº do projeto", value=str(meta.get("project_number") or ""))
            status_options = ["", "Em andamento", "Em atraso", "Concluído", "Pausado"]
            saved_status = str(meta.get("project_status") or "").strip()
            try:
                status_idx = status_options.index(saved_status) if saved_status in status_options else 0
            except Exception:
                status_idx = 0
            project_status = st.selectbox("Status do projeto", status_options, index=status_idx)

            c1, c2 = st.columns(2)
            with c1:
                logo_bk_up = st.file_uploader("Logo BK (opcional)", type=["png", "jpg", "jpeg", "svg"])
            with c2:
                logo_cli_up = st.file_uploader("Logo Cliente (opcional)", type=["png", "jpg", "jpeg", "svg"])

            save_meta = st.form_submit_button("💾 Salvar informações do relatório")

        if save_meta:
            meta_new = {
                "client_name": client_name.strip(),
                "project_name": project_name.strip(),
                "project_number": project_number.strip(),
                "project_status": str(project_status or "").strip(),
                "logo_bk_uri": _file_to_data_uri(logo_bk_up) or meta.get("logo_bk_uri") or "",
                "logo_client_uri": _file_to_data_uri(logo_cli_up) or meta.get("logo_client_uri") or "",
            }
            _upsert_meta(SessionLocal, pid, meta_new)
            st.success("Informações do relatório salvas.")
            st.rerun()

        st.divider()

        # Relatório (sempre visível)
        tasks_df = _list_doc_tasks(engine, pid)
        metrics_df = _compute_metrics(engine, pid)

        meta = _load_meta(engine, pid) or meta  # recarrega
        html_report = _build_report_html(meta, tasks_df, metrics_df)
        html_manager_report = _build_manager_report_html(meta, tasks_df, metrics_df)

        st.download_button(
            "⬇️ Baixar relatório (HTML)",
            data=html_report.encode("utf-8"),
            file_name=f"controle_projetos_{pid}.html",
            mime="text/html",
            width='stretch',
        )
        
        # Relatório filtrado: somente documentos em "Em análise - Cliente"
        tasks_df_cli = (
            tasks_df[tasks_df["status"].fillna("").astype(str).eq("Em análise - Cliente")].copy()
            if tasks_df is not None and not tasks_df.empty and "status" in tasks_df.columns
            else pd.DataFrame()
        )
        metrics_cli = metrics_df.copy() if metrics_df is not None else pd.DataFrame()
        if not metrics_cli.empty and not tasks_df_cli.empty and "id" in tasks_df_cli.columns:
            _ids = pd.to_numeric(tasks_df_cli["id"], errors="coerce").dropna().astype(int).tolist()
            metrics_cli = metrics_cli[metrics_cli["id"].isin(_ids)]
        html_report_cli = _build_report_html(meta, tasks_df_cli, metrics_cli)

        st.download_button(
            "⬇️ Baixar relatório (Em análise - Cliente) (HTML)",
            data=html_report_cli.encode("utf-8"),
            file_name=f"controle_projetos_em_analise_cliente_{pid}.html",
            mime="text/html",
            width='stretch',
        )
        st.download_button(
            "⬇️ Baixar relatório gerencial (HTML)",
            data=html_manager_report.encode("utf-8"),
            file_name=f"controle_projetos_gerencial_{pid}.html",
            mime="text/html",
            width='stretch',
        )

        with st.expander("👁️ Pré-visualizar relatório gerencial", expanded=False):
            st.components.v1.html(html_manager_report, height=700, scrolling=True)

        # também exporta CSV (tabela + tempos)
        export_df = tasks_df.copy()
        if not metrics_df.empty and not tasks_df.empty:
            export_df = export_df.merge(metrics_df, on="id", how="left")
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar CSV (tabela + tempos)",
            data=csv_bytes,
            file_name=f"controle_projetos_{pid}.csv",
            mime="text/csv",
            width='stretch',
        )

        with st.expander("👁️ Pré-visualizar relatório", expanded=False):
            st.components.v1.html(html_report, height=700, scrolling=True)

    with right:
        st.subheader("🧾 Tabela (estilo Excel)")
        st.caption("Preencha/edite direto na tabela e clique em **Salvar tabela**. Use **Excluir** para remover linhas.")

        services_df = _list_services(engine)
        service_names = services_df["name"].astype(str).tolist() if not services_df.empty else []

        tasks_df = _list_doc_tasks(engine, pid)
        if tasks_df.empty:
            tasks_df = pd.DataFrame(columns=[
                "id","service_name","doc_name","doc_number","revision_code","responsible",
                "start_date","delivery_date","status","observation"
            ])

        # garante colunas
        if "Excluir" not in tasks_df.columns:
            tasks_df["Excluir"] = False

        # botão para adicionar linha vazia (deixa claro onde informar serviço)
        if st.button("➕ Nova linha", type="secondary", width='stretch'):
            new_row = {
                "id": None,
                "service_name": "",
                "doc_name": "",
                "doc_number": "",
                "revision_code": "R0A",
                "responsible": "BK",
                "start_date": date.today(),
                "delivery_date": date.today(),
                "status": "Em andamento - BK",
                "observation": "",
                "Excluir": False,
            }
            tasks_df = pd.concat([pd.DataFrame([new_row]), tasks_df], ignore_index=True)

        # Column configs
        col_cfg = {
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "service_name": st.column_config.SelectboxColumn(
                "Serviço",
                options=service_names,
                help="Vem do cadastro de Serviços (digite para pesquisar).",
                required=True,
            ),
            "doc_name": st.column_config.TextColumn("Nome do documento"),
            "doc_number": st.column_config.TextColumn("Nº do documento"),
            "revision_code": st.column_config.TextColumn("Revisão", help="R0A, R0B... Incrementa automático ao voltar para revisão."),
            "responsible": st.column_config.SelectboxColumn("Responsável", options=RESPONSIBLE_OPTIONS),
            "responsible_bk": st.column_config.TextColumn("Responsável BK", help="Responsável da BK pelo documento"),
            "start_date": st.column_config.DateColumn("Data de início"),
            "delivery_date": st.column_config.DateColumn("Data de conclusão"),
            "status": st.column_config.SelectboxColumn("Status", options=STATUS_OPTIONS),
            "observation": st.column_config.TextColumn("Observação"),
            "Excluir": st.column_config.CheckboxColumn("Excluir", help="Marque e clique em Salvar tabela."),
        }

        edited = st.data_editor(
            tasks_df,
            column_config=col_cfg,
            hide_index=True,
            num_rows="dynamic",
            width='stretch',
            key=f"doc_editor_{pid}",
        )

        csave1, csave2 = st.columns([1, 1])
        with csave1:
            if st.button("💾 Salvar tabela", type="primary", width='stretch'):
                ins, upd, dele = _upsert_doc_tasks(SessionLocal, engine, pid, edited)
                st.success(f"Salvo. Inseridos: {ins} | Atualizados: {upd} | Excluídos: {dele}")
                st.rerun()
        with csave2:
            if st.button("🔄 Recarregar", width='stretch'):
                st.rerun()

        st.divider()

        st.subheader("📊 Indicadores (BK x Cliente)")
        metrics_df = _compute_metrics(engine, pid)
        if metrics_df.empty:
            st.info("Ainda não há eventos de status suficientes para calcular tempos. Salve a tabela para gerar histórico.")
        else:
            merged = _list_doc_tasks(engine, pid).merge(metrics_df, on="id", how="left")
            merged["dias_BK"] = merged["dias_BK"].fillna(0).astype(int)
            merged["dias_CLIENTE"] = merged["dias_CLIENTE"].fillna(0).astype(int)
            merged["revisoes"] = merged["revisoes"].fillna(0).astype(int)

            st.dataframe(
                merged[["id","doc_name","doc_number","revision_code","status","dias_BK","dias_CLIENTE","revisoes"]],
                width='stretch',
                hide_index=True,
            )

            merged["dias_total"] = merged["dias_BK"] + merged["dias_CLIENTE"]
            top_total = merged.sort_values("dias_total", ascending=False).head(10)
            top_cli = merged.sort_values("dias_CLIENTE", ascending=False).head(10)
            top_bk = merged.sort_values("dias_BK", ascending=False).head(10)

            if not top_total.empty:
                st.bar_chart(top_total.set_index("doc_name")["dias_total"], width='stretch')
            if not top_cli.empty:
                st.bar_chart(top_cli.set_index("doc_name")["dias_CLIENTE"], width='stretch')
            if not top_bk.empty:
                st.bar_chart(top_bk.set_index("doc_name")["dias_BK"], width='stretch')



def main() -> None:
    ensure_erp_tables()
    engine, SessionLocal = get_finance_db()
    login_and_guard(SessionLocal)
    apply_theme()

    _ensure_control_tables(engine, SessionLocal)

    st.markdown(
        '<div class="bk-card"><div class="bk-title">🖥️ Controle de Projetos</div>'
        '<p class="bk-subtitle">Tabela estilo Excel + relatório com tempos BK x Cliente (múltiplas análises/revisões).</p></div>',
        unsafe_allow_html=True,
    )

    # renderização (modo standalone)
    render_controle_projetos(engine)
if __name__ == "__main__":
    main()
