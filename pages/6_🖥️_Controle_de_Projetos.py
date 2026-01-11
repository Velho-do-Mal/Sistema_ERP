# pages/6_🖥️_Controle_de_Projetos.py
# -*- coding: utf-8 -*-

import base64
from datetime import date, datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import text

from bk_erp_shared.auth import login_and_guard
from bk_erp_shared.erp_db import ensure_erp_tables, get_finance_db


# ------------------------------
# Helpers
# ------------------------------
def _to_date(v) -> Optional[date]:
    """Aceita date/datetime/ISO-string e devolve date (ou None)."""
    if v is None or v == "":
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return date.fromisoformat(s[:10])
        except Exception:
            return None
    return None


def _b64_of_upload(file) -> Optional[str]:
    if not file:
        return None
    data = file.getvalue()
    if not data:
        return None
    return base64.b64encode(data).decode("utf-8")


def _actor_from_status(status: str) -> str:
    s = (status or "").strip().lower()
    if "cliente" in s or "análise" in s or "analise" in s or "aprov" in s:
        return "CLIENTE"
    return "BK"


def ensure_controle_tables(engine):
    """Cria tabelas do módulo Controle de Projetos (docs + metadados)."""
    with engine.begin() as conn:
        # metadados do relatório (nome/cliente/código + logos)
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS project_control_meta (
                    project_id INTEGER PRIMARY KEY,
                    client_name TEXT,
                    project_code TEXT,
                    project_name TEXT,
                    bk_logo_b64 TEXT,
                    client_logo_b64 TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
        )

        # tabela principal (documentos/tarefas)
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS project_doc_tasks (
                    id SERIAL PRIMARY KEY,
                    project_id INTEGER NOT NULL,
                    service_name TEXT,
                    doc_name TEXT,
                    doc_number TEXT,
                    revision TEXT DEFAULT 'R0A',
                    status TEXT DEFAULT 'Em andamento - BK',
                    responsible TEXT DEFAULT 'BK',
                    start_date DATE,
                    completion_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
        )

        # histórico de status para somar tempos BK x Cliente
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS project_doc_events (
                    id SERIAL PRIMARY KEY,
                    doc_task_id INTEGER NOT NULL,
                    project_id INTEGER NOT NULL,
                    status TEXT,
                    revision TEXT,
                    actor TEXT,
                    started_at TIMESTAMP NOT NULL,
                    ended_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
        )


def _list_projects(engine) -> pd.DataFrame:
    """Lista projetos de forma tolerante a diferentes esquemas."""
    sql = text(
        """
        SELECT id,
               COALESCE(nome, name, project_name, '') AS nome,
               COALESCE(status, project_status, '') AS status
        FROM projects
        ORDER BY id DESC
        """
    )
    try:
        return pd.read_sql(sql, engine)
    except Exception:
        # fallback mínimo
        try:
            return pd.read_sql(text("SELECT id FROM projects ORDER BY id DESC"), engine)
        except Exception:
            return pd.DataFrame(columns=["id", "nome", "status"])


def _load_project_row(engine, pid: int) -> Dict:
    """Carrega linha do projeto (inclui JSON 'data' se existir)."""
    # tenta várias colunas
    for sql in [
        text("SELECT * FROM projects WHERE id=:pid"),
    ]:
        try:
            df = pd.read_sql(sql, engine, params={"pid": int(pid)})
            if df.empty:
                return {}
            row = dict(df.iloc[0].to_dict())
            # tenta decodificar campo data/json
            data = row.get("data")
            if isinstance(data, str) and data.strip().startswith("{"):
                try:
                    import json
                    row["data"] = json.loads(data)
                except Exception:
                    pass
            return row
        except Exception:
            continue
    return {}


def _save_project_basic(engine, pid: int, status: str, progress: int, planned: Optional[date], actual: Optional[date], delay_resp: str):
    """Salva campos básicos no projeto. Nem todos os bancos têm colunas -> usa JSON 'data' se necessário."""
    row = _load_project_row(engine, pid)
    # 1) tenta colunas diretas
    with engine.begin() as conn:
        try:
            conn.execute(
                text(
                    """
                    UPDATE projects
                    SET status=:status,
                        progress=:progress,
                        entrega_prevista=:planned,
                        entrega_real=:actual,
                        delay_responsibility=:delay_resp
                    WHERE id=:pid
                    """
                ),
                {
                    "pid": int(pid),
                    "status": status,
                    "progress": int(progress),
                    "planned": planned,
                    "actual": actual,
                    "delay_resp": delay_resp,
                },
            )
            return
        except Exception:
            pass

    # 2) fallback: salva no JSON 'data' (se existir)
    data = row.get("data") if isinstance(row.get("data"), dict) else {}
    data = dict(data or {})
    data.update(
        {
            "status": status,
            "progress": int(progress),
            "entrega_prevista": planned.isoformat() if planned else None,
            "entrega_real": actual.isoformat() if actual else None,
            "delay_responsibility": delay_resp,
        }
    )
    import json

    with engine.begin() as conn:
        try:
            conn.execute(
                text("UPDATE projects SET data=:data WHERE id=:pid"),
                {"pid": int(pid), "data": json.dumps(data, ensure_ascii=False)},
            )
        except Exception:
            # se nada funcionar, ignora
            pass


def _load_meta(engine, pid: int) -> Dict:
    try:
        df = pd.read_sql(text("SELECT * FROM project_control_meta WHERE project_id=:pid"), engine, params={"pid": int(pid)})
        return dict(df.iloc[0].to_dict()) if not df.empty else {}
    except Exception:
        return {}


def _save_meta(engine, pid: int, client_name: str, project_name: str, project_code: str, bk_logo_b64: Optional[str], client_logo_b64: Optional[str]):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO project_control_meta (project_id, client_name, project_name, project_code, bk_logo_b64, client_logo_b64, updated_at)
                VALUES (:pid, :client, :pname, :pcode, :bk_logo, :client_logo, CURRENT_TIMESTAMP)
                ON CONFLICT (project_id)
                DO UPDATE SET
                    client_name=EXCLUDED.client_name,
                    project_name=EXCLUDED.project_name,
                    project_code=EXCLUDED.project_code,
                    bk_logo_b64=COALESCE(EXCLUDED.bk_logo_b64, project_control_meta.bk_logo_b64),
                    client_logo_b64=COALESCE(EXCLUDED.client_logo_b64, project_control_meta.client_logo_b64),
                    updated_at=CURRENT_TIMESTAMP
                """
            ),
            {
                "pid": int(pid),
                "client": client_name,
                "pname": project_name,
                "pcode": project_code,
                "bk_logo": bk_logo_b64,
                "client_logo": client_logo_b64,
            },
        )


def _list_services(engine) -> List[str]:
    try:
        df = pd.read_sql(
            text(
                """
                SELECT name
                FROM product_services
                WHERE COALESCE(active, TRUE) = TRUE
                ORDER BY name
                """
            ),
            engine,
        )
        return [str(x) for x in df["name"].dropna().tolist()]
    except Exception:
        return []


def _list_doc_tasks(engine, pid: int) -> pd.DataFrame:
    sql = text(
        """
        SELECT id,
               COALESCE(service_name,'') AS service_name,
               COALESCE(doc_name,'') AS doc_name,
               COALESCE(doc_number,'') AS doc_number,
               COALESCE(revision,'R0A') AS revision,
               COALESCE(status,'Em andamento - BK') AS status,
               COALESCE(responsible,'BK') AS responsible,
               start_date,
               completion_date
        FROM project_doc_tasks
        WHERE project_id = :pid
        ORDER BY id ASC
        """
    )
    try:
        return pd.read_sql(sql, engine, params={"pid": int(pid)})
    except Exception:
        return pd.DataFrame(
            columns=[
                "id",
                "service_name",
                "doc_name",
                "doc_number",
                "revision",
                "status",
                "responsible",
                "start_date",
                "completion_date",
            ]
        )


def _insert_doc_task(engine, pid: int, row: Dict) -> int:
    now = datetime.utcnow()
    actor = _actor_from_status(row.get("status") or "")
    responsible = (row.get("responsible") or actor).upper()
    start_date = _to_date(row.get("start_date")) or date.today()
    completion_date = _to_date(row.get("completion_date"))
    with engine.begin() as conn:
        res = conn.execute(
            text(
                """
                INSERT INTO project_doc_tasks
                    (project_id, service_name, doc_name, doc_number, revision, status, responsible, start_date, completion_date, updated_at)
                VALUES
                    (:pid, :service, :dname, :dnum, :rev, :status, :resp, :sd, :cd, CURRENT_TIMESTAMP)
                RETURNING id
                """
            ),
            {
                "pid": int(pid),
                "service": (row.get("service_name") or "").strip(),
                "dname": (row.get("doc_name") or "").strip(),
                "dnum": (row.get("doc_number") or "").strip(),
                "rev": (row.get("revision") or "R0A").strip(),
                "status": (row.get("status") or "Em andamento - BK").strip(),
                "resp": responsible,
                "sd": start_date,
                "cd": completion_date,
            },
        )
        new_id = int(res.scalar() or 0)

        # cria evento inicial
        conn.execute(
            text(
                """
                INSERT INTO project_doc_events (doc_task_id, project_id, status, revision, actor, started_at)
                VALUES (:doc_id, :pid, :status, :rev, :actor, :started_at)
                """
            ),
            {
                "doc_id": new_id,
                "pid": int(pid),
                "status": (row.get("status") or "Em andamento - BK").strip(),
                "rev": (row.get("revision") or "R0A").strip(),
                "actor": actor,
                "started_at": now,
            },
        )
        return new_id


def _transition_event(engine, pid: int, doc_id: int, new_status: str, new_revision: str):
    """Fecha evento aberto e abre novo."""
    now = datetime.utcnow()
    actor = _actor_from_status(new_status)

    with engine.begin() as conn:
        # fecha evento aberto
        conn.execute(
            text(
                """
                UPDATE project_doc_events
                   SET ended_at=:now
                 WHERE project_id=:pid
                   AND doc_task_id=:doc_id
                   AND ended_at IS NULL
                """
            ),
            {"now": now, "pid": int(pid), "doc_id": int(doc_id)},
        )
        # abre novo
        conn.execute(
            text(
                """
                INSERT INTO project_doc_events (doc_task_id, project_id, status, revision, actor, started_at)
                VALUES (:doc_id, :pid, :status, :rev, :actor, :started_at)
                """
            ),
            {
                "doc_id": int(doc_id),
                "pid": int(pid),
                "status": (new_status or "").strip(),
                "rev": (new_revision or "R0A").strip(),
                "actor": actor,
                "started_at": now,
            },
        )


def _update_doc_task(engine, pid: int, doc_id: int, row: Dict, original: Dict):
    # detecta mudança de status/revisão para log
    new_status = (row.get("status") or "Em andamento - BK").strip()
    new_rev = (row.get("revision") or "R0A").strip()
    old_status = (original.get("status") or "").strip()
    old_rev = (original.get("revision") or "").strip()

    actor = _actor_from_status(new_status)
    responsible = (row.get("responsible") or actor).upper()
    start_date = _to_date(row.get("start_date")) or _to_date(original.get("start_date")) or date.today()
    completion_date = _to_date(row.get("completion_date"))

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE project_doc_tasks
                   SET service_name=:service,
                       doc_name=:dname,
                       doc_number=:dnum,
                       revision=:rev,
                       status=:status,
                       responsible=:resp,
                       start_date=:sd,
                       completion_date=:cd,
                       updated_at=CURRENT_TIMESTAMP
                 WHERE id=:id AND project_id=:pid
                """
            ),
            {
                "id": int(doc_id),
                "pid": int(pid),
                "service": (row.get("service_name") or "").strip(),
                "dname": (row.get("doc_name") or "").strip(),
                "dnum": (row.get("doc_number") or "").strip(),
                "rev": new_rev,
                "status": new_status,
                "resp": responsible,
                "sd": start_date,
                "cd": completion_date,
            },
        )

    if (new_status != old_status) or (new_rev != old_rev):
        _transition_event(engine, pid, doc_id, new_status, new_rev)


def _delete_doc_task(engine, pid: int, doc_id: int):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM project_doc_tasks WHERE id=:id AND project_id=:pid"), {"id": int(doc_id), "pid": int(pid)})
        conn.execute(text("DELETE FROM project_doc_events WHERE doc_task_id=:id AND project_id=:pid"), {"id": int(doc_id), "pid": int(pid)})


def _compute_times(engine, pid: int) -> pd.DataFrame:
    """Retorna tempos por documento e totais BK x CLIENTE (dias)."""
    try:
        ev = pd.read_sql(
            text(
                """
                SELECT doc_task_id, actor, started_at, ended_at
                FROM project_doc_events
                WHERE project_id=:pid
                ORDER BY doc_task_id, started_at
                """
            ),
            engine,
            params={"pid": int(pid)},
        )
        if ev.empty:
            return pd.DataFrame(columns=["doc_task_id", "actor", "days"])

        now = datetime.utcnow()
        ev["ended_at"] = ev["ended_at"].fillna(now)
        ev["started_at"] = pd.to_datetime(ev["started_at"])
        ev["ended_at"] = pd.to_datetime(ev["ended_at"])
        ev["days"] = (ev["ended_at"] - ev["started_at"]).dt.total_seconds() / 86400.0
        out = ev.groupby(["doc_task_id", "actor"], as_index=False)["days"].sum()
        return out
    except Exception:
        return pd.DataFrame(columns=["doc_task_id", "actor", "days"])


# ------------------------------
# UI
# ------------------------------
def main():
    login_and_guard()
    st.title("🖥️ Controle de Projetos")
    st.caption("Acompanhe documentos/tarefas (BK x Cliente) com tabela estilo Excel e relatório.")

    engine, SessionLocal = get_finance_db()
    ensure_erp_tables(engine)
    ensure_controle_tables(engine)

    projects = _list_projects(engine)
    if projects.empty:
        st.info("Nenhum projeto cadastrado ainda.")
        return

    # Seleção do projeto
    proj_label = projects.apply(lambda r: f"#{int(r['id'])} - {str(r.get('nome') or '').strip()} ({str(r.get('status') or '').strip()})", axis=1)
    opt_map = dict(zip(proj_label.tolist(), projects["id"].tolist()))
    selected = st.selectbox("Projetos", list(opt_map.keys()), index=0)
    pid = int(opt_map[selected])

    # Carrega linhas
    proj_row = _load_project_row(engine, pid) or {}
    meta = _load_meta(engine, pid) or {}

    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.subheader("Projeto")

        status_options = ["em_elaboracao", "em_analise", "em_revisao", "em_aprovacao", "concluido"]
        cur_status = str(proj_row.get("status") or proj_row.get("project_status") or meta.get("status") or "em_aprovacao")
        if cur_status not in status_options:
            status_options = [cur_status] + [s for s in status_options if s != cur_status]
        status = st.selectbox("Status", status_options, index=status_options.index(cur_status) if cur_status in status_options else 0)

        progress = st.slider("Progresso (%)", 0, 100, int(proj_row.get("progress") or proj_row.get("progresso") or 0))

        planned_default = _to_date(proj_row.get("entrega_prevista")) or date.today()
        actual_default = _to_date(proj_row.get("entrega_real"))

        planned = st.date_input("Entrega prevista", value=planned_default)
        actual = st.date_input("Entrega real (se concluído)", value=actual_default or planned_default)

        resp = st.selectbox("Responsável pelo atraso", ["N/A", "CLIENTE", "BK"], index=0)
        if st.button("Salvar projeto", type="primary", width="stretch"):
            _save_project_basic(engine, pid, status=status, progress=progress, planned=_to_date(planned), actual=_to_date(actual), delay_resp=resp)
            st.success("Projeto atualizado.")

    with right:
        st.subheader("Tarefas / Documentos (tabela estilo Excel)")

        # --- Formulário (informações do relatório) ---
        with st.expander("Informações do relatório (cliente / projeto / logos)", expanded=True):
            client_name = st.text_input("Cliente", value=str(meta.get("client_name") or ""))
            project_name = st.text_input("Nome do projeto", value=str(meta.get("project_name") or (proj_row.get("nome") or proj_row.get("name") or "")))
            project_code = st.text_input("Nº do projeto", value=str(meta.get("project_code") or (proj_row.get("cod_projeto") or proj_row.get("code") or "")))

            c1, c2 = st.columns(2)
            with c1:
                bk_logo_up = st.file_uploader("Logo BK (PNG/JPG)", type=["png", "jpg", "jpeg"], key="bk_logo_up")
            with c2:
                client_logo_up = st.file_uploader("Logo Cliente (PNG/JPG)", type=["png", "jpg", "jpeg"], key="client_logo_up")

            if st.button("Salvar dados do relatório", width="stretch"):
                _save_meta(
                    engine,
                    pid,
                    client_name=client_name.strip(),
                    project_name=project_name.strip(),
                    project_code=project_code.strip(),
                    bk_logo_b64=_b64_of_upload(bk_logo_up),
                    client_logo_b64=_b64_of_upload(client_logo_up),
                )
                st.success("Dados do relatório salvos.")

        # --- Tabela ---
        services = _list_services(engine)
        status_doc_opts = ["Em andamento - BK", "Em análise - Cliente", "Em revisão - BK", "Aprovado - Cliente"]

        doc_df = _list_doc_tasks(engine, pid)
        if doc_df.empty:
            doc_df = pd.DataFrame(
                [
                    {
                        "id": None,
                        "service_name": "" if not services else services[0],
                        "doc_name": "",
                        "doc_number": "",
                        "revision": "R0A",
                        "status": "Em andamento - BK",
                        "responsible": "BK",
                        "start_date": date.today(),
                        "completion_date": None,
                    }
                ]
            )

        original_rows = {int(r["id"]): dict(r) for r in doc_df.to_dict(orient="records") if pd.notna(r.get("id"))}

        st.caption("Preencha/edite direto na tabela. Você pode adicionar/remover linhas.")

        edited = st.data_editor(
            doc_df,
            num_rows="dynamic",
            width='stretch',
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", disabled=True),
                "service_name": st.column_config.SelectboxColumn("Serviço", options=services or [""], required=False),
                "doc_name": st.column_config.TextColumn("Nome do documento", required=False),
                "doc_number": st.column_config.TextColumn("Nº do documento", required=False),
                "revision": st.column_config.TextColumn("Revisão", required=False),
                "status": st.column_config.SelectboxColumn("Status", options=status_doc_opts, required=False),
                "responsible": st.column_config.SelectboxColumn("Responsável", options=["BK", "CLIENTE"], required=False),
                "start_date": st.column_config.DateColumn("Data de início", required=False),
                "completion_date": st.column_config.DateColumn("Data de conclusão", required=False),
            },
            key=f"doc_table_{pid}",
        )

        cbtn1, cbtn2 = st.columns([1, 1])
        with cbtn1:
            if st.button("Salvar tabela", type="primary", width="stretch"):
                edited_rows = edited.to_dict(orient="records")
                edited_ids = set()

                # upsert
                for r in edited_rows:
                    rid = r.get("id")
                    if rid is None or (isinstance(rid, float) and pd.isna(rid)):
                        # ignora linha totalmente vazia
                        if not any(str(r.get(k) or "").strip() for k in ["service_name", "doc_name", "doc_number"]):
                            continue
                        _insert_doc_task(engine, pid, r)
                    else:
                        rid_int = int(rid)
                        edited_ids.add(rid_int)
                        _update_doc_task(engine, pid, rid_int, r, original_rows.get(rid_int, {}))

                # delete (linhas removidas)
                for old_id in list(original_rows.keys()):
                    if old_id not in edited_ids:
                        _delete_doc_task(engine, pid, old_id)

                st.success("Tabela atualizada.")
                st.rerun()

        with cbtn2:
            # exclusão explícita (opcional)
            ids = [int(x) for x in doc_df["id"].dropna().tolist()] if "id" in doc_df.columns else []
            del_id = st.selectbox("Excluir por ID", options=[0] + ids, format_func=lambda x: "Selecione..." if x == 0 else str(x))
            if st.button("Excluir", width="stretch") and del_id and del_id != 0:
                _delete_doc_task(engine, pid, int(del_id))
                st.success("Linha excluída.")
                st.rerun()

        # --- Métricas e gráficos ---
        st.divider()
        st.subheader("Tempos BK x Cliente")

        times = _compute_times(engine, pid)
        if times.empty:
            st.info("Ainda não há histórico suficiente para calcular tempos (salve a tabela e altere status/revisão).") 
        else:
            # total por ator
            total = times.groupby("actor", as_index=False)["days"].sum().sort_values("days", ascending=False)
            st.dataframe(total, width='stretch')

            # por documento
            pivot = times.pivot_table(index="doc_task_id", columns="actor", values="days", aggfunc="sum", fill_value=0).reset_index()
            st.dataframe(pivot, width='stretch')

        st.divider()
        st.subheader("Relatório (HTML)")
        st.caption("Gera um HTML com cabeçalho + tabela + resumo de tempos (padrão BK).")

        meta = _load_meta(engine, pid) or {}
        doc_df_now = _list_doc_tasks(engine, pid)

        if st.button("Gerar relatório", width="stretch"):
            html = _render_html_report(meta, doc_df_now, times)
            st.download_button(
                "Baixar relatório (HTML)",
                data=html.encode("utf-8"),
                file_name=f"controle_projetos_{pid}.html",
                mime="text/html",
                width="stretch",
            )


def _render_html_report(meta: Dict, doc_df: pd.DataFrame, times: pd.DataFrame) -> str:
    """HTML simples (sem alterar templates)."""
    bk_logo = meta.get("bk_logo_b64") or ""
    cl_logo = meta.get("client_logo_b64") or ""
    client = (meta.get("client_name") or "").strip()
    proj = (meta.get("project_name") or "").strip()
    code = (meta.get("project_code") or "").strip()

    def img(b64: str, align: str) -> str:
        if not b64:
            return f"<div class='logo placeholder {align}'></div>"
        return f"<img class='logo {align}' src='data:image/png;base64,{b64}' />"

    # tabela
    cols = ["id", "service_name", "doc_name", "doc_number", "revision", "status", "responsible", "start_date", "completion_date"]
    df = doc_df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    rows = []
    for _, r in df.iterrows():
        rows.append(
            "<tr>" +
            "".join([f"<td>{'' if pd.isna(r[c]) else r[c]}</td>" for c in cols]) +
            "</tr>"
        )
    body_rows = "\n".join(rows) if rows else "<tr><td colspan='9' class='muted'>Sem dados.</td></tr>"

    # resumo tempos
    resumo = ""
    if not times.empty:
        tot = times.groupby("actor", as_index=False)["days"].sum()
        bk = float(tot.loc[tot["actor"] == "BK", "days"].sum())
        cl = float(tot.loc[tot["actor"] == "CLIENTE", "days"].sum())
        resumo = f"<div class='kpi'>BK: <b>{bk:.1f} dias</b> &nbsp;|&nbsp; CLIENTE: <b>{cl:.1f} dias</b></div>"

    html = f"""<!DOCTYPE html>
<html lang='pt-br'>
<head>
<meta charset='utf-8' />
<meta name='viewport' content='width=device-width, initial-scale=1' />
<title>Controle de Projetos</title>
<style>
  body {{ font-family: Arial, Helvetica, sans-serif; background:#f6f7fb; margin:0; padding:24px; color:#1f2937; }}
  .card {{ background:white; border:1px solid #e5e7eb; border-radius:14px; padding:18px 18px 24px 18px; box-shadow:0 6px 18px rgba(15,23,42,.06); }}
  .hdr {{ display:flex; align-items:center; gap:12px; }}
  .logo {{ height:56px; object-fit:contain; }}
  .logo.left {{ margin-right:auto; }}
  .logo.right {{ margin-left:auto; }}
  .placeholder {{ width:120px; height:56px; border:1px dashed #cbd5e1; border-radius:10px; }}
  h1 {{ margin:0; font-size:22px; }}
  .sub {{ margin-top:4px; color:#6b7280; font-size:13px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:16px; }}
  th, td {{ border:1px solid #e5e7eb; padding:8px 10px; font-size:12px; vertical-align:top; }}
  th {{ background:#f3f4f6; text-align:left; }}
  .muted {{ color:#6b7280; }}
  .kpi {{ margin-top:12px; padding:10px 12px; background:#f1f5ff; border:1px solid #dbeafe; border-radius:12px; font-size:13px; }}
</style>
</head>
<body>
  <div class='card'>
    <div class='hdr'>
      {img(bk_logo,'left')}
      <div style='text-align:center; flex:1'>
        <h1>BK Engenharia e Tecnologia</h1>
        <div class='sub'>Controle de Projetos</div>
        <div class='sub'><b>{client}</b> — {proj} {('('+code+')') if code else ''}</div>
      </div>
      {img(cl_logo,'right')}
    </div>

    {resumo}

    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Serviço</th>
          <th>Nome do documento</th>
          <th>Nº do documento</th>
          <th>Revisão</th>
          <th>Status</th>
          <th>Responsável</th>
          <th>Data início</th>
          <th>Data conclusão</th>
        </tr>
      </thead>
      <tbody>
        {body_rows}
      </tbody>
    </table>
  </div>
</body>
</html>"""
    return html


if __name__ == "__main__":
    main()
