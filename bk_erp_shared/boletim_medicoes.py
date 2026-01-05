"""
BK_ERP - Boletim de Medições (camada de dados)

Implementa as funções esperadas por pages/7_📏_Boletim_de_Medicoes.py:
- list_projects
- list_contract_items
- upsert_contract_items
- list_measurements
- create_measurement
- get_measurement_header
- get_measurement_items_with_context
- save_measurement_items
- set_measurement_status
- measurement_summary

Compatível com Postgres (Neon) e SQLite (fallback).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sqlalchemy import inspect, text

import bk_finance


def _get_db() -> Tuple[Any, Any]:
    SessionLocal, engine = bk_finance.get_finance_db()
    return SessionLocal, engine


def _has_column(engine, table: str, col: str) -> bool:
    insp = inspect(engine)
    try:
        cols = [c["name"] for c in insp.get_columns(table)]
        return col in cols
    except Exception:
        return False


def _ensure_tables(SessionLocal) -> None:
    # contract items
    ddl_pg_contract = """
    CREATE TABLE IF NOT EXISTS contract_items (
        id SERIAL PRIMARY KEY,
        project_id INTEGER NOT NULL,
        code TEXT,
        descricao TEXT NOT NULL,
        unidade TEXT DEFAULT '',
        qtde_contratada DOUBLE PRECISION DEFAULT 0,
        valor_unit DOUBLE PRECISION DEFAULT 0,
        ativo BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    ddl_sqlite_contract = """
    CREATE TABLE IF NOT EXISTS contract_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        code TEXT,
        descricao TEXT NOT NULL,
        unidade TEXT DEFAULT '',
        qtde_contratada REAL DEFAULT 0,
        valor_unit REAL DEFAULT 0,
        ativo INTEGER DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    ddl_pg_meas = """
    CREATE TABLE IF NOT EXISTS measurements (
        id SERIAL PRIMARY KEY,
        project_id INTEGER NOT NULL,
        period_start DATE,
        period_end DATE,
        reference TEXT,
        status TEXT DEFAULT 'RASCUNHO',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    ddl_sqlite_meas = """
    CREATE TABLE IF NOT EXISTS measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        period_start DATE,
        period_end DATE,
        reference TEXT,
        status TEXT DEFAULT 'RASCUNHO',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    ddl_pg_items = """
    CREATE TABLE IF NOT EXISTS measurement_items (
        id SERIAL PRIMARY KEY,
        measurement_id INTEGER NOT NULL,
        contract_item_id INTEGER NOT NULL,
        qtde_periodo DOUBLE PRECISION DEFAULT 0,
        valor_unit DOUBLE PRECISION DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    ddl_sqlite_items = """
    CREATE TABLE IF NOT EXISTS measurement_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        measurement_id INTEGER NOT NULL,
        contract_item_id INTEGER NOT NULL,
        qtde_periodo REAL DEFAULT 0,
        valor_unit REAL DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    with SessionLocal() as conn:
        # tenta Postgres, cai para SQLite
        try:
            conn.execute(text(ddl_pg_contract))
            conn.execute(text(ddl_pg_meas))
            conn.execute(text(ddl_pg_items))
        except Exception:
            conn.execute(text(ddl_sqlite_contract))
            conn.execute(text(ddl_sqlite_meas))
            conn.execute(text(ddl_sqlite_items))


def list_projects(active_only: bool = True) -> pd.DataFrame:
    SessionLocal, engine = _get_db()
    has_created = _has_column(engine, "projects", "created_at")
    has_archived = _has_column(engine, "projects", "archived")

    select_cols = "id, COALESCE(nome,'(sem nome)') as nome, COALESCE(status,'') as status"
    select_cols += ", created_at" if has_created else ", NULL as created_at"

    where = "WHERE 1=1"
    if active_only and has_archived:
        where += " AND COALESCE(archived, FALSE) = FALSE"

    order = "created_at DESC, id DESC" if has_created else "id DESC"
    sql = f"SELECT {select_cols} FROM projects {where} ORDER BY {order}"
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def list_contract_items(project_id: int) -> pd.DataFrame:
    SessionLocal, engine = _get_db()
    _ensure_tables(SessionLocal)
    sql = """
        SELECT id, COALESCE(code,'') as code, COALESCE(descricao,'') as descricao,
               COALESCE(unidade,'') as unidade,
               COALESCE(qtde_contratada,0) as qtde_contratada,
               COALESCE(valor_unit,0) as valor_unit,
               COALESCE(ativo,TRUE) as ativo
        FROM contract_items
        WHERE project_id=:pid
        ORDER BY id DESC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"pid": int(project_id)})
    # normaliza ativo para bool no streamlit
    if "ativo" in df.columns:
        df["ativo"] = df["ativo"].astype(bool)
    return df


def upsert_contract_items(project_id: int, df: pd.DataFrame) -> None:
    SessionLocal, engine = _get_db()
    _ensure_tables(SessionLocal)
    # garante colunas
    cols = {c.lower(): c for c in df.columns}
    def get_col(name, default=None):
        return cols.get(name, default)

    id_col = get_col("id")
    code_col = get_col("code") or get_col("codigo") or get_col("código")
    desc_col = get_col("descricao") or get_col("descrição") or get_col("description")
    unit_col = get_col("unidade") or get_col("unit")
    qty_col = get_col("qtde_contratada") or get_col("qtd_contratada") or get_col("quantidade")
    vu_col = get_col("valor_unit") or get_col("vlr_unit") or get_col("unit_price")
    ativo_col = get_col("ativo") or get_col("active")

    with SessionLocal() as conn:
        for _, r in df.iterrows():
            rid = int(r[id_col]) if id_col and pd.notna(r.get(id_col)) else None
            payload = {
                "pid": int(project_id),
                "code": (str(r.get(code_col)) if code_col else "").strip() if pd.notna(r.get(code_col, "")) else "",
                "descricao": (str(r.get(desc_col)) if desc_col else "").strip() if pd.notna(r.get(desc_col, "")) else "",
                "unidade": (str(r.get(unit_col)) if unit_col else "").strip() if pd.notna(r.get(unit_col, "")) else "",
                "qtde_contratada": float(r.get(qty_col, 0) or 0),
                "valor_unit": float(r.get(vu_col, 0) or 0),
                "ativo": bool(r.get(ativo_col, True)) if ativo_col else True,
            }
            if not payload["descricao"]:
                # ignora linhas vazias
                continue

            if rid:
                conn.execute(
                    text(
                        """
                        UPDATE contract_items
                        SET code=:code, descricao=:descricao, unidade=:unidade,
                            qtde_contratada=:qtde_contratada, valor_unit=:valor_unit, ativo=:ativo
                        WHERE id=:id AND project_id=:pid
                        """
                    ),
                    {**payload, "id": rid},
                )
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO contract_items (project_id, code, descricao, unidade, qtde_contratada, valor_unit, ativo)
                        VALUES (:pid, :code, :descricao, :unidade, :qtde_contratada, :valor_unit, :ativo)
                        """
                    ),
                    payload,
                )


def list_measurements(project_id: int) -> pd.DataFrame:
    SessionLocal, engine = _get_db()
    _ensure_tables(SessionLocal)
    sql = """
        SELECT id, project_id, period_start, period_end, COALESCE(reference,'') as reference,
               COALESCE(status,'RASCUNHO') as status, created_at
        FROM measurements
        WHERE project_id=:pid
        ORDER BY id DESC
    """
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params={"pid": int(project_id)})


def create_measurement(project_id: int, period_start: date, period_end: date, reference: str = "") -> int:
    SessionLocal, engine = _get_db()
    _ensure_tables(SessionLocal)
    with SessionLocal() as conn:
        # tenta Postgres RETURNING
        try:
            res = conn.execute(
                text(
                    """
                    INSERT INTO measurements (project_id, period_start, period_end, reference, status, updated_at)
                    VALUES (:pid, :ps, :pe, :ref, 'RASCUNHO', CURRENT_TIMESTAMP)
                    RETURNING id
                    """
                ),
                {"pid": int(project_id), "ps": period_start, "pe": period_end, "ref": reference or ""},
            )
            new_id = int(res.scalar())
        except Exception:
            conn.execute(
                text(
                    """
                    INSERT INTO measurements (project_id, period_start, period_end, reference, status, updated_at)
                    VALUES (:pid, :ps, :pe, :ref, 'RASCUNHO', CURRENT_TIMESTAMP)
                    """
                ),
                {"pid": int(project_id), "ps": period_start, "pe": period_end, "ref": reference or ""},
            )
            # SQLite
            try:
                res2 = conn.execute(text("SELECT last_insert_rowid()"))
                new_id = int(res2.scalar())
            except Exception:
                # fallback
                res3 = conn.execute(text("SELECT MAX(id) FROM measurements WHERE project_id=:pid"), {"pid": int(project_id)})
                new_id = int(res3.scalar() or 0)
    return new_id


def get_measurement_header(measurement_id: int) -> Dict[str, Any]:
    SessionLocal, engine = _get_db()
    _ensure_tables(SessionLocal)
    sql = """
        SELECT id, project_id, period_start, period_end, COALESCE(reference,'') as reference,
               COALESCE(status,'RASCUNHO') as status, created_at
        FROM measurements
        WHERE id=:id
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params={"id": int(measurement_id)})
    return df.iloc[0].to_dict() if not df.empty else {}


def get_measurement_items_with_context(measurement_id: int) -> pd.DataFrame:
    SessionLocal, engine = _get_db()
    _ensure_tables(SessionLocal)
    header = get_measurement_header(measurement_id)
    if not header:
        return pd.DataFrame()

    pid = int(header["project_id"])
    items = list_contract_items(pid)
    if items.empty:
        return pd.DataFrame()

    # itens já lançados neste boletim
    sql_curr = """
        SELECT id as measurement_item_id, contract_item_id, COALESCE(qtde_periodo,0) as qtde_periodo,
               COALESCE(valor_unit,0) as valor_unit
        FROM measurement_items
        WHERE measurement_id=:mid
    """
    with engine.connect() as conn:
        curr = pd.read_sql(text(sql_curr), conn, params={"mid": int(measurement_id)})

    # acumulado anterior (soma de qtde_periodo antes deste measurement_id)
    sql_prev = """
        SELECT mi.contract_item_id, SUM(COALESCE(mi.qtde_periodo,0)) as qtde_acumulada_anterior
        FROM measurement_items mi
        JOIN measurements m ON m.id = mi.measurement_id
        WHERE m.project_id=:pid AND mi.measurement_id <> :mid AND m.id < :mid
        GROUP BY mi.contract_item_id
    """
    with engine.connect() as conn:
        prev = pd.read_sql(text(sql_prev), conn, params={"pid": pid, "mid": int(measurement_id)})

    df = items.merge(prev, how="left", left_on="id", right_on="contract_item_id")
    df["qtde_acumulada_anterior"] = df["qtde_acumulada_anterior"].fillna(0.0)

    # junta com o atual
    df = df.merge(curr, how="left", left_on="id", right_on="contract_item_id", suffixes=("", "_curr"))
    df["measurement_item_id"] = df["measurement_item_id"].fillna(pd.NA)
    df["qtde_periodo"] = df["qtde_periodo"].fillna(0.0)
    # valor unit: se informado no item do boletim, usa, senão do contrato
    df["valor_unit"] = df["valor_unit"].where(df["valor_unit"].notna(), df["valor_unit_curr"])
    df["valor_unit"] = df["valor_unit"].fillna(df["valor_unit_curr"]).fillna(df["valor_unit"].fillna(0.0))
    # quando o valor_unit do contrato é coluna valor_unit; manter
    df["valor_unit"] = df["valor_unit"].fillna(0.0)

    df = df.rename(columns={
        "id": "contract_item_id",
    })

    df["qtde_acumulada"] = df["qtde_acumulada_anterior"] + df["qtde_periodo"]
    df["qtde_saldo"] = df["qtde_contratada"].fillna(0.0) - df["qtde_acumulada"]

    df["valor_periodo"] = df["qtde_periodo"] * df["valor_unit"]
    df["valor_acumulado"] = df["qtde_acumulada"] * df["valor_unit"]
    df["valor_saldo"] = df["qtde_saldo"] * df["valor_unit"]

    out = pd.DataFrame({
        "measurement_item_id": df["measurement_item_id"],
        "contract_item_id": df["contract_item_id"],
        "descricao": df["descricao"],
        "unidade": df["unidade"],
        "qtde_contratada": df["qtde_contratada"].fillna(0.0),
        "valor_unit": df["valor_unit"].fillna(0.0),
        "qtde_periodo": df["qtde_periodo"].fillna(0.0),
        "qtde_acumulada_anterior": df["qtde_acumulada_anterior"].fillna(0.0),
        "qtde_acumulada": df["qtde_acumulada"].fillna(0.0),
        "qtde_saldo": df["qtde_saldo"].fillna(0.0),
        "valor_periodo": df["valor_periodo"].fillna(0.0),
        "valor_acumulado": df["valor_acumulado"].fillna(0.0),
        "valor_saldo": df["valor_saldo"].fillna(0.0),
    })
    return out


def save_measurement_items(measurement_id: int, edited_df: pd.DataFrame) -> None:
    SessionLocal, engine = _get_db()
    _ensure_tables(SessionLocal)

    # espera colunas measurement_item_id, contract_item_id, valor_unit, qtde_periodo
    with SessionLocal() as conn:
        for _, r in edited_df.iterrows():
            cid = int(r["contract_item_id"])
            miid = r.get("measurement_item_id")
            miid_int = int(miid) if pd.notna(miid) else None
            qtde = float(r.get("qtde_periodo", 0) or 0)
            vu = float(r.get("valor_unit", 0) or 0)

            if miid_int:
                conn.execute(
                    text(
                        """
                        UPDATE measurement_items
                        SET qtde_periodo=:q, valor_unit=:vu
                        WHERE id=:id AND measurement_id=:mid
                        """
                    ),
                    {"q": qtde, "vu": vu, "id": miid_int, "mid": int(measurement_id)},
                )
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO measurement_items (measurement_id, contract_item_id, qtde_periodo, valor_unit)
                        VALUES (:mid, :cid, :q, :vu)
                        """
                    ),
                    {"mid": int(measurement_id), "cid": cid, "q": qtde, "vu": vu},
                )


def set_measurement_status(measurement_id: int, status: str) -> None:
    SessionLocal, engine = _get_db()
    _ensure_tables(SessionLocal)
    with SessionLocal() as conn:
        conn.execute(
            text("UPDATE measurements SET status=:s, updated_at=CURRENT_TIMESTAMP WHERE id=:id"),
            {"s": status, "id": int(measurement_id)},
        )


def measurement_summary(measurement_id: int) -> Dict[str, float]:
    items = get_measurement_items_with_context(measurement_id)
    if items.empty:
        return {"valor_periodo": 0.0, "valor_acumulado": 0.0, "saldo": 0.0}
    return {
        "valor_periodo": float(items["valor_periodo"].sum()),
        "valor_acumulado": float(items["valor_acumulado"].sum()),
        "saldo": float(items["valor_saldo"].sum()),
    }
