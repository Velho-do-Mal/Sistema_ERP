from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, List, Dict, Any

import pandas as pd
from sqlalchemy import text


UNITS = ["hh", "Gb", "Vb", "m", "m³", "m²", "kg", "un"]


def list_product_services(conn) -> pd.DataFrame:
    return pd.read_sql(
        text("""
            SELECT id, code, type, name, description, default_unit, default_unit_price, active, created_at
            FROM product_services
            ORDER BY active DESC, name ASC
        """),
        conn
    )


def upsert_product_service(conn, row: Dict[str, Any]) -> None:
    # Insert or update
    if row.get("id"):
        conn.execute(
            text("""
                UPDATE product_services
                SET code=:code, type=:type, name=:name, description=:description,
                    default_unit=:default_unit, default_unit_price=:default_unit_price,
                    active=:active
                WHERE id=:id
            """),
            row
        )
    else:
        conn.execute(
            text("""
                INSERT INTO product_services (code, type, name, description, default_unit, default_unit_price, active)
                VALUES (:code, :type, :name, :description, :default_unit, :default_unit_price, :active)
            """),
            row
        )


def delete_product_service(conn, product_id: int) -> None:
    conn.execute(text("DELETE FROM product_services WHERE id=:id"), {"id": int(product_id)})


def list_proposals(conn) -> pd.DataFrame:
    return pd.read_sql(
        text("""
            SELECT p.id, p.code, p.title, p.client_id, c.name AS client_name,
                   p.project_id, prj.nome AS project_name,
                   p.value_total, p.status, p.created_at, p.valid_until
            FROM proposals p
            LEFT JOIN clients c ON c.id = p.client_id
            LEFT JOIN projects prj ON prj.id = p.project_id
            ORDER BY p.created_at DESC, p.id DESC
        """),
        conn
    )


def get_proposal(conn, proposal_id: int) -> Dict[str, Any]:
    p = conn.execute(text("SELECT * FROM proposals WHERE id=:id"), {"id": int(proposal_id)}).mappings().first()
    if not p:
        raise ValueError("Proposta não encontrada.")
    items = pd.read_sql(
        text("""
            SELECT i.*, ps.name AS catalog_name
            FROM proposal_items i
            LEFT JOIN product_services ps ON ps.id = i.product_service_id
            WHERE i.proposal_id=:pid
            ORDER BY i.sort_order ASC, i.id ASC
        """),
        conn,
        params={"pid": int(proposal_id)}
    )
    return {"proposal": dict(p), "items": items}



def save_proposal(conn, proposal: Dict[str, Any]) -> int:
    """Cria/atualiza proposta. Retorna proposal_id.

    Compatível com Postgres e SQLite:
    - Postgres: usa RETURNING id
    - SQLite: usa last_insert_rowid()
    """
    dialect = getattr(getattr(conn, "engine", None), "dialect", None)
    dialect_name = getattr(dialect, "name", "")

    # proposta básica
    if proposal.get("id"):
        conn.execute(
            text("""
                UPDATE proposals
                SET code=:code, title=:title, client_id=:client_id, lead_id=:lead_id,
                    project_id=:project_id,
                    value_total=:value_total, status=:status, valid_until=:valid_until,
                    notes=:notes, objective=:objective, scope=:scope,
                    resp_contratante=:resp_contratante, resp_contratado=:resp_contratado,
                    payment_terms=:payment_terms, delivery_terms=:delivery_terms,
                    reference=:reference, observations=:observations,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=:id
            """),
            proposal
        )
        return int(proposal["id"])

    # Insert
    if dialect_name == "sqlite":
        conn.execute(
            text("""
                INSERT INTO proposals
                    (code, title, client_id, lead_id, project_id, value_total, status, valid_until, notes,
                     objective, scope, resp_contratante, resp_contratado, payment_terms, delivery_terms,
                     reference, observations)
                VALUES
                    (:code, :title, :client_id, :lead_id, :project_id, :value_total, :status, :valid_until, :notes,
                     :objective, :scope, :resp_contratante, :resp_contratado, :payment_terms, :delivery_terms,
                     :reference, :observations)
            """),
            proposal
        )
        new_id = conn.execute(text("SELECT last_insert_rowid()")).scalar()
        return int(new_id)

    # Postgres (ou outros com RETURNING)
    r = conn.execute(
        text("""
            INSERT INTO proposals
                (code, title, client_id, lead_id, project_id, value_total, status, valid_until, notes,
                 objective, scope, resp_contratante, resp_contratado, payment_terms, delivery_terms,
                 reference, observations)
            VALUES
                (:code, :title, :client_id, :lead_id, :project_id, :value_total, :status, :valid_until, :notes,
                 :objective, :scope, :resp_contratante, :resp_contratado, :payment_terms, :delivery_terms,
                 :reference, :observations)
            RETURNING id
        """),
        proposal
    )
    return int(r.scalar())



def replace_proposal_items(conn, proposal_id: int, items_df: pd.DataFrame) -> None:
    conn.execute(text("DELETE FROM proposal_items WHERE proposal_id=:pid"), {"pid": int(proposal_id)})
    for idx, row in items_df.reset_index(drop=True).iterrows():
        conn.execute(
            text("""
                INSERT INTO proposal_items
                    (proposal_id, product_service_id, description, unit, qty, unit_price, total, sort_order)
                VALUES
                    (:proposal_id, :product_service_id, :description, :unit, :qty, :unit_price, :total, :sort_order)
            """),
            {
                "proposal_id": int(proposal_id),
                "product_service_id": int(row["product_service_id"]) if str(row.get("product_service_id","")).strip() not in ["", "None", "nan"] else None,
                "description": str(row.get("description","")).strip(),
                "unit": str(row.get("unit","")).strip(),
                "qty": float(row.get("qty") or 0),
                "unit_price": float(row.get("unit_price") or 0),
                "total": float(row.get("total") or 0),
                "sort_order": int(row.get("sort_order") or idx),
            }
        )


def compute_items_totals(items_df: pd.DataFrame) -> pd.DataFrame:
    df = items_df.copy()
    if "qty" not in df: df["qty"] = 0
    if "unit_price" not in df: df["unit_price"] = 0
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0.0)
    df["total"] = (df["qty"] * df["unit_price"]).round(2)
    return df



def next_proposal_code(conn) -> str:
    """Gera código BK-PROP-YYYY-XXXX.

    - Postgres: usa date_trunc
    - SQLite: usa strftime('%Y', created_at)
    """
    year = date.today().year
    dialect = getattr(getattr(conn, "engine", None), "dialect", None)
    dialect_name = getattr(dialect, "name", "")

    if dialect_name == "sqlite":
        r = conn.execute(
            text("""
                SELECT COUNT(*) FROM proposals
                WHERE strftime('%Y', created_at) = :y
            """),
            {"y": str(year)},
        ).scalar() or 0
    else:
        r = conn.execute(
            text("""
                SELECT COUNT(*) FROM proposals
                WHERE created_at >= date_trunc('year', CURRENT_TIMESTAMP)
            """)
        ).scalar() or 0

    seq = int(r) + 1
    return f"BK-PROP-{year}-{seq:04d}"
