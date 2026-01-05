
import streamlit as st
import pandas as pd
from datetime import date
from sqlalchemy import text

from bk_erp_shared.theme import apply_theme
from bk_erp_shared.erp_db import get_finance_db, ensure_erp_tables
from bk_erp_shared.auth import login_and_guard, can_view

import bk_finance

st.set_page_config(page_title="BK_ERP - Compras", layout="wide")
apply_theme()
ensure_erp_tables()

engine, SessionLocal = get_finance_db()
login_and_guard(SessionLocal)

st.markdown('<div class="bk-card"><div class="bk-title">Compras</div><div class="bk-subtitle">Pedidos de compra integrados com Fornecedores e Projetos.</div></div>', unsafe_allow_html=True)

session = SessionLocal()
suppliers = session.query(bk_finance.Supplier).order_by(bk_finance.Supplier.name.asc()).all()
projects = pd.read_sql(text("SELECT id, nome FROM projects ORDER BY id DESC"), engine)
session.close()

sup_map = {s.id: s.name for s in suppliers}
proj_map = {int(r["id"]): (r["nome"] or f"Projeto {r['id']}") for _, r in projects.iterrows()}

def fmt_money(v):
    return bk_finance.format_currency(float(v or 0))

with st.expander("➕ Novo / Editar pedido de compra", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        po_id = st.number_input("ID (0 = novo)", min_value=0, step=1, value=0)
        code = st.text_input("Código (ex: PC-2026-001)")
        status = st.selectbox("Status", ["aberta", "aprovacao", "aprovada", "encerrada", "cancelada"], index=0)
    with c2:
        supplier_id = st.selectbox("Fornecedor", options=[0] + [s.id for s in suppliers], format_func=lambda i: "—" if i==0 else sup_map.get(i,""), index=0)
        project_id = st.selectbox("Projeto", options=[0] + list(proj_map.keys()), format_func=lambda i: "—" if i==0 else proj_map.get(i,""), index=0)
    with c3:
        order_date = st.date_input("Data do pedido", value=date.today())
        expected_date = st.date_input("Entrega prevista", value=date.today())
        value_total = st.number_input("Valor total (R$)", min_value=0.0, step=100.0, value=0.0)

    notes = st.text_area("Observações", height=80)

    if st.button("Salvar pedido", type="primary"):
        if not can_view("financeiro"):
            st.error("Sem permissão para salvar.")
        elif not code.strip():
            st.error("Informe um código.")
        else:
            with engine.begin() as conn:
                if po_id and po_id > 0:
                    conn.execute(text("""
                        UPDATE purchase_orders
                        SET code=:code, supplier_id=:supplier_id, project_id=:project_id, order_date=:order_date,
                            expected_date=:expected_date, value_total=:value_total, status=:status, notes=:notes
                        WHERE id=:id
                    """), dict(id=int(po_id), code=code.strip(), supplier_id=int(supplier_id) if supplier_id else None,
                              project_id=int(project_id) if project_id else None,
                              order_date=order_date.strftime("%Y-%m-%d"),
                              expected_date=expected_date.strftime("%Y-%m-%d"),
                              value_total=float(value_total), status=status, notes=notes.strip() or None))
                    st.success("Pedido atualizado.")
                else:
                    conn.execute(text("""
                        INSERT INTO purchase_orders (code, supplier_id, project_id, order_date, expected_date, value_total, status, notes)
                        VALUES (:code,:supplier_id,:project_id,:order_date,:expected_date,:value_total,:status,:notes)
                    """), dict(code=code.strip(), supplier_id=int(supplier_id) if supplier_id else None,
                              project_id=int(project_id) if project_id else None,
                              order_date=order_date.strftime("%Y-%m-%d"),
                              expected_date=expected_date.strftime("%Y-%m-%d"),
                              value_total=float(value_total), status=status, notes=notes.strip() or None))
                    st.success("Pedido criado.")
            st.rerun()

st.markdown("---")
df = pd.read_sql(text("""
    SELECT id, code, supplier_id, project_id, order_date, expected_date, value_total, status
    FROM purchase_orders
    ORDER BY id DESC
"""), engine)

if df.empty:
    st.info("Nenhum pedido cadastrado ainda.")
else:
    df["Fornecedor"] = df["supplier_id"].map(lambda i: sup_map.get(i,"—"))
    df["Projeto"] = df["project_id"].map(lambda i: proj_map.get(int(i), "—") if pd.notna(i) else "—")
    df["Valor"] = df["value_total"].map(fmt_money)
    show = df[["id","code","Fornecedor","Projeto","order_date","expected_date","Valor","status"]]
    st.dataframe(show, use_container_width=True, height=420)

c1, c2 = st.columns([1,2])
with c1:
    del_id = st.number_input("ID para excluir", min_value=0, step=1, value=0, key="po_del")
with c2:
    if st.button("Excluir pedido", type="secondary"):
        if not can_view("financeiro"):
            st.error("Sem permissão.")
        elif del_id and del_id>0:
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM purchase_orders WHERE id=:id"), {"id": int(del_id)})
            st.success("Excluído.")
            st.rerun()
