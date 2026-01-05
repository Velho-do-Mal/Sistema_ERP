
import streamlit as st
import pandas as pd
from sqlalchemy import text

from bk_erp_shared.theme import apply_theme
from bk_erp_shared.erp_db import get_finance_db, ensure_erp_tables
from bk_erp_shared.auth import login_and_guard, can_view
from bk_erp_shared.sales import list_product_services, upsert_product_service, delete_product_service, UNITS

import bk_finance

st.set_page_config(page_title="BK_ERP - Cadastros", layout="wide")
apply_theme()
ensure_erp_tables()

engine, SessionLocal = get_finance_db()
login_and_guard(SessionLocal)

st.markdown('<div class="bk-card"><div class="bk-title">Cadastros</div><div class="bk-subtitle">Base central do BK_ERP: clientes, fornecedores, centros de custo, categorias e leads.</div></div>', unsafe_allow_html=True)

tabs = st.tabs(["👥 Clientes", "🏭 Fornecedores", "🏷️ Categorias", "🎯 Centros de Custo", "🧲 Leads", "🧰 Serviços/Produtos"])

with tabs[0]:
    if can_view("financeiro"):
        bk_finance.clients_ui(SessionLocal)
    else:
        st.warning("Sem permissão para editar clientes.")

with tabs[1]:
    if can_view("financeiro"):
        bk_finance.suppliers_ui(SessionLocal)
    else:
        st.warning("Sem permissão para editar fornecedores.")

with tabs[2]:
    if can_view("financeiro"):
        bk_finance.categories_ui(SessionLocal)
    else:
        st.warning("Sem permissão para editar categorias.")

with tabs[3]:
    if can_view("financeiro"):
        bk_finance.cost_centers_ui(SessionLocal)
    else:
        st.warning("Sem permissão para editar centros de custo.")

with tabs[4]:
    st.header("🧲 Leads (CRM)")
    if not can_view("financeiro"):
        st.info("Perfil leitura: você pode visualizar, mas não editar.")
    # CRUD simples via SQL
    with st.form("lead_form"):
        lead_id = st.number_input("ID (0 = novo)", min_value=0, step=1, value=0)
        name = st.text_input("Nome *")
        company = st.text_input("Empresa")
        email = st.text_input("Email")
        phone = st.text_input("Telefone")
        status = st.selectbox("Status", ["novo", "contato", "proposta", "ganho", "perdido"], index=0)
        notes = st.text_area("Observações", height=90)
        ok = st.form_submit_button("Salvar")

    if ok:
        if not name.strip():
            st.error("Nome é obrigatório.")
        elif not can_view("financeiro"):
            st.error("Sem permissão para salvar.")
        else:
            with engine.begin() as conn:
                if lead_id and lead_id > 0:
                    conn.execute(text("""
                        UPDATE leads SET name=:name, company=:company, email=:email, phone=:phone, status=:status, notes=:notes
                        WHERE id=:id
                    """), dict(id=int(lead_id), name=name.strip(), company=company.strip() or None, email=email.strip() or None,
                              phone=phone.strip() or None, status=status, notes=notes.strip() or None))
                    st.success("Lead atualizado.")
                else:
                    conn.execute(text("""
                        INSERT INTO leads (name, company, email, phone, status, notes)
                        VALUES (:name,:company,:email,:phone,:status,:notes)
                    """), dict(name=name.strip(), company=company.strip() or None, email=email.strip() or None,
                              phone=phone.strip() or None, status=status, notes=notes.strip() or None))
                    st.success("Lead criado.")
            st.rerun()

    st.markdown("---")
    df = pd.read_sql(text("SELECT id, name AS Nome, company AS Empresa, email AS Email, phone AS Telefone, status AS Status, created_at AS CriadoEm FROM leads ORDER BY id DESC"), engine)
    st.dataframe(df, use_container_width=True, height=360)

    c1, c2 = st.columns([1,2])
    with c1:
        del_id = st.number_input("ID para excluir", min_value=0, step=1, value=0, key="lead_del")
    with c2:
        if st.button("Excluir lead", type="secondary"):
            if not can_view("financeiro"):
                st.error("Sem permissão.")
            elif del_id and del_id>0:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM leads WHERE id=:id"), {"id": int(del_id)})
                st.success("Excluído.")
                st.rerun()


with tabs[5]:
    if can_view("financeiro"):
        st.markdown("### 🧰 Catálogo de Serviços/Produtos")
        st.caption("Cadastre itens reutilizáveis para montar propostas (valores e unidades podem ser ajustados na proposta).")

        with engine.begin() as conn:
            df = list_product_services(conn)

        if df.empty:
            st.info("Nenhum item cadastrado ainda.")
        else:
            st.dataframe(
                df[["id","type","name","default_unit","default_unit_price","active","created_at"]],
                use_container_width=True,
                hide_index=True
            )

        st.markdown("#### ➕ Novo / Editar item")
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            edit_id = st.number_input("ID (para editar)", min_value=0, value=0, step=1)
        with col2:
            name = st.text_input("Nome do serviço/produto", value="")
        with col3:
            item_type = st.selectbox("Tipo", ["servico","produto"], index=0)

        code = st.text_input("Código (opcional)", value="")
        description = st.text_area("Descrição", value="", height=90)
        c4, c5, c6 = st.columns([1,1,1])
        with c4:
            default_unit = st.selectbox("Unidade padrão", UNITS, index=0)
        with c5:
            default_price = st.number_input("Preço padrão (R$)", min_value=0.0, value=0.0, step=50.0)
        with c6:
            active = st.toggle("Ativo", value=True)

        a1, a2 = st.columns([1,1])
        with a1:
            if st.button("Salvar item", use_container_width=True):
                with engine.begin() as conn:
                    upsert_product_service(conn, {
                        "id": int(edit_id) if edit_id else None,
                        "code": code.strip() or None,
                        "type": item_type,
                        "name": name.strip(),
                        "description": description.strip() or None,
                        "default_unit": default_unit,
                        "default_unit_price": float(default_price),
                        "active": bool(active),
                    })
                st.success("Item salvo.")
                st.rerun()
        with a2:
            if st.button("Excluir item", use_container_width=True, disabled=(not edit_id)):
                with engine.begin() as conn:
                    delete_product_service(conn, int(edit_id))
                st.warning("Item excluído.")
                st.rerun()
