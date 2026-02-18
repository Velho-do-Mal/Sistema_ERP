
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
    st.markdown('<div class="bk-section">🧲 Leads (CRM)</div>', unsafe_allow_html=True)
    if not can_view("financeiro"):
        st.info("Perfil leitura: você pode visualizar, mas não editar.")

    # ── Criar novo lead ──
    st.markdown("**➕ Novo Lead**")
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        new_lead_name = st.text_input("Nome *", key="new_lead_name")
        new_lead_company = st.text_input("Empresa", key="new_lead_company")
    with lc2:
        new_lead_email = st.text_input("Email", key="new_lead_email")
        new_lead_phone = st.text_input("Telefone", key="new_lead_phone")
    with lc3:
        new_lead_status = st.selectbox("Status", ["novo", "contato", "proposta", "ganho", "perdido"], key="new_lead_status")
        new_lead_notes = st.text_input("Observações", key="new_lead_notes")

    if st.button("💾 Criar Lead", key="btn_create_lead", type="primary"):
        if not new_lead_name.strip():
            st.error("Nome é obrigatório.")
        elif not can_view("financeiro"):
            st.error("Sem permissão.")
        else:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO leads (name, company, email, phone, status, notes)
                    VALUES (:name,:company,:email,:phone,:status,:notes)
                """), dict(
                    name=new_lead_name.strip(),
                    company=new_lead_company.strip() or None,
                    email=new_lead_email.strip() or None,
                    phone=new_lead_phone.strip() or None,
                    status=new_lead_status,
                    notes=new_lead_notes.strip() or None
                ))
            st.success(f"Lead '{new_lead_name.strip()}' criado.")
            st.rerun()

    st.markdown("---")

    # ── Tabela editável ──
    # IMPORTANTE: usar colunas minúsculas no SQL e renomear depois no pandas.
    # pd.read_sql com aliases maiúsculos (AS Nome) pode retornar "name" em
    # alguns drivers SQLAlchemy+SQLite — causaria KeyError na linha do dropdown.
    df_leads_raw = pd.read_sql(text("""
        SELECT id, name, company, email, phone, status, notes
        FROM leads ORDER BY id DESC
    """), engine)

    if df_leads_raw.empty:
        st.info("Nenhum lead cadastrado ainda.")
    else:
        # Renomear AQUI, depois do read_sql — seguro em todos os drivers
        df_leads = df_leads_raw.rename(columns={
            "name": "Nome", "company": "Empresa", "email": "Email",
            "phone": "Telefone", "status": "Status", "notes": "Obs"
        })

        edited_leads = st.data_editor(
            df_leads,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="editor_leads",
            column_config={
                "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                "Nome": st.column_config.TextColumn("Nome", width="medium"),
                "Empresa": st.column_config.TextColumn("Empresa", width="medium"),
                "Email": st.column_config.TextColumn("Email", width="medium"),
                "Telefone": st.column_config.TextColumn("Telefone", width="small"),
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=["novo", "contato", "proposta", "ganho", "perdido"],
                    width="small"
                ),
                "Obs": st.column_config.TextColumn("Observações", width="large"),
            }
        )

        if st.button("💾 Salvar edições de Leads", key="btn_save_leads") and can_view("financeiro"):
            changed = 0
            for _, row in edited_leads.iterrows():
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE leads SET name=:name, company=:company, email=:email,
                               phone=:phone, status=:status, notes=:notes WHERE id=:id
                    """), dict(
                        id=int(row["id"]),
                        name=str(row["Nome"]).strip(),
                        company=str(row.get("Empresa") or "").strip() or None,
                        email=str(row.get("Email") or "").strip() or None,
                        phone=str(row.get("Telefone") or "").strip() or None,
                        status=str(row["Status"]),
                        notes=str(row.get("Obs") or "").strip() or None,
                    ))
                changed += 1
            st.success(f"{changed} lead(s) salvo(s).")
            st.rerun()

        st.markdown("---")
        st.markdown("**🗑️ Excluir lead:**")
        # Usa df_leads_raw (colunas minúsculas) — 100% seguro, sem alias
        lead_del_opts = {"— selecione —": None} | {
            f"{int(row['id'])} — {row['name']}": int(row["id"])
            for _, row in df_leads_raw.iterrows()
        }
        lead_del_sel = st.selectbox("Selecione para excluir", list(lead_del_opts.keys()), key="lead_del_sel")
        if st.button("Excluir selecionado", type="secondary", key="btn_del_lead") and can_view("financeiro"):
            lead_del_id = lead_del_opts.get(lead_del_sel)
            if lead_del_id:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM leads WHERE id=:id"), {"id": lead_del_id})
                st.success("Lead excluído.")
                st.rerun()


with tabs[5]:
    if can_view("financeiro"):
        st.markdown('<div class="bk-section">🧰 Catálogo de Serviços / Produtos</div>', unsafe_allow_html=True)
        st.caption("Cadastre itens reutilizáveis para montar propostas. Valores podem ser ajustados na proposta.")

        # ── Criar novo item ──
        st.markdown("**➕ Novo Item**")
        p1, p2, p3 = st.columns([2, 1, 1])
        with p1:
            new_ps_name = st.text_input("Nome *", key="new_ps_name")
        with p2:
            new_ps_type = st.selectbox("Tipo", ["servico", "produto"], key="new_ps_type",
                                        format_func=lambda x: "Serviço" if x == "servico" else "Produto")
        with p3:
            new_ps_code = st.text_input("Código (opcional)", key="new_ps_code")

        p4, p5, p6 = st.columns([3, 1, 1])
        with p4:
            new_ps_desc = st.text_input("Descrição breve", key="new_ps_desc")
        with p5:
            new_ps_unit = st.selectbox("Unidade", UNITS, key="new_ps_unit")
        with p6:
            new_ps_price = st.number_input("Preço padrão (R$)", min_value=0.0, value=0.0, step=50.0, key="new_ps_price")

        if st.button("💾 Criar Item", key="btn_create_ps", type="primary"):
            if not new_ps_name.strip():
                st.error("Nome é obrigatório.")
            else:
                with engine.begin() as conn:
                    upsert_product_service(conn, {
                        "id": None,
                        "code": new_ps_code.strip() or None,
                        "type": new_ps_type,
                        "name": new_ps_name.strip(),
                        "description": new_ps_desc.strip() or None,
                        "default_unit": new_ps_unit,
                        "default_unit_price": float(new_ps_price),
                        "active": True,
                    })
                st.success(f"Item '{new_ps_name.strip()}' criado.")
                st.rerun()

        st.markdown("---")

        with engine.begin() as conn:
            df_ps = list_product_services(conn)

        if df_ps.empty:
            st.info("Nenhum item cadastrado ainda.")
        else:
            # Renomear para exibição amigável
            df_ps_show = df_ps[["id", "code", "type", "name", "default_unit", "default_unit_price", "active"]].copy()
            df_ps_show.columns = ["id", "Código", "Tipo", "Nome", "Unidade", "Preço Padrão (R$)", "Ativo"]

            edited_ps = st.data_editor(
                df_ps_show,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="editor_ps",
                column_config={
                    "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                    "Código": st.column_config.TextColumn("Código", width="small"),
                    "Tipo": st.column_config.SelectboxColumn("Tipo", options=["servico", "produto"], width="small"),
                    "Nome": st.column_config.TextColumn("Nome", width="large"),
                    "Unidade": st.column_config.SelectboxColumn("Unidade", options=UNITS, width="small"),
                    "Preço Padrão (R$)": st.column_config.NumberColumn("Preço Padrão (R$)", format="R$ %.2f", step=10.0, width="medium"),
                    "Ativo": st.column_config.CheckboxColumn("Ativo", width="small"),
                }
            )

            if st.button("💾 Salvar edições", key="btn_save_ps"):
                changed = 0
                for _, row in edited_ps.iterrows():
                    with engine.begin() as conn:
                        upsert_product_service(conn, {
                            "id": int(row["id"]),
                            "code": str(row["Código"]).strip() or None,
                            "type": str(row["Tipo"]),
                            "name": str(row["Nome"]).strip(),
                            "description": None,
                            "default_unit": str(row["Unidade"]),
                            "default_unit_price": float(row["Preço Padrão (R$)"]),
                            "active": bool(row["Ativo"]),
                        })
                    changed += 1
                st.success(f"{changed} item(ns) salvo(s).")
                st.rerun()

            st.markdown("---")
            st.markdown("**🗑️ Excluir item:**")
            # Usa df_ps (colunas originais) para evitar KeyError de alias
            ps_del_opts = {"— selecione —": None} | {
                f"{int(row['id'])} — {row['name']}": int(row["id"])
                for _, row in df_ps.iterrows()
            }
            ps_del_sel = st.selectbox("Selecione para excluir", list(ps_del_opts.keys()), key="ps_del_sel")
            if st.button("Excluir selecionado", type="secondary", key="btn_del_ps"):
                ps_del_id = ps_del_opts.get(ps_del_sel)
                if ps_del_id:
                    with engine.begin() as conn:
                        delete_product_service(conn, int(ps_del_id))
                    st.success("Item excluído.")
                    st.rerun()
    else:
        st.warning("Sem permissão para editar serviços/produtos.")
