import streamlit as st
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import text

from bk_erp_shared.theme import apply_theme
from bk_erp_shared.erp_db import get_finance_db, ensure_erp_tables
from bk_erp_shared.auth import login_and_guard, can_view, current_user

from bk_erp_shared.sales import (
    UNITS,
    list_product_services,
    list_proposals,
    get_proposal,
    save_proposal,
    replace_proposal_items,
    compute_items_totals,
    next_proposal_code,
)

from reports.render import render_proposta_html, money_br

import bk_finance


st.set_page_config(page_title="BK_ERP - Vendas & Propostas", layout="wide")
apply_theme()
ensure_erp_tables()

engine, SessionLocal = get_finance_db()
login_and_guard(SessionLocal)

st.markdown(
    '<div class="bk-card"><div class="bk-title">📈 Vendas & Propostas</div>'
    '<div class="bk-subtitle">Fluxo inspirado em ERPs financeiros modernos (ex.: “Conta Azul-like”), '
    'mas otimizado para engenharia/empreiteiras: catálogo de serviços, proposta técnica e integração com o financeiro.</div></div>',
    unsafe_allow_html=True
)

if not can_view("financeiro"):
    st.info("Seu perfil não tem permissão para este módulo.")
    st.stop()

tab_prop, tab_leads, tab_pedidos = st.tabs(["📑 Propostas", "🧲 Leads/Pipeline", "🧾 Pedidos (em evolução)"])


def _load_clients():
    with SessionLocal() as s:
        clients = s.query(bk_finance.Client).order_by(bk_finance.Client.name.asc()).all()
        return clients


def _load_accounts():
    with SessionLocal() as s:
        accs = s.query(bk_finance.Account).order_by(bk_finance.Account.name.asc()).all()
        return accs


def _load_categories():
    with SessionLocal() as s:
        cats = s.query(bk_finance.Category).order_by(bk_finance.Category.name.asc()).all()
        return cats


def _create_receivable_from_proposal(proposal_id: int, description: str, amount: float, due: date,
                                     account_id: int | None, category_id: int | None):
    ref = f"PROPOSAL:{proposal_id}"
    with engine.begin() as conn:
        exists = conn.execute(text("SELECT id FROM transactions WHERE reference=:ref LIMIT 1"), {"ref": ref}).first()
        if exists:
            return False, "Já existe lançamento criado a partir desta proposta."
        conn.execute(
            text("""
                INSERT INTO transactions (date, due_date, paid, description, amount, type, account_id, category_id, reference)
                VALUES (:date, :due_date, FALSE, :description, :amount, 'entrada', :account_id, :category_id, :reference)
            """),
            {
                "date": due,
                "due_date": due,
                "description": description,
                "amount": float(amount),
                "account_id": int(account_id) if account_id else None,
                "category_id": int(category_id) if category_id else None,
                "reference": ref,
            }
        )
    return True, "Conta a receber criada no Financeiro."


with tab_prop:
    left, right = st.columns([1.05, 1.45], gap="large")

    with engine.begin() as conn:
        df_prop = list_proposals(conn)
        df_catalog = list_product_services(conn)

    with left:
        st.markdown("### 📑 Propostas")
        st.caption("Crie propostas técnicas com orçamento por serviços/produtos (unidade, quantidade e valores editáveis por proposta).")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("➕ Nova proposta", use_container_width=True):
                st.session_state["proposal_id"] = None
                st.session_state["items_df"] = pd.DataFrame(columns=["product_service_id","description","unit","qty","unit_price","total","sort_order"])
                st.session_state["proposal_form_seed"] = {"code": None}
                st.rerun()
        with c2:
            st.download_button(
                "⬇️ Baixar modelo (HTML)",
                data=(Path("reports/templates/proposta_BK.html").read_text(encoding="utf-8")),
                file_name="modelo_proposta_BK.html",
                mime="text/html",
                use_container_width=True
            )

        if df_prop.empty:
            st.info("Ainda não há propostas cadastradas.")
            sel_id = None
        else:
            df_view = df_prop.copy()
            df_view["valor"] = df_view["value_total"].fillna(0).map(money_br)
            st.dataframe(df_view[["id","code","title","client_name","status","valor","created_at"]], hide_index=True, use_container_width=True)
            sel_id = st.selectbox(
                "Abrir proposta (ID)",
                options=[None] + df_prop["id"].tolist(),
                format_func=lambda x: "—" if x is None else f"{int(x)} • {df_prop.loc[df_prop['id']==x,'code'].values[0]}",
                index=0
            )

        if sel_id is not None:
            st.session_state["proposal_id"] = int(sel_id)

    with right:
        proposal_id = st.session_state.get("proposal_id")

        clients = _load_clients()
        client_options = {c.name: c for c in clients}

        # seed
        if "items_df" not in st.session_state:
            st.session_state["items_df"] = pd.DataFrame(columns=["product_service_id","description","unit","qty","unit_price","total","sort_order"])

        if proposal_id:
            with engine.begin() as conn:
                data = get_proposal(conn, int(proposal_id))
            p = data["proposal"]
            items = data["items"]
            st.session_state["items_df"] = items[["product_service_id","description","unit","qty","unit_price","total","sort_order"]].copy()
        else:
            p = st.session_state.get("proposal_form_seed", {}) or {}
            if not p.get("code"):
                with engine.begin() as conn:
                    p["code"] = next_proposal_code(conn)
            p.setdefault("title", "")
            p.setdefault("status", "rascunho")
            p.setdefault("client_id", None)
            p.setdefault("valid_until", (date.today() + timedelta(days=15)).isoformat())
            p.setdefault("objective", "Apresentar proposta técnica/comercial para atendimento da demanda do contratante.")
            p.setdefault("scope", "Descrever aqui o escopo: entregáveis, limites e premissas do serviço.")
            p.setdefault("resp_contratante", "Fornecer acesso ao local, informações e aprovações necessárias.")
            p.setdefault("resp_contratado", "Executar os serviços conforme escopo e boas práticas técnicas.")
            p.setdefault("payment_terms", "30% na aprovação e 70% na entrega (ajuste conforme necessário).")
            p.setdefault("delivery_terms", "Conforme cronograma acordado em conjunto.")
            p.setdefault("reference", "")
            p.setdefault("observations", "")
            p.setdefault("notes", "")

        st.markdown("### ✍️ Editor de Proposta")
        f1, f2, f3 = st.columns([1,2,1])
        with f1:
            code = st.text_input("Código", value=p.get("code",""), disabled=True)
        with f2:
            title = st.text_input("Título", value=p.get("title",""))
        with f3:
            status = st.selectbox("Status", ["rascunho","enviada","aprovada","rejeitada","faturada"], index=["rascunho","enviada","aprovada","rejeitada","faturada"].index(p.get("status","rascunho")))

        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            client_name = st.selectbox("Cliente", options=["—"] + list(client_options.keys()),
                                       index=0 if not p.get("client_id") else 1)
            client_id = client_options.get(client_name).id if client_name != "—" else None
        with c2:
            valid_until = st.date_input("Validade até", value=date.fromisoformat(p.get("valid_until") or date.today().isoformat()))
        with c3:
            reference = st.text_input("Referência (obra/contrato)", value=p.get("reference",""))

        observations = st.text_area("Observações (visíveis no documento)", value=p.get("observations",""), height=80)

        st.markdown("#### Texto técnico")
        t1, t2 = st.columns(2)
        with t1:
            objective = st.text_area("Objetivo", value=p.get("objective",""), height=120)
            resp_contratante = st.text_area("Responsabilidade do Contratante", value=p.get("resp_contratante",""), height=120)
        with t2:
            scope = st.text_area("Escopo", value=p.get("scope",""), height=120)
            resp_contratado = st.text_area("Responsabilidade do Contratado", value=p.get("resp_contratado",""), height=120)

        pay1, pay2 = st.columns(2)
        with pay1:
            payment_terms = st.text_area("Condições de pagamento", value=p.get("payment_terms",""), height=90)
        with pay2:
            delivery_terms = st.text_area("Prazo/Entrega", value=p.get("delivery_terms",""), height=90)

        st.markdown("#### Orçamento")
        if df_catalog.empty:
            st.info("Cadastre serviços/produtos em **Cadastros → Serviços/Produtos** para reutilizar na proposta.")
        addc1, addc2, addc3, addc4 = st.columns([2,1,1,1])
        with addc1:
            catalog_pick = st.selectbox(
                "Adicionar do catálogo",
                options=[None] + df_catalog["id"].tolist(),
                format_func=lambda x: "—" if x is None else f"{int(x)} • {df_catalog.loc[df_catalog['id']==x,'name'].values[0]}",
                index=0
            )
        with addc2:
            qty = st.number_input("Qtd", min_value=0.0, value=1.0, step=1.0)
        with addc3:
            unit = st.selectbox("Un.", UNITS, index=0)
        with addc4:
            unit_price = st.number_input("V. Unit (R$)", min_value=0.0, value=0.0, step=50.0)

        if st.button("➕ Inserir item", use_container_width=True):
            desc = ""
            ps_id = None
            if catalog_pick is not None:
                row = df_catalog[df_catalog["id"] == catalog_pick].iloc[0]
                ps_id = int(row["id"])
                desc = str(row["name"])
                if not unit or unit == "hh":
                    unit = str(row.get("default_unit") or unit)
                if unit_price == 0.0:
                    unit_price = float(row.get("default_unit_price") or 0)
            else:
                desc = "Item (descreva)"

            df_items = st.session_state["items_df"].copy()
            df_items = df_items.loc[df_items["description"].astype(str).str.strip() != ""].copy()
            df_items.loc[len(df_items)] = {
                "product_service_id": ps_id,
                "description": desc,
                "unit": unit,
                "qty": float(qty),
                "unit_price": float(unit_price),
                "total": float(qty) * float(unit_price),
                "sort_order": int(len(df_items)),
            }
            st.session_state["items_df"] = compute_items_totals(df_items)
            st.rerun()

        df_edit = st.session_state["items_df"].copy()
        df_edit = compute_items_totals(df_edit)

        edited = st.data_editor(
            df_edit,
            use_container_width=True,
            hide_index=True,
            column_config={
                "product_service_id": st.column_config.NumberColumn("Catálogo (id)", disabled=True),
                "description": st.column_config.TextColumn("Descrição"),
                "unit": st.column_config.SelectboxColumn("Un.", options=UNITS),
                "qty": st.column_config.NumberColumn("Qtd", min_value=0.0, step=1.0),
                "unit_price": st.column_config.NumberColumn("V. Unit (R$)", min_value=0.0, step=50.0),
                "total": st.column_config.NumberColumn("Total (R$)", disabled=True),
                "sort_order": st.column_config.NumberColumn("Ord.", step=1),
            },
            key=f"items_editor_{proposal_id or 'new'}"
        )

        edited = compute_items_totals(edited)
        total_geral = float(edited["total"].sum()) if not edited.empty else 0.0

        k1, k2, k3 = st.columns([1,1,1])
        with k1:
            st.markdown(f"<div class='bk-kpi'><div class='bk-kpi-label'>Total da Proposta</div><div class='bk-kpi-value'>{money_br(total_geral)}</div></div>", unsafe_allow_html=True)
        with k2:
            if st.button("💾 Salvar proposta", use_container_width=True):
                with engine.begin() as conn:
                    pid = save_proposal(conn, {
                        "id": int(proposal_id) if proposal_id else None,
                        "code": code,
                        "title": title.strip() or "Proposta",
                        "client_id": client_id,
                        "lead_id": None,
                        "value_total": float(total_geral),
                        "status": status,
                        "valid_until": valid_until.isoformat(),
                        "notes": "",
                        "objective": objective,
                        "scope": scope,
                        "resp_contratante": resp_contratante,
                        "resp_contratado": resp_contratado,
                        "payment_terms": payment_terms,
                        "delivery_terms": delivery_terms,
                        "reference": reference,
                        "observations": observations,
                    })
                    replace_proposal_items(conn, pid, edited)
                st.success("Proposta salva.")
                st.session_state["proposal_id"] = int(pid)
                st.rerun()

        with k3:
            if st.button("🧹 Limpar itens", use_container_width=True, disabled=edited.empty):
                st.session_state["items_df"] = pd.DataFrame(columns=["product_service_id","description","unit","qty","unit_price","total","sort_order"])
                st.rerun()

        st.markdown("#### 📄 Gerar/Imprimir HTML")
        # dados do cliente
        cliente = client_options.get(client_name) if client_name != "—" else None
        cliente_doc = (cliente.document or "") if cliente else ""
        cliente_end = (cliente.notes or "") if cliente else ""

        rows = []
        for _, r in edited.sort_values("sort_order").iterrows():
            rows.append(
                "<tr>"
                f"<td>{str(r.get('description',''))}</td>"
                f"<td>{str(r.get('unit',''))}</td>"
                f"<td class='num'>{float(r.get('qty') or 0):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") + "</td>"
                f"<td class='num'>{money_br(r.get('unit_price') or 0)}</td>"
                f"<td class='num'>{money_br(r.get('total') or 0)}</td>"
                "</tr>"
            )
        itens_html = "\n".join(rows) if rows else "<tr><td colspan='5' class='muted'>Sem itens</td></tr>"

        html_context = {
            "PROPOSTA_CODIGO": code,
            "STATUS": status.upper(),
            "DATA_EMISSAO": date.today().strftime("%d/%m/%Y"),
            "VALIDADE": valid_until.strftime("%d/%m/%Y"),
            "TITULO": title.strip() or "Proposta",
            "CLIENTE_NOME": cliente.name if cliente else "—",
            "CLIENTE_DOC": cliente_doc,
            "CLIENTE_ENDERECO": cliente_end,
            "REFERENCIA": reference or "—",
            "OBSERVACOES": observations or "",
            "OBJETIVO": objective.replace("\n", "<br/>"),
            "ESCOPO": scope.replace("\n", "<br/>"),
            "RESP_CONTRATANTE": resp_contratante.replace("\n", "<br/>"),
            "RESP_CONTRATADO": resp_contratado.replace("\n", "<br/>"),
            "ITENS_TABELA": itens_html,
            "TOTAL_GERAL": money_br(total_geral),
            "PAGAMENTO": payment_terms.replace("\n", "<br/>"),
            "ENTREGA": delivery_terms.replace("\n", "<br/>"),
        }

        html = render_proposta_html(
            template_path=Path("reports/templates/proposta_BK.html"),
            logo_path=Path("assets/logo.svg"),
            context=html_context
        )

        colh1, colh2 = st.columns([1,1])
        with colh1:
            st.download_button("⬇️ Baixar HTML da Proposta", data=html.encode("utf-8"), file_name=f"{code}.html", mime="text/html", use_container_width=True)
        with colh2:
            show = st.toggle("👀 Pré-visualizar aqui", value=True)
        if show:
            st.components.v1.html(html, height=900, scrolling=True)

        st.markdown("#### 💰 Gerar conta a receber (Financeiro)")
        st.caption("Ao aprovar uma proposta, você pode criar um lançamento de conta a receber com vencimento e conta/categoria.")
        a1, a2, a3, a4 = st.columns([1,1,1,1])
        with a1:
            due = st.date_input("Vencimento", value=date.today() + timedelta(days=15))
        with a2:
            accounts = _load_accounts()
            acc_opt = {a.name: a for a in accounts}
            acc_name = st.selectbox("Conta", options=["—"] + list(acc_opt.keys()), index=0)
            account_id = acc_opt.get(acc_name).id if acc_name != "—" else None
        with a3:
            cats = _load_categories()
            cat_opt = {c.name: c for c in cats}
            cat_name = st.selectbox("Categoria", options=["—"] + list(cat_opt.keys()), index=0)
            category_id = cat_opt.get(cat_name).id if cat_name != "—" else None
        with a4:
            if st.button("Criar conta a receber", use_container_width=True, disabled=(not proposal_id or total_geral <= 0)):
                ok, msg = _create_receivable_from_proposal(int(proposal_id), f"Proposta {code} - {title}", total_geral, due, account_id, category_id)
                (st.success if ok else st.warning)(msg)


with tab_leads:
    st.markdown("### 🧲 Leads / Pipeline (visão leve)")
    st.caption("Este painel é um resumo. O detalhamento completo pode ser expandido conforme seu processo comercial.")

    with engine.begin() as conn:
        try:
            leads = pd.read_sql(text("""
                SELECT id, name, source, stage, value_estimate, created_at
                FROM leads
                ORDER BY created_at DESC, id DESC
            """), conn)
        except Exception:
            # Compatibilidade com bancos antigos (schema sem colunas do pipeline)
            leads = pd.read_sql(text("""
                SELECT id, name, '' AS source, COALESCE(status,'novo') AS stage, 0 AS value_estimate, created_at
                FROM leads
                ORDER BY created_at DESC, id DESC
            """), conn)

    if leads.empty:
        st.info("Nenhum lead cadastrado.")
    else:
        leads["valor"] = leads["value_estimate"].fillna(0).map(money_br)
        st.dataframe(leads[["id","name","source","stage","valor","created_at"]], hide_index=True, use_container_width=True)


with tab_pedidos:
    st.info("Pedidos/vendas serão evoluídos aqui (ex.: pedido, faturamento, integração com compras e projetos).")