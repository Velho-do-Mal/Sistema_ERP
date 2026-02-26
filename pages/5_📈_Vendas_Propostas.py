import streamlit as st
import pandas as pd
import re
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import text

from bk_erp_shared.theme import apply_theme
from bk_erp_shared.erp_db import get_finance_db, ensure_erp_tables
from bk_erp_shared.auth import login_and_guard, can_view, current_user
from bk_erp_shared.theme import bk_section, bk_kpi_row
try:
    from bk_erp_shared.bk_charts import (
        fig_pipeline_leads, fig_propostas_status, fig_propostas_valor, bk_plotly,
    )
    HAS_BK_CHARTS = True
except Exception:
    HAS_BK_CHARTS = False

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

def _next_code_from_existing(df_prop: pd.DataFrame) -> str:
    """Gera o próximo código BK-PROP-AAAA-NN (sempre +1)."""
    year = date.today().year
    prefix = f"BK-PROP-{year}-"
    if df_prop is None or getattr(df_prop, 'empty', True) or 'code' not in df_prop.columns:
        return f"{prefix}01"
    max_num = None
    for c in df_prop['code'].dropna().astype(str).tolist():
        if not c.startswith(prefix):
            continue
        suf = c[len(prefix):].strip()
        if suf.isdigit():
            n = int(suf)
            if (max_num is None) or (n > max_num):
                max_num = n
    if max_num is None:
        return f"{prefix}01"
    return f"{prefix}{max_num + 1:02d}"




def _load_projects():
    with engine.begin() as conn:
        try:
            df = pd.read_sql(text("SELECT id, COALESCE(nome,'(sem nome)') AS nome FROM projects ORDER BY id DESC"), conn)
        except Exception:
            df = pd.DataFrame(columns=["id", "nome"])
    return df


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


def delete_proposal(conn, proposal_id: int) -> None:
    """Remove proposta e seus itens (FK). Deve ser chamada dentro de engine.begin()."""
    conn.execute(text("DELETE FROM proposal_items WHERE proposal_id = :pid"), {"pid": proposal_id})
    conn.execute(text("DELETE FROM proposals WHERE id = :pid"), {"pid": proposal_id})


def _create_receivable_from_proposal(proposal_id: int, description: str, amount: float, due: date,
                                     account_id: int | None, category_id: int | None,
                                     client_id: int | None = None, project_id: int | None = None):
    ref = f"PROPOSAL:{proposal_id}"
    with engine.begin() as conn:
        exists = conn.execute(text("SELECT id FROM transactions WHERE reference=:ref LIMIT 1"), {"ref": ref}).first()
        if exists:
            return False, "Já existe lançamento criado a partir desta proposta."
        conn.execute(
            text("""
                INSERT INTO transactions (date, due_date, paid, description, amount, type, account_id, category_id, client_id, project_id, reference)
                VALUES (:date, :due_date, FALSE, :description, :amount, 'entrada', :account_id, :category_id, :client_id, :project_id, :reference)
            """),
            {
                "date": due,
                "due_date": due,
                "description": description,
                "amount": float(amount),
                "account_id": int(account_id) if account_id else None,
                "category_id": int(category_id) if category_id else None,
                "client_id": int(client_id) if client_id else None,
                "project_id": int(project_id) if project_id else None,
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
            st.dataframe(df_view[["id","code","title","project_name","client_name","status","valor","created_at"]], hide_index=True, use_container_width=True)
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

        df_projects = _load_projects()
        project_options = {str(r['nome']): int(r['id']) for _, r in df_projects.iterrows()} if not df_projects.empty else {}

        # seed
        if "items_df" not in st.session_state:
            st.session_state["items_df"] = pd.DataFrame(columns=["product_service_id","description","unit","qty","unit_price","total","sort_order"])
        if "items_editor_v" not in st.session_state:
            st.session_state["items_editor_v"] = 0

        if proposal_id:
            # Evita sobrescrever itens inseridos no front a cada rerun.
            if st.session_state.get("_loaded_proposal_id") != int(proposal_id):
                with engine.begin() as conn:
                    data = get_proposal(conn, int(proposal_id))
                p = data["proposal"]
                items = data["items"]
                st.session_state["items_df"] = items[["product_service_id","description","unit","qty","unit_price","total","sort_order"]].copy()
                st.session_state["proposal_form_seed"] = dict(p)
                st.session_state["_loaded_proposal_id"] = int(proposal_id)
                st.session_state["items_editor_v"] = 0  # reset ao abrir nova proposta
            else:
                p = st.session_state.get("proposal_form_seed", {}) or {}
        else:
            st.session_state["_loaded_proposal_id"] = None

            p = st.session_state.get("proposal_form_seed", {}) or {}
            if not p.get("code"):
                    p["code"] = _next_code_from_existing(df_prop)
            p.setdefault("title", "")
            p.setdefault("status", "rascunho")
            p.setdefault("client_id", None)
            p.setdefault("project_id", None)
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
            code = st.text_input("Código", value=p.get("code",""), disabled=bool(proposal_id))
        with f2:
            title = st.text_input("Título", value=p.get("title",""))
        with f3:
            status = st.selectbox("Status", ["rascunho","enviada","aprovada","rejeitada","faturada"], index=["rascunho","enviada","aprovada","rejeitada","faturada"].index(p.get("status","rascunho")))

        c1, c2, c3, c4 = st.columns([2,2,1,1])
        with c1:
            client_name = st.selectbox("Cliente", options=["—"] + list(client_options.keys()),
                                       index=0 if not p.get("client_id") else 1)
            client_id = client_options.get(client_name).id if client_name != "—" else None
        with c2:
            # Seleção de projeto (opcional)
            project_names = list(project_options.keys())
            current_pid = p.get("project_id")
            current_pname = None
            if current_pid:
                for _name, _pid in project_options.items():
                    if int(_pid) == int(current_pid):
                        current_pname = _name
                        break
            project_index = 0
            if current_pname and current_pname in project_names:
                project_index = 1 + project_names.index(current_pname)

            project_name = st.selectbox(
                "Projeto (opcional)",
                options=["—"] + project_names,
                index=project_index,
            )
            project_id = project_options.get(project_name) if project_name != "—" else None
        with c3:
            valid_until = st.date_input("Validade até", value=date.fromisoformat(p.get("valid_until") or date.today().isoformat()))
        with c4:
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
            st.session_state["items_editor_v"] = st.session_state.get("items_editor_v", 0) + 1
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
            key=f"items_editor_{proposal_id or 'new'}_v{st.session_state.get('items_editor_v', 0)}"
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
                        "project_id": project_id,
                        "project_id": project_id,
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

            # Excluir somente 1 item do orçamento
            if not edited.empty:
                _labels = [f"{i+1} - {str(row.get('description','')).strip() or '(sem descrição)'}" for i, row in edited.reset_index(drop=True).iterrows()]
                _sel = st.selectbox("Item para excluir", ["—"] + _labels, index=0, key="del_item_sel")
                if st.button("🗑️ Excluir item selecionado", use_container_width=True, disabled=_sel == "—", key="btn_del_item"):
                    _idx = int(_sel.split(" - ", 1)[0]) - 1
                    _new_df = edited.drop(index=_idx).reset_index(drop=True)
                    st.session_state["items_df"] = _new_df
                    st.session_state["items_editor_v"] = st.session_state.get("items_editor_v", 0) + 1
                    st.rerun()

        # Excluir um item específico (além do limpar tudo)
        if not edited.empty:
            delc1, delc2 = st.columns([3,1])
            with delc1:
                _del_i = st.selectbox(
                    "Excluir item",
                    options=list(range(len(edited))),
                    format_func=lambda i: f"{int(edited.iloc[i].get('sort_order', i))} • {str(edited.iloc[i].get('description',''))}"
                )
            with delc2:
                if st.button("🗑️ Excluir item", use_container_width=True):
                    df_new = edited.drop(index=_del_i).reset_index(drop=True)
                    if not df_new.empty:
                        df_new["sort_order"] = list(range(len(df_new)))
                    st.session_state["items_df"] = compute_items_totals(df_new)
                    st.session_state["items_editor_v"] = st.session_state.get("items_editor_v", 0) + 1
                    st.rerun()



        # ── Excluir proposta selecionada ─────────────────────────
        if proposal_id:
            st.divider()
            with st.expander("🗑️ Excluir proposta selecionada", expanded=False):
                st.warning(
                    f"⚠️ Esta ação é **irreversível**. Todos os itens da proposta **{code}** "
                    f"— *{title or 'sem título'}* — serão removidos permanentemente."
                )
                _confirm_delete = st.checkbox(
                    "Confirmo que desejo excluir esta proposta.",
                    value=False,
                    key="confirm_delete_proposal",
                )
                if st.button(
                    "🗑️ Excluir proposta",
                    use_container_width=True,
                    disabled=not _confirm_delete,
                    type="primary",
                    key="btn_delete_proposal",
                ):
                    with engine.begin() as conn:
                        delete_proposal(conn, int(proposal_id))
                    st.success(f"Proposta {code} excluída com sucesso.")
                    st.session_state["proposal_id"] = None
                    st.session_state["_loaded_proposal_id"] = None
                    st.session_state["items_editor_v"] = 0
                    st.session_state["items_df"] = pd.DataFrame(
                        columns=["product_service_id","description","unit","qty","unit_price","total","sort_order"]
                    )
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
                ok, msg = _create_receivable_from_proposal(int(proposal_id), f"Proposta {code} - {title}", total_geral, due, account_id, category_id, client_id=client_id, project_id=project_id)
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

# ════════════════════════════════════════
# PAINEL ANALÍTICO — Vendas & Propostas
# (injetado após as abas existentes)
# ════════════════════════════════════════
st.markdown("---")
bk_section("📊 Painel Analítico de Vendas")

try:
    with engine.begin() as _conn:
        df_prop_all = pd.read_sql(
            text("SELECT id, code, title, status, value_total, created_at FROM proposals ORDER BY id DESC"),
            _conn
        )
        df_leads_all = pd.read_sql(
            text("SELECT id, name, company, status, stage, value_estimate FROM leads ORDER BY id DESC"),
            _conn
        )

    if df_prop_all.empty and df_leads_all.empty:
        st.info("Nenhuma proposta ou lead cadastrada ainda.")
    else:
        # KPIs
        total_prop = df_prop_all["value_total"].sum() if not df_prop_all.empty else 0
        ganhas = len(df_prop_all[df_prop_all["status"] == "aprovada"]) if not df_prop_all.empty else 0
        n_leads = len(df_leads_all)
        leads_ganhos = len(df_leads_all[df_leads_all.get("stage", df_leads_all.get("status","")) == "ganho"]) if not df_leads_all.empty else 0

        bk_kpi_row([
            ("Propostas", len(df_prop_all), "blue"),
            ("Valor Total", f"R$ {total_prop:,.0f}".replace(",", "."), "teal"),
            ("Aprovadas", ganhas, "green"),
            ("Leads", n_leads, "orange"),
            ("Leads Ganhos", leads_ganhos, "green"),
        ])

        if HAS_BK_CHARTS:
            col1, col2 = st.columns(2)
            with col1:
                if not df_prop_all.empty:
                    bk_plotly(fig_propostas_status(df_prop_all), key="vnd_prop_status")
                else:
                    st.info("Nenhuma proposta ainda.")
            with col2:
                if not df_prop_all.empty:
                    bk_plotly(fig_propostas_valor(df_prop_all), key="vnd_prop_valor")
                else:
                    st.info("Nenhuma proposta ainda.")

            if not df_leads_all.empty:
                stage_col = "stage" if "stage" in df_leads_all.columns else "status"
                df_leads_all["stage"] = df_leads_all.get(stage_col, df_leads_all.get("status",""))
                bk_plotly(fig_pipeline_leads(df_leads_all), key="vnd_pipeline_leads")
except Exception as _e:
    st.info(f"Painel analítico indisponível: {_e}")
