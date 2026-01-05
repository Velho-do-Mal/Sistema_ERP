import streamlit as st
import pandas as pd
from datetime import date
from pathlib import Path
from bk_erp_shared.theme import apply_theme
from bk_erp_shared.auth import login_and_guard, can_view
from bk_erp_shared.erp_db import ensure_erp_tables, get_finance_db
from bk_erp_shared.boletim_medicoes import (
    list_projects,
    list_contract_items,
    upsert_contract_items,
    list_measurements,
    create_measurement,
    get_measurement_header,
    get_measurement_items_with_context,
    save_measurement_items,
    set_measurement_status,
    measurement_summary,
)
from bk_erp_shared.render_boletim_medicao import render_boletim_html
st.set_page_config(page_title="BK_ERP • Boletim de Medições", layout="wide")
apply_theme()
engine, SessionLocal = get_finance_db()
login_and_guard(SessionLocal)
ensure_erp_tables()
if not can_view("medicoes"):
    st.error("Você não tem permissão para acessar o módulo Boletim de Medições.")
    st.stop()
st.title("📏 Boletim de Medições")
st.caption("Controle de medições por projeto: itens do contrato, boletins por período, consolidação e relatório BK.")
projects = list_projects(active_only=True)
if projects.empty:
    st.info("Nenhum projeto encontrado. Cadastre/importe um projeto primeiro.")
    st.stop()
proj_map = {f"#{int(r.id)} • {r.nome}": int(r.id) for r in projects.itertuples(index=False)}
proj_label = st.selectbox("Selecione o projeto", list(proj_map.keys()), index=0)
project_id = proj_map[proj_label]
project_name = proj_label.split("•", 1)[1].strip() if "•" in proj_label else proj_label
tab_itens, tab_boletins, tab_rel = st.tabs(["📋 Itens do Contrato", "📏 Boletins", "📑 Relatório"])
with tab_itens:
    st.subheader("Itens do Contrato (por projeto)")
    st.caption("Base para medições. Você pode ajustar unidade, quantidade contratada e valor unitário.")
    items = list_contract_items(project_id)
    if items.empty:
        st.info("Este projeto ainda não possui itens do contrato. Adicione abaixo.")
        items = pd.DataFrame(columns=["id", "item", "atividade_geral", "unidade", "qtde_contratada", "valor_unit", "ativo"])
    editable = items[["id", "item", "atividade_geral", "unidade", "qtde_contratada", "valor_unit", "ativo"]].copy()
    edited = st.data_editor(
        editable,
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "item": st.column_config.TextColumn("Item", required=True),
            "atividade_geral": st.column_config.TextColumn("Atividade (opcional)"),
            "unidade": st.column_config.TextColumn("Unidade", help="ex.: un, m, m², m³, kg, hh, vb"),
            "qtde_contratada": st.column_config.NumberColumn("Qtde Contratada", format="%.2f"),
            "valor_unit": st.column_config.NumberColumn("Valor Unitário", format="%.2f"),
            "ativo": st.column_config.CheckboxColumn("Ativo"),
        },
        key="contract_items_editor",
    )
    col_a, col_b, col_c = st.columns([1, 1, 3])
    with col_a:
        if st.button("💾 Salvar itens do contrato", type="primary"):
            upsert_contract_items(project_id, edited)
            st.success("Itens salvos.")
            st.rerun()
    with col_b:
        st.download_button(
            "⬇️ Exportar CSV",
            data=edited.to_csv(index=False).encode("utf-8"),
            file_name=f"itens_contrato_projeto_{project_id}.csv",
            mime="text/csv",
        )
with tab_boletins:
    st.subheader("Boletins por período")
    st.caption("Crie um boletim e lance as quantidades medidas no período. O sistema calcula acumulados e saldos.")
    col_new1, col_new2, col_new3, col_new4 = st.columns([1, 1, 2, 1])
    with col_new1:
        ps = st.date_input("Início do período", value=date.today().replace(day=1), key="bm_ps")
    with col_new2:
        pe = st.date_input("Fim do período", value=date.today(), key="bm_pe")
    with col_new3:
        ref = st.text_input("Referência (ex.: Medição 2026-01)", value=f"Medição {date.today():%Y-%m}", key="bm_ref")
    with col_new4:
        if st.button("➕ Criar boletim", type="primary"):
            mid = create_measurement(project_id, ps, pe, ref)
            st.success(f"Boletim criado (ID {mid}).")
            st.rerun()
    st.divider()
    bms = list_measurements(project_id)
    if bms.empty:
        st.info("Nenhum boletim criado para este projeto ainda.")
        st.stop()
    bm_labels = []
    bm_map = {}
    for r in bms.itertuples(index=False):
        label = f"#{int(r.id)} • {r.reference or ''} • {r.period_start or ''} → {r.period_end or ''} • {r.status}"
        bm_labels.append(label)
        bm_map[label] = int(r.id)
    selected = st.selectbox("Selecione um boletim", bm_labels, index=0)
    measurement_id = bm_map[selected]
    header = get_measurement_header(measurement_id)
    summary = measurement_summary(measurement_id)
    items_ctx = get_measurement_items_with_context(measurement_id)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Contrato", f"R$ {summary['contrato']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    c2.metric("Medição no período", f"R$ {summary['periodo']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    c3.metric("Acumulado", f"R$ {summary['acumulado']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    c4.metric("Saldo", f"R$ {summary['saldo']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    st.divider()
    if items_ctx.empty:
        st.warning("Este boletim não possui itens. Verifique os itens do contrato e recrie o boletim.")
        st.stop()
    edit_cols = ["measurement_item_id","contract_item_id","descricao","unidade","qtde_contratada","valor_unit","qtde_periodo",
                "qtde_acumulada_anterior","qtde_acumulada","qtde_saldo","valor_periodo","valor_acumulado","valor_saldo"]
    show = items_ctx[edit_cols].copy()
    edited_bm = st.data_editor(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "measurement_item_id": st.column_config.NumberColumn("ID Item", disabled=True),
            "contract_item_id": st.column_config.NumberColumn("ID Contrato", disabled=True),
            "descricao": st.column_config.TextColumn("Descrição", disabled=True),
            "unidade": st.column_config.TextColumn("Un", disabled=True),
            "qtde_contratada": st.column_config.NumberColumn("Qtde Contr.", disabled=True, format="%.2f"),
            "valor_unit": st.column_config.NumberColumn("Vlr Unit", format="%.2f"),
            "qtde_periodo": st.column_config.NumberColumn("Qtde Período", format="%.2f"),
            "qtde_acumulada_anterior": st.column_config.NumberColumn("Acum. Anterior", disabled=True, format="%.2f"),
            "qtde_acumulada": st.column_config.NumberColumn("Acum. (c/ período)", disabled=True, format="%.2f"),
            "qtde_saldo": st.column_config.NumberColumn("Saldo Qtde", disabled=True, format="%.2f"),
            "valor_periodo": st.column_config.NumberColumn("Vlr Período", disabled=True, format="%.2f"),
            "valor_acumulado": st.column_config.NumberColumn("Vlr Acum.", disabled=True, format="%.2f"),
            "valor_saldo": st.column_config.NumberColumn("Saldo R$", disabled=True, format="%.2f"),
        },
        key="bm_items_editor",
    )
    col_s1, col_s2, col_s3, col_s4 = st.columns([1, 1, 1, 2])
    with col_s1:
        if st.button("💾 Salvar boletim", type="primary"):
            save_measurement_items(measurement_id, edited_bm)
            st.success("Boletim salvo.")
            st.rerun()
    with col_s2:
        status = st.selectbox("Status", ["rascunho", "enviado", "aprovado", "rejeitado"], index=["rascunho","enviado","aprovado","rejeitado"].index((header.get("status") or "rascunho").lower()))
    with col_s3:
        resp = st.selectbox("Responsável por atraso", ["N/A", "BK", "CLIENTE"], index=["N/A","BK","CLIENTE"].index((header.get("delay_responsibility") or "N/A").upper() if (header.get("delay_responsibility") or "N/A").upper() in ["N/A","BK","CLIENTE"] else "N/A"))
    with col_s4:
        notes = st.text_input("Observações", value=str(header.get("notes") or ""), placeholder="Ex.: aguardando documentos do cliente, ajustes em escopo...", key="bm_notes")
    if st.button("✅ Aplicar status / observações"):
        set_measurement_status(measurement_id, status=status, delay_responsibility=resp, notes=notes)
        st.success("Atualizado.")
        st.rerun()
    st.divider()
    # Gráficos
    st.subheader("Gráficos do boletim")
    plot_df = get_measurement_items_with_context(measurement_id)
    plot_df = plot_df.sort_values("valor_periodo", ascending=False)
    st.bar_chart(plot_df.set_index("descricao")["valor_periodo"], use_container_width=True)
    contrato = summary["contrato"] or 0.0
    perc = (summary["acumulado"] / contrato * 100.0) if contrato > 0 else 0.0
    st.progress(min(max(perc / 100.0, 0.0), 1.0), text=f"Progresso financeiro do contrato (acumulado): {perc:.1f}%")
with tab_rel:
    st.subheader("Relatório BK (HTML)")
    st.caption("Gera relatório no padrão do template BK e permite baixar em HTML.")
    bms = list_measurements(project_id)
    bm_labels = []
    bm_map = {}
    for r in bms.itertuples(index=False):
        label = f"#{int(r.id)} • {r.reference or ''} • {r.period_start or ''} → {r.period_end or ''} • {r.status}"
        bm_labels.append(label)
        bm_map[label] = int(r.id)
    sel = st.selectbox("Selecione o boletim para gerar relatório", bm_labels, index=0, key="bm_report_select")
    mid = bm_map[sel]
    header = get_measurement_header(mid)
    items_ctx = get_measurement_items_with_context(mid)
    summary = measurement_summary(mid)
    # Caminho do template dentro do projeto
    tpl = Path("reports/templates/boletim_medicao_BK.html")
    if not tpl.exists():
        # fallback: tenta ao lado do arquivo (quando executado isolado)
        tpl = Path(__file__).resolve().parent.parent / "reports" / "templates" / "boletim_medicao_BK.html"
    html = render_boletim_html(
        template_path=tpl,
        header=header,
        project_name=project_name,
        items=items_ctx,
        summary=summary,
    )
    st.download_button(
        "⬇️ Baixar HTML",
        data=html.encode("utf-8"),
        file_name=f"boletim_medicao_projeto_{project_id}_bm_{mid}.html",
        mime="text/html",
    )
    with st.expander("👀 Pré-visualizar relatório"):
        st.components.v1.html(html, height=900, scrolling=True)