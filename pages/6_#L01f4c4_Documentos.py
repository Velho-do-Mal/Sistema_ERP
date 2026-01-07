
import streamlit as st
import pandas as pd
from datetime import date
from sqlalchemy import text

from bk_erp_shared.theme import apply_theme
from bk_erp_shared.erp_db import get_finance_db, ensure_erp_tables
from bk_erp_shared.auth import login_and_guard, can_view, current_user

import bk_finance

st.set_page_config(page_title="BK_ERP - Documentos", layout="wide")
apply_theme()
ensure_erp_tables()

engine, SessionLocal = get_finance_db()
login_and_guard(SessionLocal)

st.markdown('<div class="bk-card"><div class="bk-title">Gestão de Documentos</div><div class="bk-subtitle">Central de arquivos: contratos, ART/RRT, medições, notas, propostas, anexos de obra e procedimentos.</div></div>', unsafe_allow_html=True)

session = SessionLocal()
clients = session.query(bk_finance.Client).order_by(bk_finance.Client.name.asc()).all()
suppliers = session.query(bk_finance.Supplier).order_by(bk_finance.Supplier.name.asc()).all()
session.close()

client_map = {c.id: c.name for c in clients}
sup_map = {s.id: s.name for s in suppliers}
proj_df = pd.read_sql(text("SELECT id, nome FROM projects ORDER BY id DESC"), engine)
proj_map = {int(r["id"]): (r["nome"] or f"Projeto {r['id']}") for _, r in proj_df.iterrows()}

with st.expander("➕ Enviar documento", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        title = st.text_input("Título *")
        doc_type = st.selectbox("Tipo", ["Contratos", "Doc. Empresa", "Seguros", "Doc. funcionários", "CATs", "Procedimentos", "Outros"], index=0)
        tags = st.text_input("Tags (separadas por vírgula)")
    with c2:
        project_id = st.selectbox("Projeto", options=[0]+list(proj_map.keys()), format_func=lambda i: "—" if i==0 else proj_map.get(i,""), index=0)
        client_id = st.selectbox("Cliente", options=[0]+[c.id for c in clients], format_func=lambda i: "—" if i==0 else client_map.get(i,""), index=0)
    with c3:
        supplier_id = st.selectbox("Fornecedor", options=[0]+[s.id for s in suppliers], format_func=lambda i: "—" if i==0 else sup_map.get(i,""), index=0)
        notes = st.text_area("Observações", height=80)

    up = st.file_uploader("Arquivo", type=None)

    if st.button("Salvar documento", type="primary"):
        if not can_view("financeiro"):
            st.error("Sem permissão para salvar.")
        elif not title.strip():
            st.error("Título é obrigatório.")
        elif up is None:
            st.error("Selecione um arquivo.")
        else:
            data = up.getvalue()
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO documents (title, doc_type, project_id, client_id, supplier_id, uploaded_by, filename, content_type, data, tags, notes)
                    VALUES (:title,:doc_type,:project_id,:client_id,:supplier_id,:uploaded_by,:filename,:content_type,:data,:tags,:notes)
                """), dict(
                    title=title.strip(),
                    doc_type=doc_type,
                    project_id=int(project_id) if project_id else None,
                    client_id=int(client_id) if client_id else None,
                    supplier_id=int(supplier_id) if supplier_id else None,
                    uploaded_by=current_user().get("email"),
                    filename=up.name,
                    content_type=up.type or "application/octet-stream",
                    data=data,
                    tags=tags.strip() or None,
                    notes=notes.strip() or None
                ))
            st.success("Documento salvo.")
            st.rerun()

st.markdown("---")

df = pd.read_sql(text("""
    SELECT id, title, doc_type, project_id, client_id, supplier_id, uploaded_by, uploaded_at, filename, content_type, tags
    FROM documents
    ORDER BY id DESC
"""), engine)

if df.empty:
    st.info("Nenhum documento cadastrado.")
else:
    df["Projeto"] = df["project_id"].map(lambda i: proj_map.get(int(i),"—") if pd.notna(i) else "—")
    df["Cliente"] = df["client_id"].map(lambda i: client_map.get(i,"—") if pd.notna(i) else "—")
    df["Fornecedor"] = df["supplier_id"].map(lambda i: sup_map.get(i,"—") if pd.notna(i) else "—")
    st.dataframe(df[["id","title","doc_type","Projeto","Cliente","Fornecedor","uploaded_by","uploaded_at","filename","tags"]], use_container_width=True, height=420)

    st.markdown("### Download / Excluir")
    c1, c2 = st.columns(2)
    with c1:
        sel_id = st.number_input("ID para download", min_value=0, step=1, value=0)
        if st.button("Carregar arquivo"):
            if sel_id:
                row = pd.read_sql(text("SELECT filename, content_type, data FROM documents WHERE id=:id"), engine, params={"id": int(sel_id)})
                if row.empty:
                    st.error("Documento não encontrado.")
                else:
                    fn = row.iloc[0]["filename"]
                    ct = row.iloc[0]["content_type"]
                    data = row.iloc[0]["data"]
                    st.download_button("⬇️ Baixar", data=data, file_name=fn, mime=ct)
            else:
                st.warning("Informe um ID.")

    with c2:
        del_id = st.number_input("ID para excluir", min_value=0, step=1, value=0, key="doc_del")
        if st.button("Excluir documento", type="secondary"):
            if not can_view("financeiro"):
                st.error("Sem permissão.")
            elif del_id:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM documents WHERE id=:id"), {"id": int(del_id)})
                st.success("Excluído.")
                st.rerun()
