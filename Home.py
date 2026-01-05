
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta

from bk_erp_shared.theme import apply_theme, load_svg
from bk_erp_shared.erp_db import get_finance_db, ensure_erp_tables
from bk_erp_shared.auth import login_and_guard, current_user

# Finance models
import bk_finance
from sqlalchemy import text

st.set_page_config(page_title="BK_ERP", layout="wide")

apply_theme()
ensure_erp_tables()

engine, SessionLocal = get_finance_db()
login_and_guard(SessionLocal)

# Header
logo_uri = load_svg("assets/logo.svg")
hero_uri = load_svg("assets/hero.svg")

st.markdown(
    f"""
    <div class="bk-card" style="padding:22px 24px; display:flex; gap:16px; align-items:center; overflow:hidden;">
        <img src="{logo_uri}" style="width:54px; height:54px;" />
        <div style="flex:1;">
            <div class="bk-title">BK_ERP</div>
            <div class="bk-subtitle">ERP para Engenharia, Empreiteiras e Construtoras • Financeiro • Projetos • Compras • Vendas • Documentos</div>
        </div>
        <div style="font-size:12px; color:#64748b; text-align:right;">
            Usuário: <b>{current_user().get("email")}</b><br/>
            Perfil: <b>{current_user().get("role")}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="bk-card" style="padding:0; overflow:hidden; margin-top:14px;">
        <img src="{hero_uri}" style="width:100%; display:block;"/>
    </div>
    """,
    unsafe_allow_html=True
)

# ---- Helper queries
def q_one(sql, params=None):
    with engine.connect() as conn:
        r = conn.execute(text(sql), params or {}).fetchone()
        return r[0] if r else 0

def q_df(sql, params=None):
    return pd.read_sql(text(sql), engine, params=params or {})

today = date.today()
win_end = today + timedelta(days=15)
month_start = today.replace(day=1)
month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)

# ---- KPIs Financeiro
receber_15 = q_one(
    """
    SELECT COALESCE(SUM(amount),0)
    FROM transactions
    WHERE paid = FALSE
      AND type = 'entrada'
      AND due_date IS NOT NULL
      AND due_date BETWEEN :d1 AND :d2
    """,
    {"d1": today, "d2": win_end},
)
pagar_15 = q_one(
    """
    SELECT COALESCE(SUM(amount),0)
    FROM transactions
    WHERE paid = FALSE
      AND type = 'saida'
      AND due_date IS NOT NULL
      AND due_date BETWEEN :d1 AND :d2
    """,
    {"d1": today, "d2": win_end},
)
overdue = q_one(
    """
    SELECT COUNT(*)
    FROM transactions
    WHERE paid = FALSE
      AND due_date IS NOT NULL
      AND due_date < :d1
    """,
    {"d1": today},
)

receitas_mes = q_one(
    """
    SELECT COALESCE(SUM(amount),0)
    FROM transactions
    WHERE paid = TRUE
      AND type = 'entrada'
      AND paid_date BETWEEN :d1 AND :d2
    """,
    {"d1": month_start, "d2": month_end},
)
despesas_mes = q_one(
    """
    SELECT COALESCE(SUM(amount),0)
    FROM transactions
    WHERE paid = TRUE
      AND type = 'saida'
      AND paid_date BETWEEN :d1 AND :d2
    """,
    {"d1": month_start, "d2": month_end},
)

# ---- KPIs Projetos
proj_abertos = q_one("SELECT COUNT(*) FROM projects WHERE encerrado = FALSE")
proj_encerrados = q_one("SELECT COUNT(*) FROM projects WHERE encerrado = TRUE")

# ---- KPIs Compras / Vendas / Documentos
compras_abertas = q_one("SELECT COUNT(*) FROM purchase_orders WHERE status != 'encerrada'")
propostas_abertas = q_one("SELECT COUNT(*) FROM proposals WHERE status != 'encerrada'")
docs_total = q_one("SELECT COUNT(*) FROM documents")

# ---- Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="bk-card">', unsafe_allow_html=True)
    st.metric("A Receber (15 dias)", bk_finance.format_currency(float(receber_15)))
    st.caption("Contas a receber com vencimento nos próximos 15 dias")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="bk-card">', unsafe_allow_html=True)
    st.metric("A Pagar (15 dias)", bk_finance.format_currency(float(pagar_15)))
    st.caption("Contas a pagar com vencimento nos próximos 15 dias")
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="bk-card">', unsafe_allow_html=True)
    st.metric("Atrasados", int(overdue))
    st.caption("Títulos vencidos e não realizados")
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.markdown('<div class="bk-card">', unsafe_allow_html=True)
    st.metric("Projetos abertos", int(proj_abertos))
    st.caption(f"Encerrados: {int(proj_encerrados)}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Gráficos
st.markdown("## 📊 Painel executivo")

g1, g2 = st.columns([1.25, 1])
with g1:
    st.markdown('<div class="bk-card">', unsafe_allow_html=True)
    st.subheader("Fluxo de caixa (Previsto x Realizado) - últimos 9 meses")
    # usa função do financeiro (série)
    session = SessionLocal()
    try:
        start = bk_finance.month_add(month_start, -8)
        df_flow = bk_finance.build_cashflow_series(session, start, month_end, "Monthly")
    finally:
        session.close()

    if df_flow.empty:
        st.caption("Cadastre lançamentos para visualizar.")
    else:
        fig = px.bar(df_flow, x="period", y=["previsto", "realizado"], barmode="group")
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with g2:
    st.markdown('<div class="bk-card">', unsafe_allow_html=True)
    st.subheader("Mês atual")
    st.metric("Receitas (realizado)", bk_finance.format_currency(float(receitas_mes)))
    st.metric("Despesas (realizado)", bk_finance.format_currency(float(despesas_mes)))
    st.metric("Resultado (mês)", bk_finance.format_currency(float(receitas_mes - despesas_mes)))
    st.caption("Baseado em pagamentos/recebimentos realizados")
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Listas rápidas
l1, l2 = st.columns(2)
with l1:
    st.markdown('<div class="bk-card">', unsafe_allow_html=True)
    st.subheader("Contas a vencer (15 dias)")
    df = q_df(
        """
        SELECT due_date AS "Vencimento", description AS "Descrição", type AS "Tipo", amount AS "Valor"
        FROM transactions
        WHERE paid = FALSE
          AND due_date IS NOT NULL
          AND due_date BETWEEN :d1 AND :d2
        ORDER BY due_date ASC
        LIMIT 20
        """,
        {"d1": today, "d2": win_end},
    )
    if df.empty:
        st.caption("Sem registros.")
    else:
        df["Valor"] = df["Valor"].map(lambda v: bk_finance.format_currency(float(v)))
        df["Tipo"] = df["Tipo"].map(lambda t: "A Receber" if t=="entrada" else "A Pagar")
        st.dataframe(df, use_container_width=True, height=320)
    st.markdown("</div>", unsafe_allow_html=True)

with l2:
    st.markdown('<div class="bk-card">', unsafe_allow_html=True)
    st.subheader("Resumo operacional")
    st.metric("Compras em aberto", int(compras_abertas))
    st.metric("Propostas em andamento", int(propostas_abertas))
    st.metric("Documentos cadastrados", int(docs_total))
    st.caption("Abra os módulos no menu lateral do Streamlit")
    st.markdown("</div>", unsafe_allow_html=True)
