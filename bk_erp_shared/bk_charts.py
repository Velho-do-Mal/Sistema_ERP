# bk_erp_shared/bk_charts.py
"""
BK_ERP — Gráficos modernos compartilhados.
Mesmo padrão visual do BK Planejamento Estratégico:
  - Paleta BK consistente
  - Template plotly_white customizado
  - Funções prontas para Financeiro, Projetos, Vendas, Compras
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Paleta BK ──────────────────────────────
BK_BLUE       = "#1565C0"
BK_BLUE_LIGHT = "#42A5F5"
BK_TEAL       = "#00897B"
BK_GREEN      = "#43A047"
BK_ORANGE     = "#FB8C00"
BK_RED        = "#E53935"
BK_PURPLE     = "#7B1FA2"
BK_GRAY       = "#546E7A"
BK_DARK       = "#0D1B2A"
BK_COLORS     = [BK_BLUE, BK_TEAL, BK_GREEN, BK_ORANGE, BK_RED, BK_PURPLE, BK_GRAY, BK_BLUE_LIGHT]

TEMPLATE = "plotly_white"

STATUS_COLORS = {
    "aberta":      BK_BLUE,
    "aprovacao":   BK_ORANGE,
    "aprovada":    BK_GREEN,
    "encerrada":   BK_GRAY,
    "cancelada":   BK_RED,
    "rascunho":    BK_GRAY,
    "enviado":     BK_BLUE_LIGHT,
    "aprovado":    BK_GREEN,
    "rejeitado":   BK_RED,
    "Pendente":    BK_ORANGE,
    "Em andamento":BK_BLUE,
    "Concluído":   BK_GREEN,
    "Atrasado":    BK_RED,
    "ganho":       BK_GREEN,
    "perdido":     BK_RED,
    "novo":        BK_BLUE_LIGHT,
    "proposta":    BK_BLUE,
    "contato":     BK_TEAL,
}


def _layout(fig: go.Figure, title: str = "", height: int = 380, xangle: int = -30) -> go.Figure:
    """Aplica layout padrão BK em qualquer figura Plotly."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=BK_DARK, family="Segoe UI"),
                   x=0, xanchor="left"),
        template=TEMPLATE,
        height=height,
        margin=dict(l=30, r=30, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11, family="Segoe UI")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Segoe UI"),
        font=dict(family="Segoe UI, Arial", color=BK_DARK),
        colorway=BK_COLORS,
    )
    fig.update_xaxes(tickangle=xangle, gridcolor="rgba(0,0,0,0.05)",
                     tickfont=dict(size=11, family="Segoe UI"))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)",
                     tickfont=dict(size=11, family="Segoe UI"))
    return fig


def bk_plotly(fig: go.Figure, key: str, height: Optional[int] = None) -> None:
    """Substitui st_plotly do bk_finance com key obrigatório (evita DuplicateElementId)."""
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, key=key)


# ═══════════════════════════════════════
# FINANCEIRO — gráficos
# ═══════════════════════════════════════

def fig_cashflow(df: pd.DataFrame, show_mode: str = "Comparativo") -> go.Figure:
    """
    Fluxo de caixa: barras Previsto x Realizado + linha acumulada.
    df deve ter colunas: period, previsto, realizado, cum_previsto, cum_realizado
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if show_mode in ("Comparativo (Previsto x Realizado)", "Somente Previsto", "Comparativo"):
        fig.add_trace(go.Bar(
            x=df["period"], y=df["previsto"], name="Previsto",
            marker_color=BK_BLUE_LIGHT, opacity=0.85,
        ), secondary_y=False)

    if show_mode in ("Comparativo (Previsto x Realizado)", "Somente Realizado", "Comparativo"):
        fig.add_trace(go.Bar(
            x=df["period"], y=df["realizado"], name="Realizado",
            marker_color=BK_TEAL, opacity=0.85,
        ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["period"], y=df["cum_previsto"],
        name="Acumulado Previsto", mode="lines+markers",
        line=dict(color=BK_BLUE, width=2, dash="dot"),
        marker=dict(size=6),
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df["period"], y=df["cum_realizado"],
        name="Acumulado Realizado", mode="lines+markers",
        line=dict(color=BK_GREEN, width=2),
        marker=dict(size=6, symbol="diamond"),
    ), secondary_y=True)

    fig.update_yaxes(title_text="Valor (R$)", secondary_y=False,
                     gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(title_text="Saldo Acumulado (R$)", secondary_y=True,
                     gridcolor="rgba(0,0,0,0)")

    return _layout(fig, "Fluxo de Caixa — Previsto × Realizado", height=440, xangle=-30)


def fig_breakdown_h(df: pd.DataFrame, title: str = "", top: int = 20) -> go.Figure:
    """
    Barras horizontais para breakdowns (categoria, centro de custo).
    df deve ter colunas: Item, Valor
    """
    df = df.copy()
    df["Abs"] = df["Valor"].abs()
    df = df.nlargest(top, "Abs").sort_values("Abs", ascending=True)
    colors = [BK_GREEN if v >= 0 else BK_RED for v in df["Valor"]]

    fig = go.Figure(go.Bar(
        x=df["Valor"], y=df["Item"], orientation="h",
        marker_color=colors,
        text=df["Valor"].apply(lambda v: f"R$ {v:,.0f}"),
        textposition="outside",
        textfont=dict(size=11, family="Segoe UI"),
    ))
    return _layout(fig, title, height=max(320, len(df) * 28 + 80), xangle=0)


def fig_donut_status(labels: list, values: list, title: str = "Status") -> go.Figure:
    """Donut chart para status de qualquer lista."""
    colors = [STATUS_COLORS.get(l, BK_GRAY) for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors, line=dict(color="#fff", width=2)),
        textfont=dict(size=12, family="Segoe UI"),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    total = sum(values)
    fig.add_annotation(text=f"<b>{total}</b><br>total", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=14, family="Segoe UI", color=BK_DARK))
    return _layout(fig, title, height=320, xangle=0)


def fig_saldo_contas(df: pd.DataFrame) -> go.Figure:
    """Barras de saldo por conta bancária."""
    df = df.sort_values("Saldo", ascending=True)
    colors = [BK_GREEN if v >= 0 else BK_RED for v in df["Saldo"]]

    fig = go.Figure(go.Bar(
        x=df["Saldo"], y=df["Conta"], orientation="h",
        marker_color=colors,
        text=df["Saldo"].apply(lambda v: f"R$ {v:,.2f}"),
        textposition="outside",
        textfont=dict(size=11),
    ))
    return _layout(fig, "Saldo por Conta", height=max(280, len(df) * 36 + 80), xangle=0)


def fig_entradas_saidas_mensal(df_month: pd.DataFrame) -> go.Figure:
    """
    Barras agrupadas Entradas x Saídas por mês.
    df deve ter colunas: period, entradas, saidas
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_month["period"], y=df_month.get("entradas", []),
                         name="Entradas", marker_color=BK_GREEN, opacity=0.85))
    fig.add_trace(go.Bar(x=df_month["period"], y=df_month.get("saidas", []),
                         name="Saídas", marker_color=BK_RED, opacity=0.85))
    return _layout(fig, "Entradas × Saídas por Período", height=360, xangle=-30)


# ═══════════════════════════════════════
# PROJETOS — gráficos
# ═══════════════════════════════════════

def fig_projetos_status(projetos: list[dict]) -> go.Figure:
    """Donut com status dos projetos."""
    from collections import Counter
    counts = Counter(p.get("status", "N/A") for p in projetos)
    labels = list(counts.keys())
    values = list(counts.values())
    return fig_donut_status(labels, values, "Status dos Projetos")


def fig_gantt_projetos(projetos: list[dict]) -> go.Figure:
    """Gantt de projetos: dataInicio → hoje (ou encerrado)."""
    today = date.today()
    rows = []
    for p in projetos:
        try:
            ini = datetime.strptime(str(p.get("dataInicio",""))[:10], "%Y-%m-%d")
        except Exception:
            ini = datetime.now()
        fim = datetime.now()
        rows.append(dict(
            Projeto=str(p.get("nome",""))[:35],
            Início=ini, Fim=fim,
            Status=p.get("status","N/A"),
        ))

    if not rows:
        return go.Figure()

    df = pd.DataFrame(rows).sort_values("Início")
    colors = {s: STATUS_COLORS.get(s, BK_GRAY) for s in df["Status"].unique()}
    fig = px.timeline(df, x_start="Início", x_end="Fim", y="Projeto",
                      color="Status", color_discrete_map=colors,
                      labels={"Projeto": "Projeto"})
    fig.update_yaxes(autorange="reversed")

    hoje = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=hoje, x1=hoje, y0=0, y1=1,
                  xref="x", yref="paper",
                  line=dict(color=BK_RED, width=2, dash="dash"))
    fig.add_annotation(x=hoje, y=1.02, xref="x", yref="paper",
                       text="<b>Hoje</b>", showarrow=False,
                       font=dict(color=BK_RED, size=11),
                       bgcolor="white", bordercolor=BK_RED, borderwidth=1)

    return _layout(fig, "Timeline dos Projetos", height=max(300, len(rows) * 32 + 80), xangle=-30)


def fig_curva_s(df_eap: pd.DataFrame) -> go.Figure:
    """
    Curva S: % previsto acumulado x % realizado acumulado.
    df deve ter colunas numéricas 'previsto' e 'realizado' indexadas por mês/período.
    """
    if df_eap.empty:
        return go.Figure()

    fig = go.Figure()
    if "previsto" in df_eap.columns:
        cum_prev = df_eap["previsto"].cumsum()
        tot = cum_prev.max() or 1
        fig.add_trace(go.Scatter(
            x=list(range(len(cum_prev))), y=(cum_prev / tot * 100).tolist(),
            name="Previsto %", mode="lines+markers",
            line=dict(color=BK_BLUE, width=2, dash="dot"),
            marker=dict(size=6),
        ))
    if "realizado" in df_eap.columns:
        cum_real = df_eap["realizado"].cumsum()
        tot_r = cum_real.max() or 1
        fig.add_trace(go.Scatter(
            x=list(range(len(cum_real))), y=(cum_real / tot_r * 100).tolist(),
            name="Realizado %", mode="lines+markers",
            line=dict(color=BK_GREEN, width=2),
            marker=dict(size=6, symbol="diamond"),
        ))

    return _layout(fig, "Curva S — Previsto × Realizado (%)", height=380, xangle=0)


# ═══════════════════════════════════════
# VENDAS / PROPOSTAS — gráficos
# ═══════════════════════════════════════

def fig_pipeline_leads(df_leads: pd.DataFrame) -> go.Figure:
    """Funil de leads por estágio."""
    if df_leads.empty:
        return go.Figure()

    stage_order = ["novo", "contato", "proposta", "ganho", "perdido"]
    counts = df_leads["stage"].value_counts().reindex(stage_order, fill_value=0)
    colors = [STATUS_COLORS.get(s, BK_GRAY) for s in counts.index]

    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=colors,
        text=counts.values,
        textposition="outside",
        textfont=dict(size=13, family="Segoe UI", color=BK_DARK),
    ))
    return _layout(fig, "Pipeline de Leads por Estágio", height=320, xangle=0)


def fig_propostas_status(df_prop: pd.DataFrame) -> go.Figure:
    """Donut de propostas por status."""
    if df_prop.empty:
        return go.Figure()
    counts = df_prop["status"].value_counts()
    return fig_donut_status(counts.index.tolist(), counts.values.tolist(), "Propostas por Status")


def fig_propostas_valor(df_prop: pd.DataFrame) -> go.Figure:
    """Barras de valor total por status de proposta."""
    if df_prop.empty or "value_total" not in df_prop.columns:
        return go.Figure()

    df_g = df_prop.groupby("status")["value_total"].sum().reset_index()
    df_g = df_g.sort_values("value_total", ascending=True)
    colors = [STATUS_COLORS.get(s, BK_GRAY) for s in df_g["status"]]

    fig = go.Figure(go.Bar(
        x=df_g["value_total"], y=df_g["status"], orientation="h",
        marker_color=colors,
        text=df_g["value_total"].apply(lambda v: f"R$ {v:,.0f}"),
        textposition="outside",
        textfont=dict(size=11),
    ))
    return _layout(fig, "Valor Total por Status de Proposta (R$)", height=320, xangle=0)


# ═══════════════════════════════════════
# COMPRAS — gráficos
# ═══════════════════════════════════════

def fig_compras_status(df_po: pd.DataFrame) -> go.Figure:
    """Donut de pedidos de compra por status."""
    if df_po.empty:
        return go.Figure()
    counts = df_po["status"].value_counts()
    return fig_donut_status(counts.index.tolist(), counts.values.tolist(), "Pedidos por Status")


def fig_compras_fornecedor(df_po: pd.DataFrame) -> go.Figure:
    """Barras horizontais: valor por fornecedor."""
    if df_po.empty or "value_total" not in df_po.columns:
        return go.Figure()

    col_forn = "supplier" if "supplier" in df_po.columns else \
               "supplier_id" if "supplier_id" in df_po.columns else None
    if col_forn is None:
        return go.Figure()

    df_g = df_po.groupby(col_forn)["value_total"].sum().reset_index()
    df_g.columns = ["Item", "Valor"]
    return fig_breakdown_h(df_g, "Compras por Fornecedor (R$)", top=15)


def fig_compras_mensal(df_po: pd.DataFrame) -> go.Figure:
    """Evolução mensal de pedidos de compra."""
    if df_po.empty or "order_date" not in df_po.columns:
        return go.Figure()

    df = df_po.copy()
    df["mes"] = pd.to_datetime(df["order_date"], errors="coerce").dt.to_period("M").astype(str)
    df_m = df.groupby("mes")["value_total"].sum().reset_index()
    df_m.columns = ["Mês", "Total"]

    fig = go.Figure(go.Bar(
        x=df_m["Mês"], y=df_m["Total"],
        marker_color=BK_BLUE, opacity=0.85,
        text=df_m["Total"].apply(lambda v: f"R$ {v:,.0f}"),
        textposition="outside",
        textfont=dict(size=11),
    ))
    return _layout(fig, "Valor de Compras por Mês (R$)", height=340, xangle=-30)
