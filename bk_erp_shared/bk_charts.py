# bk_erp_shared/bk_charts.py
"""
BK_ERP — Gráficos dark de alta qualidade.
Template dark, glassmorphism, fontes Inter, hover rico, grid sutil.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Paleta BK Dark ──────────────────────────
BK_BLUE       = "#3B82F6"
BK_BLUE_LIGHT = "#93C5FD"
BK_BLUE_DARK  = "#1E40AF"
BK_TEAL       = "#14B8A6"
BK_GREEN      = "#10B981"
BK_ORANGE     = "#F59E0B"
BK_RED        = "#EF4444"
BK_PURPLE     = "#8B5CF6"
BK_GRAY       = "#64748B"
BK_DARK       = "#0F172A"
BK_SURFACE    = "#1E293B"
BK_BORDER     = "#334155"
BK_TEXT       = "#F1F5F9"
BK_TEXT_MUTED = "#94A3B8"
BK_COLORS     = [BK_BLUE, BK_TEAL, BK_GREEN, BK_ORANGE, BK_RED, BK_PURPLE, BK_GRAY, BK_BLUE_LIGHT]

STATUS_COLORS = {
    "aberta":       BK_BLUE,
    "aprovacao":    BK_ORANGE,
    "aprovada":     BK_GREEN,
    "encerrada":    BK_GRAY,
    "cancelada":    BK_RED,
    "rascunho":     BK_GRAY,
    "enviado":      BK_BLUE_LIGHT,
    "aprovado":     BK_GREEN,
    "rejeitado":    BK_RED,
    "Pendente":     BK_ORANGE,
    "Em andamento": BK_BLUE,
    "Concluído":    BK_GREEN,
    "Atrasado":     BK_RED,
    "ganho":        BK_GREEN,
    "perdido":      BK_RED,
    "novo":         BK_BLUE_LIGHT,
    "proposta":     BK_BLUE,
    "contato":      BK_TEAL,
}


def _layout(fig: go.Figure, title: str = "", height: int = 380, xangle: int = -30) -> go.Figure:
    """Layout padrão BK Dark em qualquer figura Plotly."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=15, color=BK_BLUE_LIGHT, family="Inter, Segoe UI"),
            x=0, xanchor="left",
        ),
        height=height,
        margin=dict(l=40, r=30, t=55, b=50),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=11, color=BK_TEXT, family="Inter"),
            bgcolor="rgba(30,41,59,0.7)",
            bordercolor=BK_BORDER, borderwidth=1,
        ),
        plot_bgcolor=BK_SURFACE,
        paper_bgcolor=BK_DARK,
        hoverlabel=dict(
            bgcolor=BK_SURFACE, font_size=12, font_family="Inter",
            bordercolor=BK_BORDER,
        ),
        font=dict(family="Inter, Segoe UI", color=BK_TEXT),
        colorway=BK_COLORS,
    )
    fig.update_xaxes(
        tickangle=xangle,
        gridcolor="rgba(51,65,85,0.5)",
        linecolor=BK_BORDER,
        tickfont=dict(size=11, color=BK_TEXT_MUTED, family="Inter"),
        title_font=dict(color=BK_TEXT_MUTED),
    )
    fig.update_yaxes(
        gridcolor="rgba(51,65,85,0.5)",
        linecolor=BK_BORDER,
        tickfont=dict(size=11, color=BK_TEXT_MUTED, family="Inter"),
        title_font=dict(color=BK_TEXT_MUTED),
    )
    return fig


def bk_plotly(fig: go.Figure, key: str, height: Optional[int] = None) -> None:
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, key=key)


# ═══════════════════════════════════════
# FINANCEIRO
# ═══════════════════════════════════════

def fig_cashflow(df: pd.DataFrame, show_mode: str = "Comparativo") -> go.Figure:
    """Fluxo de caixa: barras + linhas acumuladas em eixo secundário."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if show_mode in ("Comparativo (Previsto x Realizado)", "Somente Previsto", "Comparativo"):
        fig.add_trace(go.Bar(
            x=df["period"], y=df["previsto"], name="Previsto",
            marker=dict(color=BK_BLUE, opacity=0.8, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Previsto: R$ %{y:,.2f}<extra></extra>",
        ), secondary_y=False)

    if show_mode in ("Comparativo (Previsto x Realizado)", "Somente Realizado", "Comparativo"):
        fig.add_trace(go.Bar(
            x=df["period"], y=df["realizado"], name="Realizado",
            marker=dict(color=BK_TEAL, opacity=0.8, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Realizado: R$ %{y:,.2f}<extra></extra>",
        ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["period"], y=df["cum_previsto"], name="Acum. Previsto",
        mode="lines+markers",
        line=dict(color=BK_BLUE_LIGHT, width=2.5, dash="dot"),
        marker=dict(size=7, color=BK_BLUE_LIGHT, line=dict(color=BK_DARK, width=1.5)),
        hovertemplate="<b>%{x}</b><br>Acum. Previsto: R$ %{y:,.2f}<extra></extra>",
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df["period"], y=df["cum_realizado"], name="Acum. Realizado",
        mode="lines+markers",
        line=dict(color=BK_GREEN, width=2.5),
        marker=dict(size=7, color=BK_GREEN, symbol="diamond", line=dict(color=BK_DARK, width=1.5)),
        hovertemplate="<b>%{x}</b><br>Acum. Realizado: R$ %{y:,.2f}<extra></extra>",
    ), secondary_y=True)

    fig.update_yaxes(title_text="Valor (R$)", secondary_y=False,
                     gridcolor="rgba(51,65,85,0.5)", tickfont=dict(color=BK_TEXT_MUTED))
    fig.update_yaxes(title_text="Saldo Acumulado (R$)", secondary_y=True,
                     gridcolor="rgba(0,0,0,0)", tickfont=dict(color=BK_TEXT_MUTED))

    return _layout(fig, "Fluxo de Caixa — Previsto × Realizado", height=460, xangle=-30)


def fig_breakdown_h(df: pd.DataFrame, title: str = "", top: int = 20) -> go.Figure:
    """Barras horizontais para breakdowns (categoria, centro de custo)."""
    df = df.copy()
    df["Abs"] = df["Valor"].abs()
    df = df.nlargest(top, "Abs").sort_values("Abs", ascending=True)
    colors = [BK_GREEN if v >= 0 else BK_RED for v in df["Valor"]]

    fig = go.Figure(go.Bar(
        x=df["Valor"], y=df["Item"], orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        text=df["Valor"].apply(lambda v: f"R$ {v:,.0f}"),
        textposition="outside",
        textfont=dict(size=11, family="Inter", color=BK_TEXT),
        hovertemplate="<b>%{y}</b><br>R$ %{x:,.2f}<extra></extra>",
    ))
    return _layout(fig, title, height=max(340, len(df) * 30 + 90), xangle=0)


def fig_donut_status(labels: list, values: list, title: str = "Status") -> go.Figure:
    """Donut chart para status."""
    colors = [STATUS_COLORS.get(l, BK_GRAY) for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.58,
        marker=dict(colors=colors, line=dict(color=BK_DARK, width=2.5)),
        textfont=dict(size=12, family="Inter", color=BK_TEXT),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:11px'>total</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, family="Inter", color=BK_TEXT),
    )
    return _layout(fig, title, height=330, xangle=0)


def fig_saldo_contas(df: pd.DataFrame) -> go.Figure:
    """Barras de saldo por conta bancária."""
    df = df.sort_values("Saldo", ascending=True)
    colors = [BK_GREEN if v >= 0 else BK_RED for v in df["Saldo"]]

    fig = go.Figure(go.Bar(
        x=df["Saldo"], y=df["Conta"], orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        text=df["Saldo"].apply(lambda v: f"R$ {v:,.2f}"),
        textposition="outside",
        textfont=dict(size=11, color=BK_TEXT),
        hovertemplate="<b>%{y}</b><br>Saldo: R$ %{x:,.2f}<extra></extra>",
    ))
    return _layout(fig, "Saldo por Conta", height=max(290, len(df) * 38 + 90), xangle=0)


def fig_entradas_saidas_mensal(df_month: pd.DataFrame) -> go.Figure:
    """Barras agrupadas Entradas x Saídas por mês."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_month["period"], y=df_month.get("entradas", []),
        name="Entradas",
        marker=dict(color=BK_GREEN, opacity=0.85, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Entradas: R$ %{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df_month["period"], y=df_month.get("saidas", []),
        name="Saídas",
        marker=dict(color=BK_RED, opacity=0.85, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Saídas: R$ %{y:,.2f}<extra></extra>",
    ))
    return _layout(fig, "Entradas × Saídas por Período", height=370, xangle=-30)


# ═══════════════════════════════════════
# PROJETOS
# ═══════════════════════════════════════

def fig_projetos_status(projetos: list[dict]) -> go.Figure:
    from collections import Counter
    counts = Counter(p.get("status", "N/A") for p in projetos)
    labels = list(counts.keys())
    values = list(counts.values())
    return fig_donut_status(labels, values, "Status dos Projetos")


def fig_gantt_projetos(projetos: list[dict]) -> go.Figure:
    today = date.today()
    rows = []
    for p in projetos:
        try:
            ini = datetime.strptime(str(p.get("dataInicio", ""))[:10], "%Y-%m-%d")
        except Exception:
            ini = datetime.now()
        rows.append(dict(
            Projeto=str(p.get("nome", ""))[:40],
            Início=ini, Fim=datetime.now(),
            Status=p.get("status", "N/A"),
        ))

    if not rows:
        return go.Figure()

    df = pd.DataFrame(rows).sort_values("Início")
    colors = {s: STATUS_COLORS.get(s, BK_GRAY) for s in df["Status"].unique()}
    fig = px.timeline(df, x_start="Início", x_end="Fim", y="Projeto",
                      color="Status", color_discrete_map=colors)
    fig.update_yaxes(autorange="reversed")

    hoje = datetime.now().strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=hoje, x1=hoje, y0=0, y1=1,
                  xref="x", yref="paper",
                  line=dict(color=BK_RED, width=2, dash="dash"))
    fig.add_annotation(x=hoje, y=1.03, xref="x", yref="paper",
                       text="<b>Hoje</b>", showarrow=False,
                       font=dict(color=BK_RED, size=11),
                       bgcolor=BK_SURFACE, bordercolor=BK_RED, borderwidth=1)

    return _layout(fig, "Timeline dos Projetos", height=max(320, len(rows) * 34 + 90), xangle=-30)


def fig_curva_s(df_eap: pd.DataFrame) -> go.Figure:
    """Curva S: % previsto acumulado x % realizado acumulado."""
    if df_eap.empty:
        return go.Figure()

    fig = go.Figure()
    if "previsto" in df_eap.columns:
        cum = df_eap["previsto"].cumsum()
        tot = cum.max() or 1
        fig.add_trace(go.Scatter(
            x=list(range(len(cum))), y=(cum / tot * 100).tolist(),
            name="Previsto %", mode="lines+markers",
            line=dict(color=BK_BLUE, width=2.5, dash="dot"),
            marker=dict(size=7, color=BK_BLUE, line=dict(color=BK_DARK, width=1.5)),
            hovertemplate="Período %{x}<br>Previsto: %{y:.1f}%<extra></extra>",
        ))
    if "realizado" in df_eap.columns:
        cum_r = df_eap["realizado"].cumsum()
        tot_r = cum_r.max() or 1
        fig.add_trace(go.Scatter(
            x=list(range(len(cum_r))), y=(cum_r / tot_r * 100).tolist(),
            name="Realizado %", mode="lines+markers",
            line=dict(color=BK_GREEN, width=2.5),
            marker=dict(size=7, color=BK_GREEN, symbol="diamond", line=dict(color=BK_DARK, width=1.5)),
            hovertemplate="Período %{x}<br>Realizado: %{y:.1f}%<extra></extra>",
        ))

    return _layout(fig, "Curva S — Previsto × Realizado (%)", height=390, xangle=0)


# ═══════════════════════════════════════
# VENDAS / PROPOSTAS
# ═══════════════════════════════════════

def fig_pipeline_leads(df_leads: pd.DataFrame) -> go.Figure:
    if df_leads.empty:
        return go.Figure()

    stage_order = ["novo", "contato", "proposta", "ganho", "perdido"]
    counts = df_leads["stage"].value_counts().reindex(stage_order, fill_value=0)
    colors = [STATUS_COLORS.get(s, BK_GRAY) for s in counts.index]

    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker=dict(color=colors, opacity=0.88, line=dict(width=0)),
        text=counts.values, textposition="outside",
        textfont=dict(size=13, family="Inter", color=BK_TEXT),
        hovertemplate="<b>%{x}</b><br>%{y} leads<extra></extra>",
    ))
    return _layout(fig, "Pipeline de Leads por Estágio", height=330, xangle=0)


def fig_propostas_status(df_prop: pd.DataFrame) -> go.Figure:
    if df_prop.empty:
        return go.Figure()
    counts = df_prop["status"].value_counts()
    return fig_donut_status(counts.index.tolist(), counts.values.tolist(), "Propostas por Status")


def fig_propostas_valor(df_prop: pd.DataFrame) -> go.Figure:
    if df_prop.empty or "value_total" not in df_prop.columns:
        return go.Figure()

    df_g = df_prop.groupby("status")["value_total"].sum().reset_index()
    df_g = df_g.sort_values("value_total", ascending=True)
    colors = [STATUS_COLORS.get(s, BK_GRAY) for s in df_g["status"]]

    fig = go.Figure(go.Bar(
        x=df_g["value_total"], y=df_g["status"], orientation="h",
        marker=dict(color=colors, opacity=0.88, line=dict(width=0)),
        text=df_g["value_total"].apply(lambda v: f"R$ {v:,.0f}"),
        textposition="outside",
        textfont=dict(size=11, color=BK_TEXT),
        hovertemplate="<b>%{y}</b><br>R$ %{x:,.2f}<extra></extra>",
    ))
    return _layout(fig, "Valor Total por Status de Proposta (R$)", height=330, xangle=0)


# ═══════════════════════════════════════
# COMPRAS
# ═══════════════════════════════════════

def fig_compras_status(df_po: pd.DataFrame) -> go.Figure:
    if df_po.empty:
        return go.Figure()
    counts = df_po["status"].value_counts()
    return fig_donut_status(counts.index.tolist(), counts.values.tolist(), "Pedidos por Status")


def fig_compras_fornecedor(df_po: pd.DataFrame) -> go.Figure:
    if df_po.empty or "value_total" not in df_po.columns:
        return go.Figure()

    col_forn = next((c for c in ["supplier", "supplier_id"] if c in df_po.columns), None)
    if col_forn is None:
        return go.Figure()

    df_g = df_po.groupby(col_forn)["value_total"].sum().reset_index()
    df_g.columns = ["Item", "Valor"]
    return fig_breakdown_h(df_g, "Compras por Fornecedor (R$)", top=15)


def fig_compras_mensal(df_po: pd.DataFrame) -> go.Figure:
    if df_po.empty or "order_date" not in df_po.columns:
        return go.Figure()

    df = df_po.copy()
    df["mes"] = pd.to_datetime(df["order_date"], errors="coerce").dt.to_period("M").astype(str)
    df_m = df.groupby("mes")["value_total"].sum().reset_index()
    df_m.columns = ["Mês", "Total"]

    fig = go.Figure(go.Bar(
        x=df_m["Mês"], y=df_m["Total"],
        marker=dict(color=BK_BLUE, opacity=0.85, line=dict(width=0)),
        text=df_m["Total"].apply(lambda v: f"R$ {v:,.0f}"),
        textposition="outside",
        textfont=dict(size=11, color=BK_TEXT),
        hovertemplate="<b>%{x}</b><br>R$ %{y:,.2f}<extra></extra>",
    ))
    return _layout(fig, "Valor de Compras por Mês (R$)", height=350, xangle=-30)
