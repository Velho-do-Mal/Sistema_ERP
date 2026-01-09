# bk_erp_shared/theme.py
# Tema compartilhado do BK_ERP
# --------------------------------------------------------------------------------
# - load_svg(path) -> carrega SVG e retorna data URI base64 (usado no header/hero)
# - apply_theme() -> injeta CSS com variáveis de cor e regras para layout
# Observação: preservei os seletores usados nos relatórios e nas tabelas para
# manter o layout e a aparência estrutural (margens, bordas, espaçamentos).
# --------------------------------------------------------------------------------

from __future__ import annotations
import base64
from pathlib import Path
import streamlit as st

def load_svg(path: str) -> str:
    """
    Lê um arquivo SVG local e retorna uma string data-uri base64 para uso no HTML img src.
    """
    data = Path(path).read_text(encoding="utf-8")
    b64 = base64.b64encode(data.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"

def apply_theme():
    """
    Injeta o CSS do tema na aplicação Streamlit.
    - Cores: paleta azul primária, branco e cinza-claro de fundo.
    - Mantém classes e seletores do layout original para não quebrar relatórios.
    """
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

    :root{
        --bk-primary: #0B5ED7;     /* Azul primário solicitado */
        --bk-accent: #0B84FF;
        --bk-highlight: #0B5ED7;
        --bk-bg: #F2F4F7;          /* Cinza claro de fundo */
        --bk-card: #FFFFFF;        /* Cartões brancos */
        --bk-text: #0f172a;
        --bk-muted: #6C757D;
        --bk-border: rgba(15,23,42,0.08);
        --bk-shadow: 0 18px 42px rgba(2,6,23,0.04);
        --bk-radius: 16px;
    }

    /* Fonte global */
    html, body, [class*="css"]{
        font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    }

    /* Plano de fundo e cor padrão de texto */
    .stApp {
        background: var(--bk-bg);
        color: var(--bk-text);
    }

    /* Cartões (mantive padding, borda e shadow para não alterar o layout) */
    .bk-card{
        background: var(--bk-card);
        border-radius: var(--bk-radius);
        border: 1px solid var(--bk-border);
        box-shadow: var(--bk-shadow);
        padding: 18px 20px;
    }

    /* Títulos */
    .bk-title{
        font-size: 30px;
        font-weight: 900;
        color: var(--bk-highlight);
        letter-spacing: -0.02em;
        margin: 0 0 2px 0;
        line-height: 1.1;
    }

    .bk-subtitle{
        font-size: 13px;
        color: var(--bk-muted);
        margin: 0;
    }

    /* Botões com borda arredondada */
    .stButton>button{
        border-radius: 14px !important;
        border: 1px solid rgba(0,0,0,0.06) !important;
        box-shadow: 0 12px 26px rgba(2,6,23,0.06) !important;
    }

    /* Garantir que tabelas / dataframes mantenham o layout com bordas e padding */
    div[data-testid="stDataFrame"] table,
    div[data-testid="stTable"] table,
    table{
        border-collapse: collapse !important;
    }
    div[data-testid="stDataFrame"] table th,
    div[data-testid="stDataFrame"] table td,
    div[data-testid="stTable"] table th,
    div[data-testid="stTable"] table td,
    table th, table td{
        border: 1px solid #334155 !important;
        padding: 7px 10px !important;
        font-size: 13px !important;
        white-space: nowrap !important;
    }
    div[data-testid="stDataFrame"] table th,
    div[data-testid="stTable"] table th{
        background: rgba(11,94,215,0.06) !important; /* cor delicada para cabeçalhos */
        font-weight: 800 !important;
    }

    /* Valor das métricas */
    div[data-testid="stMetricValue"]{
        font-weight: 900;
        color: var(--bk-text);
    }

    /* Footer pequeno */
    .footer-small { color:var(--bk-muted); font-size:13px; text-align:center; margin-top:18px; }
    
    /* Sidebar (menu à esquerda) - azul claro para diferenciar do conteúdo */
    section[data-testid="stSidebar"]{
        background: #DCEBFF !important;
    }

    /* Inputs e combos (baseweb) com fundo branco para melhor contraste */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="datepicker"] > div,
    div[data-baseweb="base-input"] > div {
        background: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        color: #111827 !important;
    }

    /* Placeholder */
    input::placeholder, textarea::placeholder {
        color: #6B7280 !important;
        opacity: 1 !important;
    }

    /* Foco */
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="textarea"]:focus-within,
    div[data-baseweb="datepicker"] > div:focus-within {
        border-color: #0B5ED7 !important;
        box-shadow: 0 0 0 3px rgba(11,94,215,0.18) !important;
    }

    section[data-testid="stSidebar"] > div{
        background: #DCEBFF !important;
    }
    /* Navegação (links das páginas) */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"]{
        background: transparent !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a{
        color: var(--bk-text) !important;
        border-radius: 10px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover{
        background: rgba(11,94,215,0.10) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"]{
        background: rgba(11,94,215,0.16) !important;
        border: 1px solid rgba(11,94,215,0.20) !important;
        font-weight: 800 !important;
    }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)