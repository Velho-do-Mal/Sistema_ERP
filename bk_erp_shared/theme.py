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



    /* Inputs (texto) e selects (combo) com fundo branco para melhor legibilidade */
    /* TextInput / NumberInput / DateInput / TextArea */
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stDateInput"] input,
    div[data-testid="stTimeInput"] input,
    div[data-testid="stTextArea"] textarea{
        background: #FFFFFF !important;
        color: var(--bk-text) !important;
        border: 1px solid rgba(15,23,42,0.20) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }
    div[data-testid="stTextInput"] input::placeholder,
    div[data-testid="stTextArea"] textarea::placeholder{
        color: rgba(15,23,42,0.55) !important;
    }
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stDateInput"] input:focus,
    div[data-testid="stTimeInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus{
        outline: none !important;
        border: 1px solid rgba(11,94,215,0.65) !important;
        box-shadow: 0 0 0 3px rgba(11,94,215,0.12) !important;
    }

    /* Selectbox / Multiselect (BaseWeb) */
    div[data-baseweb="select"] > div{
        background: #FFFFFF !important;
        border: 1px solid rgba(15,23,42,0.20) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }
    div[data-baseweb="select"] span{
        color: var(--bk-text) !important;
    }
    div[data-baseweb="select"] svg{
        color: rgba(15,23,42,0.70) !important;
    }
    div[data-baseweb="select"]:focus-within > div{
        border: 1px solid rgba(11,94,215,0.65) !important;
        box-shadow: 0 0 0 3px rgba(11,94,215,0.12) !important;
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
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
