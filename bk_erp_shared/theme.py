# bk_erp_shared/theme.py
# Tema compartilhado do BK_ERP
# --------------------------------------------------------------------------------
# Objetivos deste tema (conforme alinhado):
# - Layout consistente em todas as páginas (sem alterar fontes nem ícones).
# - Barra lateral em azul claro.
# - Inputs (combos/entradas) com fundo branco e boa legibilidade.
# - Visual moderno (cartões, bordas suaves, sombras leves).
# - NÃO interferir nos RELATÓRIOS (evitar CSS genérico que afeta HTML externo).
#
# Observação:
# - Este arquivo injeta apenas CSS e (opcionalmente) um cabeçalho/brand bar simples.
# - Caso você tenha uma logo em SVG/PNG, coloque em um destes caminhos (no repo):
#     assets/logo_bk.svg | assets/logo_bk.png | assets/bk_logo.svg | assets/bk_logo.png
#   O tema tentará carregar automaticamente. Se não achar, mostra um monograma "BK".
# --------------------------------------------------------------------------------

from __future__ import annotations

import base64
from pathlib import Path
import streamlit as st


def _file_to_data_uri(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    suffix = p.suffix.lower()
    if suffix == ".svg":
        data = p.read_text(encoding="utf-8").encode("utf-8")
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/svg+xml;base64,{b64}"
    if suffix in (".png", ".jpg", ".jpeg", ".webp"):
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }[suffix]
        return f"data:{mime};base64,{b64}"
    return None


def _default_monogram_svg() -> str:
    # Monograma minimalista "BK" (SVG inline) – evita depender de arquivo externo.
    svg = r'''
    <svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#2563EB"/>
          <stop offset="1" stop-color="#38BDF8"/>
        </linearGradient>
      </defs>
      <rect x="6" y="6" width="84" height="84" rx="22" fill="url(#g)"/>
      <text x="48" y="58" text-anchor="middle" font-size="34" font-weight="800"
            font-family="system-ui, -apple-system, Segoe UI, Roboto, Arial" fill="#ffffff">
        BK
      </text>
    </svg>
    '''.strip()
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


def _pick_brand_logo() -> str:
    # Tenta localizar uma logo do BK no repo, senão usa monograma.
    candidates = [
        "assets/logo_bk.svg",
        "assets/logo_bk.png",
        "assets/bk_logo.svg",
        "assets/bk_logo.png",
        "bk_erp_shared/assets/logo_bk.svg",
        "bk_erp_shared/assets/logo_bk.png",
    ]
    for c in candidates:
        uri = _file_to_data_uri(c)
        if uri:
            return uri
    return _default_monogram_svg()


def apply_theme(show_brand_bar: bool = True, brand_title: str = "BK Engenharia e Tecnologia"):
    """Injeta CSS do tema na aplicação Streamlit.

    Importante:
    - Não altera fontes globais (mantém a fonte padrão do Streamlit).
    - Não muda ícones.
    - Evita CSS genérico (ex.: `table {}`) para não afetar relatórios HTML.
    """

    css = """
    <style>
      :root{
        --bk-primary: #2563EB;
        --bk-accent:  #38BDF8;
        --bk-bg:      #F5F7FB;
        --bk-card:    #FFFFFF;
        --bk-text:    #0F172A;
        --bk-muted:   #64748B;
        --bk-border:  rgba(15, 23, 42, 0.10);
        --bk-shadow:  0 18px 42px rgba(2, 6, 23, 0.06);
        --bk-radius:  16px;

        --bk-sidebar-bg: #D9ECFF;     /* azul claro */
        --bk-sidebar-hover: rgba(37, 99, 235, 0.12);
        --bk-sidebar-active: rgba(37, 99, 235, 0.18);
      }

      /* Fundo geral */
      .stApp{
        background: var(--bk-bg);
        color: var(--bk-text);
      }

      /* Container principal */
      section.main > div.block-container{
        max-width: 1280px;
        padding-top: 1.2rem;
        padding-bottom: 2.0rem;
      }

      /* Cartões utilitários (se você usar <div class="bk-card">...) */
      .bk-card{
        background: var(--bk-card);
        border-radius: var(--bk-radius);
        border: 1px solid var(--bk-border);
        box-shadow: var(--bk-shadow);
        padding: 18px 20px;
      }

      /* Títulos utilitários */
      .bk-title{
        font-size: 30px;
        font-weight: 900;
        color: var(--bk-primary);
        letter-spacing: -0.02em;
        margin: 0 0 2px 0;
        line-height: 1.1;
      }
      .bk-subtitle{
        font-size: 13px;
        color: var(--bk-muted);
        margin: 0;
      }

      /* Botões */
      .stButton>button{
        border-radius: 14px !important;
        border: 1px solid rgba(0,0,0,0.06) !important;
        box-shadow: 0 12px 26px rgba(2,6,23,0.06) !important;
      }

      /* Sidebar */
      section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div{
        background: var(--bk-sidebar-bg) !important;
      }
      section[data-testid="stSidebar"] [data-testid="stSidebarNav"]{
        background: transparent !important;
      }
      section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a{
        color: var(--bk-text) !important;
        border-radius: 12px !important;
        padding: 8px 10px !important;
      }
      section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover{
        background: var(--bk-sidebar-hover) !important;
      }
      section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"]{
        background: var(--bk-sidebar-active) !important;
        border: 1px solid rgba(37,99,235,0.20) !important;
        font-weight: 800 !important;
      }

      /* Inputs e combos (baseweb) com fundo branco */
      div[data-baseweb="input"] > div,
      div[data-baseweb="select"] > div,
      div[data-baseweb="textarea"] textarea,
      div[data-baseweb="datepicker"] > div,
      div[data-baseweb="base-input"] > div {
        background: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
      }
      div[data-baseweb="select"] span,
      div[data-baseweb="select"] input{
        color: #111827 !important;
      }
      input::placeholder, textarea::placeholder{
        color: #6B7280 !important;
        opacity: 1 !important;
      }
      div[data-baseweb="input"] > div:focus-within,
      div[data-baseweb="select"] > div:focus-within,
      div[data-baseweb="textarea"]:focus-within,
      div[data-baseweb="datepicker"] > div:focus-within {
        border-color: var(--bk-primary) !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.18) !important;
      }

      /* Tabelas do Streamlit (não aplica em HTML genérico para não mexer nos relatórios) */
      div[data-testid="stDataFrame"] table,
      div[data-testid="stTable"] table{
        border-collapse: collapse !important;
      }
      div[data-testid="stDataFrame"] table th,
      div[data-testid="stDataFrame"] table td,
      div[data-testid="stTable"] table th,
      div[data-testid="stTable"] table td{
        border: 1px solid rgba(15,23,42,0.16) !important;
        padding: 7px 10px !important;
        font-size: 13px !important;
        white-space: nowrap !important;
      }
      div[data-testid="stDataFrame"] table th,
      div[data-testid="stTable"] table th{
        background: rgba(37,99,235,0.06) !important;
        font-weight: 800 !important;
      }

      /* Brand bar (topo) */
      .bk-topbar{
        background: linear-gradient(90deg, rgba(37,99,235,0.10), rgba(56,189,248,0.10));
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        padding: 12px 14px;
        margin: 6px 0 14px 0;
        box-shadow: 0 12px 26px rgba(2,6,23,0.04);
      }
      .bk-topbar-inner{
        display:flex;
        align-items:center;
        gap: 12px;
      }
      .bk-logo{
        width: 46px;
        height: 46px;
        border-radius: 14px;
        overflow:hidden;
        flex: 0 0 auto;
      }
      .bk-logo img{ width: 100%; height: 100%; object-fit: cover; display:block; }
      .bk-brand-title{
        font-weight: 900;
        color: var(--bk-text);
        font-size: 18px;
        letter-spacing: -0.01em;
        line-height: 1.1;
      }
      .bk-brand-sub{
        color: var(--bk-muted);
        font-size: 12px;
        margin-top: 2px;
      }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    if show_brand_bar:
        logo_uri = _pick_brand_logo()
        html = f'''
        <div class="bk-topbar">
          <div class="bk-topbar-inner">
            <div class="bk-logo"><img src="{logo_uri}" alt="BK"/></div>
            <div>
              <div class="bk-brand-title">{brand_title}</div>
              <div class="bk-brand-sub">TAP • EAP • Gantt • Curva S • Finanças • Qualidade • Riscos • Lições • Encerramento</div>
            </div>
          </div>
        </div>
        '''
        st.markdown(html, unsafe_allow_html=True)
