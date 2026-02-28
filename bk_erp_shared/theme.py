# bk_erp_shared/theme.py
"""
Tema BK_ERP — dark mode idêntico ao BK Finance
Paleta, CSS, KPI cards, sidebar dark, campos visíveis, labels claros.
"""

import base64
from pathlib import Path
import streamlit as st

# ═══════════════════════════════════════════
# PALETA BK DARK
# ═══════════════════════════════════════════
BK_BLUE       = "#3B82F6"
BK_BLUE_LIGHT = "#93C5FD"
BK_BLUE_DARK  = "#1E40AF"
BK_TEAL       = "#14B8A6"
BK_GREEN      = "#10B981"
BK_ORANGE     = "#F59E0B"
BK_RED        = "#EF4444"
BK_PURPLE     = "#8B5CF6"
BK_GRAY       = "#64748B"
BK_BG         = "#0F172A"
BK_SURFACE    = "#1E293B"
BK_BORDER     = "#334155"
BK_TEXT       = "#F1F5F9"
BK_TEXT_MUTED = "#94A3B8"

PLOTLY_TEMPLATE = "plotly_dark"
BK_COLORS = [BK_BLUE, BK_TEAL, BK_GREEN, BK_ORANGE, BK_RED, BK_PURPLE, BK_GRAY, BK_BLUE_LIGHT]

PLOTLY_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    font=dict(family="Inter, Segoe UI, Arial", size=12, color=BK_TEXT),
    margin=dict(l=30, r=30, t=50, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=11, color=BK_TEXT)),
    plot_bgcolor=BK_SURFACE,
    paper_bgcolor=BK_BG,
    hoverlabel=dict(bgcolor=BK_SURFACE, font_size=12, font_family="Inter",
                    bordercolor=BK_BORDER, font=dict(color=BK_TEXT)),
    colorway=BK_COLORS,
)


# ═══════════════════════════════════════════
# LOGO / ASSET HELPERS
# ═══════════════════════════════════════════

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
        mime = {".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".webp": "image/webp"}[suffix]
        return f"data:{mime};base64,{b64}"
    return None


def load_svg(path: str) -> str:
    return _file_to_data_uri(path) or ""


def _pick_brand_logo() -> str:
    candidates = [
        "assets/bk_icon.jpeg", "assets/bk_icon.png",
        "assets/logo_bk.svg",  "assets/logo_bk.png",
        "assets/logo.svg",     "assets/logo.png",
    ]
    for c in candidates:
        uri = _file_to_data_uri(c)
        if uri:
            return uri
    # fallback monogram SVG
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#1E40AF"/>
          <stop offset="1" stop-color="#3B82F6"/>
        </linearGradient>
      </defs>
      <rect x="6" y="6" width="84" height="84" rx="22" fill="url(#g)"/>
      <text x="48" y="62" text-anchor="middle" font-size="36" font-weight="800"
            font-family="Inter, Arial" fill="#ffffff">BK</text>
    </svg>'''
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


# ═══════════════════════════════════════════
# CSS PRINCIPAL — Dark Theme
# ═══════════════════════════════════════════

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: #F1F5F9 !important;
}

.stApp {
    background: #0F172A !important;
}

section.main > div.block-container {
    max-width: 1360px;
    padding-top: 0.5rem;
    padding-bottom: 2rem;
}

/* ── Sidebar dark ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
    border-right: 1px solid #1E40AF44 !important;
}
section[data-testid="stSidebar"] * { color: #F1F5F9 !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
    border-radius: 10px !important;
    padding: 8px 12px !important;
    margin: 2px 0 !important;
    transition: background 0.2s;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
    background: rgba(59,130,246,0.15) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(30,64,175,0.5) !important;
    border: 1px solid rgba(147,197,253,0.25) !important;
    font-weight: 800 !important;
}

/* ── Header gradiente ── */
.bk-header {
    background: linear-gradient(135deg, #1E3A8A 0%, #1E40AF 50%, #2563EB 100%);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 8px 32px rgba(59,130,246,0.2);
    border: 1px solid rgba(147,197,253,0.15);
}
.bk-header-logo {
    width: 54px; height: 54px; border-radius: 14px;
    background: rgba(255,255,255,0.12);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; overflow: hidden;
}
.bk-header-logo img { width: 100%; height: 100%; object-fit: cover; border-radius: 12px; }
.bk-header-title {
    font-size: 22px; font-weight: 900;
    color: #fff; letter-spacing: -0.02em; margin: 0;
}
.bk-header-sub {
    font-size: 12px; color: rgba(255,255,255,0.7); margin-top: 3px;
}

/* ── KPI Cards ── */
.bk-kpi-row { display: flex; gap: 14px; margin: 12px 0 18px 0; flex-wrap: wrap; }
.bk-kpi {
    flex: 1; min-width: 150px;
    background: #1E293B;
    border-radius: 14px;
    border: 1px solid #334155;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    padding: 16px 20px;
    text-align: center;
}
.bk-kpi-value {
    font-size: 26px; font-weight: 900;
    letter-spacing: -0.02em; margin: 0;
}
.bk-kpi-label {
    font-size: 11px; font-weight: 600; letter-spacing: 0.05em;
    text-transform: uppercase; color: #94A3B8 !important; margin-top: 5px;
}
.bk-kpi-blue   { color: #93C5FD; }
.bk-kpi-green  { color: #34D399; }
.bk-kpi-teal   { color: #2DD4BF; }
.bk-kpi-orange { color: #FCD34D; }
.bk-kpi-red    { color: #F87171; }
.bk-kpi-gray   { color: #94A3B8; }

/* ── Section title ── */
.bk-section {
    font-size: 16px; font-weight: 700; color: #93C5FD !important;
    border-left: 4px solid #3B82F6;
    padding-left: 10px; margin: 18px 0 10px 0;
}

/* ── Card container ── */
.bk-card {
    background: #1E293B;
    border-radius: 14px;
    border: 1px solid #334155;
    box-shadow: 0 2px 16px rgba(0,0,0,0.3);
    padding: 18px 22px;
    margin-bottom: 16px;
}
.bk-title {
    font-size: 22px; font-weight: 900; color: #93C5FD !important;
    margin: 0 0 4px 0; letter-spacing: -0.01em;
}
.bk-subtitle { font-size: 13px; color: #94A3B8 !important; margin: 0; }

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
    background: rgba(30,64,175,0.15) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
}
div[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 7px 16px !important;
    color: #94A3B8 !important;
    transition: all 0.2s !important;
}
div[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1E40AF, #2563EB) !important;
    color: #fff !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.35) !important;
}

/* ── Inputs / Selects ── */
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background: #1E293B !important;
    border: 1.5px solid #334155 !important;
    border-radius: 8px !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.2) !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    color: #F1F5F9 !important;
    background: #1E293B !important;
    font-size: 14px !important;
}
div[data-baseweb="select"] > div:first-child {
    background: #1E293B !important;
    border: 1.5px solid #334155 !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
}

/* ── Labels dos inputs — CLAROS e legíveis ── */
div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stDateInput"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stRadio"] label,
div[data-testid="stSlider"] label,
.stRadio label, .stCheckbox label {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}

/* ── Botões ── */
.stButton > button {
    border-radius: 9px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
    border: 1px solid #334155 !important;
    color: #F1F5F9 !important;
    background: #1E293B !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1E40AF, #2563EB) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 18px rgba(37,99,235,0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: #334155 !important;
    border-color: #3B82F6 !important;
}

/* ── DataEditor / DataFrame ── */
[data-testid="stDataEditor"],
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #334155;
}

/* ── Expander ── */
details > summary {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 9px !important;
    padding: 9px 14px !important;
    font-weight: 600 !important;
    color: #93C5FD !important;
}
details[open] > summary {
    border-bottom-left-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
}

/* ── Métricas nativas ── */
[data-testid="metric-container"] {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 11px !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 0.05em;
    color: #94A3B8 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px !important; font-weight: 900 !important;
    color: #93C5FD !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

/* ── Caption / st.caption ── */
[data-testid="stCaptionContainer"] {
    color: #94A3B8 !important;
    font-size: 12px !important;
}

/* ── Info / Warning / Success ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
}

/* ── Footer ── */
.footer {
    text-align: center; color: #475569; font-size: 11px;
    padding: 20px 0 4px 0; letter-spacing: 0.03em;
}

/* ── Stat-card legado ── */
.stat-card {
    background: #1E293B;
    border-radius: 12px;
    border: 1px solid #334155;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    padding: 14px 18px; text-align: center; margin-bottom: 10px;
}
.metric-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; color: #94A3B8; margin-bottom: 6px;
}
.metric-value {
    font-size: 22px; font-weight: 900; color: #93C5FD; margin: 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0F172A; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3B82F6; }

/* ── Markdown headings ── */
h1, h2, h3, h4 { color: #F1F5F9 !important; }
p, li, span { color: #CBD5E1 !important; }

/* ── st.markdown code ── */
code { background: #334155 !important; color: #93C5FD !important; border-radius: 4px; padding: 2px 6px; }
</style>
"""


# ═══════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ═══════════════════════════════════════════

def apply_theme(
    show_brand_bar: bool = True,
    brand_title: str = "BK Engenharia e Tecnologia",
    brand_subtitle: str = "ERP — Financeiro · Projetos · Vendas · Compras",
):
    """Injeta o tema BK dark em qualquer página Streamlit."""
    st.markdown(_CSS, unsafe_allow_html=True)

    if show_brand_bar:
        logo_uri = _pick_brand_logo()
        st.markdown(f"""
        <div class="bk-header">
            <div class="bk-header-logo"><img src="{logo_uri}" alt="BK"/></div>
            <div>
                <div class="bk-header-title">{brand_title}</div>
                <div class="bk-header-sub">{brand_subtitle}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# HELPERS DE UI
# ═══════════════════════════════════════════

def bk_kpi(label: str, value: str, color: str = "blue") -> str:
    return f"""<div class="bk-kpi">
        <div class="bk-kpi-value bk-kpi-{color}">{value}</div>
        <div class="bk-kpi-label">{label}</div>
    </div>"""


def bk_kpi_row(cards: list[tuple]) -> None:
    """
    Renderiza uma linha de KPI cards.
    cards = [(label, value, color), ...]
    cores: blue | green | teal | orange | red | gray
    """
    html = '<div class="bk-kpi-row">'
    for label, value, color in cards:
        html += bk_kpi(label, str(value), color)
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def bk_section(title: str) -> None:
    st.markdown(f'<div class="bk-section">{title}</div>', unsafe_allow_html=True)
