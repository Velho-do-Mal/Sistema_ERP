# bk_erp_shared/theme.py
"""
Tema BK_ERP — visual idêntico ao BK Planejamento Estratégico.
Paleta, CSS, KPI cards, header gradiente, sidebar dark, campos visíveis.
"""

import base64
from pathlib import Path
import streamlit as st

# ═══════════════════════════════════════════
# PALETA BK — cores centralizadas
# ═══════════════════════════════════════════
BK_BLUE       = "#1565C0"
BK_BLUE_LIGHT = "#42A5F5"
BK_TEAL       = "#00897B"
BK_GREEN      = "#43A047"
BK_ORANGE     = "#FB8C00"
BK_RED        = "#E53935"
BK_PURPLE     = "#7B1FA2"
BK_GRAY       = "#546E7A"
BK_BG         = "#F0F4F8"
BK_DARK       = "#0D1B2A"
BK_CARD       = "#FFFFFF"

PLOTLY_TEMPLATE = "plotly_white"

PLOTLY_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    font=dict(family="Segoe UI, Arial", size=12, color=BK_DARK),
    margin=dict(l=30, r=30, t=50, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=11)),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Segoe UI"),
)

BK_COLORS = [BK_BLUE, BK_TEAL, BK_GREEN, BK_ORANGE, BK_RED, BK_PURPLE, BK_GRAY, BK_BLUE_LIGHT]


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
    """Compatibilidade: retorna DATA URI para imagem. Retorna '' se não encontrar."""
    return _file_to_data_uri(path) or ""


def _default_monogram_svg() -> str:
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#1565C0"/>
          <stop offset="1" stop-color="#00897B"/>
        </linearGradient>
      </defs>
      <rect x="6" y="6" width="84" height="84" rx="22" fill="url(#g)"/>
      <text x="48" y="60" text-anchor="middle" font-size="36" font-weight="800"
            font-family="Segoe UI, Arial" fill="#ffffff">BK</text>
    </svg>'''
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


def _pick_brand_logo() -> str:
    candidates = [
        "assets/logo_bk.svg", "assets/logo_bk.png",
        "assets/logo.svg",    "assets/logo.png",
        "assets/bk_logo.svg", "assets/bk_logo.png",
        "bk_erp_shared/assets/logo_bk.svg",
    ]
    for c in candidates:
        uri = _file_to_data_uri(c)
        if uri:
            return uri
    return _default_monogram_svg()


# ═══════════════════════════════════════════
# CSS PRINCIPAL
# ═══════════════════════════════════════════

_CSS = """
<style>
/* ── Reset / Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Segoe UI', 'Inter', system-ui, -apple-system, sans-serif !important;
}

.stApp { background: #F0F4F8 !important; }

section.main > div.block-container {
    max-width: 1320px;
    padding-top: 0.5rem;
    padding-bottom: 2rem;
}

/* ── Header gradiente ── */
.bk-header {
    background: linear-gradient(135deg, #1565C0 0%, #00897B 100%);
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 4px 20px rgba(21,101,192,0.25);
}
.bk-header-logo {
    width: 52px; height: 52px; border-radius: 14px;
    background: rgba(255,255,255,0.15);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; overflow: hidden;
}
.bk-header-logo img { width: 100%; height: 100%; object-fit: cover; }
.bk-header-title {
    font-size: 22px; font-weight: 900;
    color: #fff; letter-spacing: -0.02em; margin: 0;
}
.bk-header-sub {
    font-size: 12px; color: rgba(255,255,255,0.75); margin-top: 2px;
}

/* ── Sidebar dark ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background: #0D1B2A !important;
}
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
    border-radius: 10px !important;
    padding: 8px 12px !important;
    margin: 2px 0 !important;
    transition: background 0.2s;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
    background: rgba(255,255,255,0.08) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(21,101,192,0.45) !important;
    border: 1px solid rgba(66,165,245,0.35) !important;
    font-weight: 800 !important;
}

/* ── KPI Cards ── */
.bk-kpi-row { display: flex; gap: 14px; margin: 12px 0 18px 0; flex-wrap: wrap; }
.bk-kpi {
    flex: 1; min-width: 140px;
    background: #fff;
    border-radius: 12px;
    border: 1px solid rgba(21,101,192,0.10);
    box-shadow: 0 2px 12px rgba(21,101,192,0.07);
    padding: 14px 18px;
    text-align: center;
}
.bk-kpi-value {
    font-size: 26px; font-weight: 900;
    letter-spacing: -0.02em; margin: 0;
}
.bk-kpi-label {
    font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; color: #546E7A; margin-top: 4px;
}
.bk-kpi-blue   { color: #1565C0; }
.bk-kpi-green  { color: #43A047; }
.bk-kpi-teal   { color: #00897B; }
.bk-kpi-orange { color: #FB8C00; }
.bk-kpi-red    { color: #E53935; }
.bk-kpi-gray   { color: #546E7A; }

/* ── Section title ── */
.bk-section {
    font-size: 17px; font-weight: 800; color: #1565C0;
    border-left: 4px solid #1565C0;
    padding-left: 10px; margin: 18px 0 10px 0;
}

/* ── Card container ── */
.bk-card {
    background: #fff;
    border-radius: 12px;
    border: 1px solid rgba(21,101,192,0.10);
    box-shadow: 0 2px 12px rgba(21,101,192,0.06);
    padding: 16px 20px;
    margin-bottom: 14px;
}
.bk-title {
    font-size: 22px; font-weight: 900; color: #1565C0;
    margin: 0 0 2px 0; letter-spacing: -0.01em;
}
.bk-subtitle { font-size: 13px; color: #546E7A; margin: 0; }

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
    background: rgba(21,101,192,0.06) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
}
div[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 6px 14px !important;
    color: #546E7A !important;
    transition: all 0.2s !important;
}
div[data-baseweb="tab"][aria-selected="true"] {
    background: #1565C0 !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(21,101,192,0.3) !important;
}

/* ── Inputs com borda visível ── */
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background: #fff !important;
    border: 1.5px solid #90A4AE !important;
    border-radius: 8px !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within {
    border-color: #1565C0 !important;
    box-shadow: 0 0 0 3px rgba(21,101,192,0.15) !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    color: #1a202c !important;
    background: #fff !important;
    font-size: 14px !important;
}
div[data-baseweb="select"] > div:first-child {
    background: #fff !important;
    border: 1.5px solid #90A4AE !important;
    border-radius: 8px !important;
}
div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stDateInput"] label {
    color: #1E3A5F !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* ── Botões ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1565C0, #1976D2) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(21,101,192,0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 14px rgba(21,101,192,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── DataEditor / DataFrame ── */
[data-testid="stDataEditor"] { border-radius: 10px; overflow: hidden; }

/* ── Expander ── */
details > summary {
    background: #EBF3FB !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-weight: 600 !important;
    color: #1565C0 !important;
}

/* ── Métricas nativas ── */
[data-testid="metric-container"] {
    background: #fff;
    border: 1px solid rgba(21,101,192,0.10);
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 2px 8px rgba(21,101,192,0.06);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 11px !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 0.05em; color: #546E7A !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px !important; font-weight: 900 !important; color: #1565C0 !important;
}

/* ── Footer ── */
.footer {
    text-align: center; color: #90A4AE; font-size: 11px;
    padding: 20px 0 4px 0; letter-spacing: 0.03em;
}

/* ── Stat-card legado ── */
.stat-card {
    background: #fff;
    border-radius: 12px;
    border: 1px solid rgba(21,101,192,0.10);
    box-shadow: 0 2px 8px rgba(21,101,192,0.06);
    padding: 14px 18px; text-align: center; margin-bottom: 10px;
}
.metric-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; color: #546E7A; margin-bottom: 6px;
}
.metric-value {
    font-size: 22px; font-weight: 900; color: #1565C0; margin: 0;
}
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
    """Injeta o tema BK em qualquer página Streamlit."""
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
    """Retorna HTML de um KPI card."""
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
    """Título de seção com barra lateral azul."""
    st.markdown(f'<div class="bk-section">{title}</div>', unsafe_allow_html=True)
