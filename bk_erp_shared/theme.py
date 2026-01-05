
from __future__ import annotations
import base64
from pathlib import Path
import streamlit as st

def load_svg(path: str) -> str:
    data = Path(path).read_text(encoding="utf-8")
    b64 = base64.b64encode(data.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"

def apply_theme():
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

    :root{
        --bk-primary:#007bff;
        --bk-accent:#00bcd4;
        --bk-highlight:#0d47a1;
        --bk-bg:#f4f6fb;
        --bk-card:#ffffff;
        --bk-text:#0f172a;
        --bk-muted:#64748b;
        --bk-border: rgba(15,23,42,0.10);
        --bk-shadow: 0 18px 42px rgba(2,6,23,0.08);
        --bk-radius: 18px;
    }

    html, body, [class*="css"]{
        font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 10% 0%, rgba(0,188,212,0.10), transparent 35%),
                    radial-gradient(circle at 90% 0%, rgba(0,123,255,0.10), transparent 40%),
                    var(--bk-bg);
        color: var(--bk-text);
    }

    /* Cards */
    .bk-card{
        background: var(--bk-card);
        border-radius: var(--bk-radius);
        border: 1px solid var(--bk-border);
        box-shadow: var(--bk-shadow);
        padding: 18px 20px;
    }

    .bk-title{
        font-size: 30px;
        font-weight: 900;
        color: var(--bk-highlight);
        letter-spacing: -0.02em;
        margin: 0 0 2px 0;
        line-height: 1.1;
    }
    
    .main-title{
        font-size: 28px;
        font-weight: 900;
        color: var(--bk-highlight);
        margin-bottom: 0;
    }
    .main-subtitle{
        font-size: 13px;
        color: var(--bk-muted);
    }
    
    .bk-subtitle{
        font-size: 13px;
        color: var(--bk-muted);
        margin: 0;
    }

    /* Buttons */
    .stButton>button{
        border-radius: 14px !important;
        border: 1px solid rgba(0,0,0,0.06) !important;
        box-shadow: 0 12px 26px rgba(2,6,23,0.08) !important;
    }

    /* Dataframe border */
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
        background: rgba(2,6,23,0.03) !important;
        font-weight: 800 !important;
    }

    /* Metric */
    div[data-testid="stMetricValue"]{
        font-weight: 900;
        color: var(--bk-text);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
