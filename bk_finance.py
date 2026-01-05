# app.py
r"""
BK - Gestão Financeira (ARQUIVO ÚNICO)
Streamlit + SQLAlchemy + Postgres (Neon) / SQLite fallback

✅ Neon:
- Opção A (recomendada): DATABASE_URL com user:SENHA@host/db?sslmode=require
- Opção B: DATABASE_URL sem senha + PGPASSWORD separado (o app injeta a senha na URL)

PowerShell (cole sem os ">>"):
  .\.venv\Scripts\Activate.ps1
  pip install -U streamlit sqlalchemy psycopg2-binary pandas plotly openpyxl python-dateutil

  $env:DATABASE_URL="postgresql://USER@HOST/neondb?sslmode=require"
  $env:PGPASSWORD="SENHA_DO_NEON"

  $env:INITIAL_ADMIN_EMAIL="marcio@bk-engenharia.com"
  $env:INITIAL_ADMIN_PASSWORD="TroqueEstaSenha!123"

  streamlit run app.py

OBS:
- Eu NÃO coloco senha hardcoded no código. O app lê do ambiente.
- Se você publicou senha por engano, rotacione no Neon.
"""

import os
import io
import base64
import uuid
import json
import hashlib
import hmac
from datetime import datetime, date, timedelta
from typing import Optional, List
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except Exception:
        tomllib = None  # type: ignore


from sqlalchemy import (
    create_engine, Column, Integer, String, Date, DateTime, Float,
    LargeBinary, Text, ForeignKey, Boolean, Index, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

try:
    from dateutil.relativedelta import relativedelta  # type: ignore
    HAS_DATEUTIL = True
except Exception:
    HAS_DATEUTIL = False


# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="BK_ERP - Financeiro", layout="wide")


# =========================
# CSS
# =========================
def apply_css():
    css = """
    <style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

    :root {
      --primary: #007bff;
      --accent: #00bcd4;
      --bg: #f4f6fb;
      --text: #222;
      --card-radius: 14px;
      --card-shadow: 0 10px 26px rgba(17,24,39,0.08);
      --border: rgba(0,0,0,0.10);
    }
    body { background: var(--bg); color: var(--text); font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif; }
    .big-title { font-size:34px; font-weight:900; color:#0d47a1; margin:0; letter-spacing:-0.02em; }
    .sub-title { font-size:14px; color:#666; margin-top:2px; }
    .card {
        background: #fff; padding: 18px; border-radius:var(--card-radius);
        box-shadow: var(--card-shadow); border:1px solid var(--border);
        margin-bottom:16px;
    }
    .stat-card { padding:16px; border-radius:var(--card-radius); border:1px solid var(--border); background:white; }
    .metric-value { font-size:22px; font-weight:800; color: #111; }
    .metric-label { color:#666; font-size:13px; }

    /* Tabelas com borda cinza escuro */
    div[data-testid="stDataFrame"] table,
    div[data-testid="stTable"] table,
    table { border-collapse: collapse !important; width: 100% !important; table-layout: auto !important; }

    div[data-testid="stDataFrame"] table th,
    div[data-testid="stDataFrame"] table td,
    div[data-testid="stTable"] table th,
    div[data-testid="stTable"] table td,
    table th, table td {
      border: 1px solid #5f5f5f !important;
      padding: 7px 10px !important;
      white-space: nowrap !important;
      font-size: 13px !important;
    }

    div[data-testid="stDataFrame"] table th,
    div[data-testid="stTable"] table th {
      background: #f4f4f4 !important;
      text-align: left !important;
      font-weight: 800 !important;
    }

    .footer { color:#666; font-size:12px; margin-top:24px; text-align:center; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


apply_css()

# =========================
# DATABASE
# =========================
Base = declarative_base()
LOCAL_SQLITE = "sqlite:///bk_gestao_local.db"

ENGINE_KW = dict(
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=1800,
)


class Account(Base):
    __tablename__ = "accounts"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    bank = Column(String, nullable=True)
    initial_balance = Column(Float, default=0.0)
    currency = Column(String, default="BRL")
    active = Column(Boolean, default=True)
    notes = Column(Text, nullable=True)


class Category(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    mov_type = Column(String, default="both")  # both/entrada/saida
    parent_id = Column(Integer, ForeignKey("categories.id"), nullable=True)
    parent = relationship("Category", remote_side=[id], backref="subcategories")
    notes = Column(Text, nullable=True)
    __table_args__ = (Index("idx_categories_parent", "parent_id"),)


class Client(Base):
    __tablename__ = "clients"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    document = Column(String, nullable=True)
    notes = Column(Text, nullable=True)


class Supplier(Base):
    __tablename__ = "suppliers"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    document = Column(String, nullable=True)
    notes = Column(Text, nullable=True)


class CostCenter(Base):
    __tablename__ = "cost_centers"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    notes = Column(Text, nullable=True)


class Goal(Base):
    __tablename__ = "goals"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    target_amount = Column(Float, nullable=False)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True)
    category = relationship("Category")
    notes = Column(Text, nullable=True)


class Budget(Base):
    __tablename__ = "budgets"
    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    amount = Column(Float, default=0.0)
    category = relationship("Category")
    __table_args__ = (Index("idx_budgets_cat_year_month", "category_id", "year", "month", unique=True),)


class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True)

    date = Column(Date, nullable=False)          # competência
    due_date = Column(Date, nullable=True)       # vencimento
    paid_date = Column(Date, nullable=True)      # pagamento/recebimento
    paid = Column(Boolean, default=False)        # realizado?

    description = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    type = Column(String, nullable=False)        # entrada/saida

    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=True)
    account = relationship("Account")

    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True)
    category = relationship("Category")

    client_id = Column(Integer, ForeignKey("clients.id"), nullable=True)
    client = relationship("Client")

    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=True)
    supplier = relationship("Supplier")

    cost_center_id = Column(Integer, ForeignKey("cost_centers.id"), nullable=True)
    cost_center = relationship("CostCenter")

    is_transfer = Column(Boolean, default=False)
    transfer_pair_id = Column(String, nullable=True)
    recurrence_group = Column(String, nullable=True)

    reference = Column(String, nullable=True)
    notes = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_transactions_date", "date"),
        Index("idx_transactions_due", "due_date"),
        Index("idx_transactions_paid", "paid"),
        Index("idx_transactions_pair", "transfer_pair_id"),
        Index("idx_transactions_recur", "recurrence_group"),
    )


class Attachment(Base):
    __tablename__ = "attachments"
    id = Column(Integer, primary_key=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False)
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=True)
    data = Column(LargeBinary, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    transaction = relationship("Transaction")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    role = Column(String, default="admin")  # admin/diretoria/financeiro/leitura
    password_hash = Column(String, nullable=False)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_email = Column(String, nullable=True)
    action = Column(String, nullable=False)
    entity = Column(String, nullable=False)
    record_id = Column(String, nullable=True)
    before_json = Column(Text, nullable=True)
    after_json = Column(Text, nullable=True)
    __table_args__ = (
        Index("idx_audit_created_at", "created_at"),
        Index("idx_audit_entity", "entity"),
    )


def try_exec(conn, sql: str):
    try:
        conn.execute(text(sql))
    except Exception:
        pass


def ensure_columns_sqlite(engine):
    with engine.begin() as conn:
        try_exec(conn, "ALTER TABLE accounts ADD COLUMN bank TEXT")
        try_exec(conn, "ALTER TABLE accounts ADD COLUMN initial_balance FLOAT DEFAULT 0.0")
        try_exec(conn, "ALTER TABLE accounts ADD COLUMN currency TEXT DEFAULT 'BRL'")
        try_exec(conn, "ALTER TABLE accounts ADD COLUMN active BOOLEAN DEFAULT 1")
        try_exec(conn, "ALTER TABLE accounts ADD COLUMN notes TEXT")

        try_exec(conn, "ALTER TABLE categories ADD COLUMN mov_type TEXT DEFAULT 'both'")
        try_exec(conn, "ALTER TABLE categories ADD COLUMN parent_id INTEGER")
        try_exec(conn, "ALTER TABLE categories ADD COLUMN notes TEXT")

        try_exec(conn, "ALTER TABLE clients ADD COLUMN document TEXT")
        try_exec(conn, "ALTER TABLE clients ADD COLUMN notes TEXT")

        try_exec(conn, "ALTER TABLE suppliers ADD COLUMN document TEXT")
        try_exec(conn, "ALTER TABLE suppliers ADD COLUMN notes TEXT")

        try_exec(conn, "ALTER TABLE cost_centers ADD COLUMN notes TEXT")

        try_exec(conn, "ALTER TABLE goals ADD COLUMN target_amount FLOAT DEFAULT 0.0")
        try_exec(conn, "ALTER TABLE goals ADD COLUMN start_date DATE")
        try_exec(conn, "ALTER TABLE goals ADD COLUMN end_date DATE")
        try_exec(conn, "ALTER TABLE goals ADD COLUMN category_id INTEGER")
        try_exec(conn, "ALTER TABLE goals ADD COLUMN notes TEXT")

        try_exec(conn, "ALTER TABLE budgets ADD COLUMN amount FLOAT DEFAULT 0.0")

        try_exec(conn, "ALTER TABLE transactions ADD COLUMN due_date DATE")
        try_exec(conn, "ALTER TABLE transactions ADD COLUMN paid_date DATE")
        try_exec(conn, "ALTER TABLE transactions ADD COLUMN paid BOOLEAN DEFAULT 0")
        try_exec(conn, "ALTER TABLE transactions ADD COLUMN is_transfer BOOLEAN DEFAULT 0")
        try_exec(conn, "ALTER TABLE transactions ADD COLUMN transfer_pair_id TEXT")
        try_exec(conn, "ALTER TABLE transactions ADD COLUMN recurrence_group TEXT")
        try_exec(conn, "ALTER TABLE transactions ADD COLUMN reference TEXT")
        try_exec(conn, "ALTER TABLE transactions ADD COLUMN notes TEXT")

        try_exec(conn, "ALTER TABLE attachments ADD COLUMN content_type TEXT")
        try_exec(conn, "ALTER TABLE attachments ADD COLUMN uploaded_at DATETIME")


def ensure_columns_postgres(engine):
    with engine.begin() as conn:

        def add_if_missing(table: str, column: str, typ: str):
            q = text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = :table AND column_name = :col
            """)
            r = conn.execute(q, {"table": table, "col": column}).fetchone()
            if not r:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {typ}"))

        add_if_missing("accounts", "bank", "text")
        add_if_missing("accounts", "initial_balance", "double precision")
        add_if_missing("accounts", "currency", "text")
        add_if_missing("accounts", "active", "boolean")
        add_if_missing("accounts", "notes", "text")

        add_if_missing("categories", "mov_type", "text")
        add_if_missing("categories", "parent_id", "integer")
        add_if_missing("categories", "notes", "text")

        add_if_missing("clients", "document", "text")
        add_if_missing("clients", "notes", "text")

        add_if_missing("suppliers", "document", "text")
        add_if_missing("suppliers", "notes", "text")

        add_if_missing("cost_centers", "notes", "text")

        add_if_missing("goals", "target_amount", "double precision")
        add_if_missing("goals", "start_date", "date")
        add_if_missing("goals", "end_date", "date")
        add_if_missing("goals", "category_id", "integer")
        add_if_missing("goals", "notes", "text")

        add_if_missing("budgets", "amount", "double precision")

        add_if_missing("transactions", "due_date", "date")
        add_if_missing("transactions", "paid_date", "date")
        add_if_missing("transactions", "paid", "boolean")
        add_if_missing("transactions", "is_transfer", "boolean")
        add_if_missing("transactions", "transfer_pair_id", "text")
        add_if_missing("transactions", "recurrence_group", "text")
        add_if_missing("transactions", "reference", "text")
        add_if_missing("transactions", "notes", "text")

        add_if_missing("attachments", "content_type", "text")
        add_if_missing("attachments", "uploaded_at", "timestamp")


def _mask_db_url(db_url: str) -> str:
    """Diagnóstico SEM expor senha."""
    try:
        p = urlparse(db_url)
        q = dict(parse_qsl(p.query))
        has_pwd = "sim" if (p.password is not None and str(p.password).strip() != "") else "não"
        return json.dumps({
            "dialeto": p.scheme,
            "hospedagem": p.hostname,
            "porta": p.port or 5432,
            "banco_de_dados": (p.path or "").lstrip("/"),
            "modo_ssl": q.get("sslmode", "(ausente)"),
            "usuário": p.username,
            "senha_na_url": has_pwd,
            "PGPASSWORD_setado": "sim" if (os.getenv("PGPASSWORD") or "").strip() else "não",
        }, ensure_ascii=False, indent=2)
    except Exception:
        return "(não foi possível diagnosticar a URL)"


def _ensure_sslmode_require(db_url: str) -> str:
    try:
        p = urlparse(db_url)
        q = dict(parse_qsl(p.query))
        if not q.get("sslmode"):
            q["sslmode"] = "require"
            new_query = urlencode(q)
            return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))
        return db_url
    except Exception:
        return db_url


def _inject_password_if_missing(db_url: str) -> str:
    """
    Se a URL não tiver senha e existir PGPASSWORD, injeta user:senha@host.
    Isso evita inconsistências entre ambientes/driver.
    """
    try:
        p = urlparse(db_url)
        if p.scheme.startswith("sqlite"):
            return db_url

        # já tem senha na URL
        if p.password is not None and str(p.password).strip() != "":
            return db_url

        pgpwd = (os.getenv("PGPASSWORD") or "").strip()
        if not pgpwd:
            return db_url

        user = p.username
        host = p.hostname
        if not user or not host:
            return db_url

        port = p.port
        # reconstroi netloc com senha (urlparse oculta password no .netloc quando não setada)
        if port:
            netloc = f"{user}:{pgpwd}@{host}:{port}"
        else:
            netloc = f"{user}:{pgpwd}@{host}"

        return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))
    except Exception:
        return db_url


def create_sessionmaker_with_schema(db_url: str):
    if db_url.startswith("sqlite"):
        engine = create_engine(db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        ensure_columns_sqlite(engine)
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        return engine, SessionLocal

    db_url = _inject_password_if_missing(db_url)
    db_url = _ensure_sslmode_require(db_url)

    engine = create_engine(
        db_url,
        **ENGINE_KW,
        connect_args={"application_name": "bk-gestao-financeira"},
    )

    # teste de conexão (se falhar, o erro aparece)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    Base.metadata.create_all(engine)
    ensure_columns_postgres(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    return engine, SessionLocal


@st.cache_resource

def _read_database_url_from_secrets_file() -> str:
    """Lê .streamlit/secrets.toml sem depender do Streamlit runtime (útil para jobs/CLI)."""
    if tomllib is None:
        return ""


@st.cache_resource
def _read_bootstrap_from_secrets_file() -> dict:
    """Lê [bootstrap] do .streamlit/secrets.toml (útil para bootstrap sem env)."""
    if tomllib is None:
        return {}
    candidates = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
    ]
    for p in candidates:
        try:
            if p.exists():
                data = tomllib.loads(p.read_text(encoding="utf-8"))
                boot = data.get("bootstrap", {}) or {}
                return {
                    "initial_admin_email": str(boot.get("initial_admin_email", "")).strip().lower(),
                    "initial_admin_password": str(boot.get("initial_admin_password", "")).strip(),
                }
        except Exception:
            continue
    return {}

    candidates = [
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
    ]
    for p in candidates:
        try:
            if p.exists():
                data = tomllib.loads(p.read_text(encoding="utf-8"))
                return str(data.get("general", {}).get("database_url", "")).strip()
        except Exception:
            continue
    return ""


def _resolve_database_url() -> str:
    """Prioridade: env DATABASE_URL -> st.secrets -> arquivo secrets.toml."""
    db_url = os.getenv("DATABASE_URL", "").strip()
    if db_url:
        return db_url

    # Streamlit secrets (quando rodando via `streamlit run`)
    try:
        general = getattr(st, "secrets", {}).get("general", {})
        db_url = str(general.get("database_url", "")).strip()
        if db_url:
            return db_url
    except Exception:
        pass

    # Fallback: ler o arquivo .streamlit/secrets.toml diretamente
    return _read_database_url_from_secrets_file()


def get_db():
    db_url = _resolve_database_url()
    if not db_url:
        st.warning("DATABASE_URL não configurado; usando SQLite local (bk_gestao_local.db). Para usar o Neon, configure .streamlit/secrets.toml ou a variável de ambiente DATABASE_URL.")
        return create_sessionmaker_with_schema(LOCAL_SQLITE)
    try:
        return create_sessionmaker_with_schema(db_url)
    except Exception as e:
        st.error("Falha ao conectar no Postgres/Neon.")
        st.caption("Diagnóstico (sem senha):")
        st.code(_mask_db_url(db_url), language="json")
        st.exception(e)
        st.stop()


# =========================
# AUTH (PBKDF2)
# =========================
def _pbkdf2_hash_password(password: str, salt_b64: Optional[str] = None, iterations: int = 180_000) -> str:
    if not salt_b64:
        salt = os.urandom(16)
        salt_b64 = base64.b64encode(salt).decode("utf-8")
    else:
        salt = base64.b64decode(salt_b64.encode("utf-8"))
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    dk_b64 = base64.b64encode(dk).decode("utf-8")
    return f"pbkdf2_sha256${iterations}${salt_b64}${dk_b64}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        alg, it_s, salt_b64, _ = stored.split("$", 3)
        if alg != "pbkdf2_sha256":
            return False
        iterations = int(it_s)
        check = _pbkdf2_hash_password(password, salt_b64=salt_b64, iterations=iterations)
        return hmac.compare_digest(check, stored)
    except Exception:
        return False


def current_user() -> Optional[dict]:
    return st.session_state.get("auth_user")


def role_allows(user_role: str, needed: str) -> bool:
    order = {"leitura": 0, "financeiro": 1, "diretoria": 2, "admin": 3}
    return order.get(user_role, 0) >= order.get(needed, 0)


def can_view(needed: str) -> bool:
    u = current_user()
    return bool(u) and role_allows(u["role"], needed)


def require_login():
    if not current_user():
        st.stop()


def audit(session, action: str, entity: str, record_id: Optional[str], before: Optional[dict], after: Optional[dict]):
    u = current_user()
    email = u.get("email") if u else None
    al = AuditLog(
        user_email=email,
        action=action,
        entity=entity,
        record_id=str(record_id) if record_id is not None else None,
        before_json=json.dumps(before, ensure_ascii=False) if before else None,
        after_json=json.dumps(after, ensure_ascii=False) if after else None,
    )
    session.add(al)


def ensure_initial_admin(SessionLocal):
    """
    Bootstrap do usuário admin inicial.

    - Lê credenciais em: env -> st.secrets[bootstrap] -> .streamlit/secrets.toml
    - Se o e-mail já existir na tabela users, atualiza a senha (útil para recuperar acesso).
    - Se não existir, cria o usuário (mesmo que já existam outros usuários).
    """
    # Prioridade: env -> st.secrets[bootstrap] -> arquivo secrets.toml
    email = os.getenv("INITIAL_ADMIN_EMAIL", "").strip().lower()
    pwd = os.getenv("INITIAL_ADMIN_PASSWORD", "").strip()

    if (not email) or (not pwd):
        try:
            boot = st.secrets.get("bootstrap", {})
            email = email or str(boot.get("initial_admin_email", "")).strip().lower()
            pwd = pwd or str(boot.get("initial_admin_password", "")).strip()
        except Exception:
            pass

    if (not email) or (not pwd):
        boot2 = _read_bootstrap_from_secrets_file()
        email = email or str(boot2.get("initial_admin_email", "")).strip().lower()
        pwd = pwd or str(boot2.get("initial_admin_password", "")).strip()

    if not email or not pwd:
        return

    session = SessionLocal()
    try:
        # Se o usuário já existe, atualiza a senha (recuperação de acesso)
        u = session.query(User).filter(User.email == email).first()
        if u:
            u.password_hash = _pbkdf2_hash_password(pwd)
            u.active = True
            if not u.role:
                u.role = "admin"
            if not u.name:
                u.name = "Administrador"
            session.commit()
            return

        # Se não existe, cria um novo admin inicial (mesmo se já houver outros usuários)
        u = User(
            name="Administrador",
            email=email,
            role="admin",
            password_hash=_pbkdf2_hash_password(pwd),
            active=True,
        )
        session.add(u)
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()



def login_ui(SessionLocal):
    ensure_initial_admin(SessionLocal)

    st.sidebar.markdown("### 🔐 Acesso")
    if current_user():
        u = current_user()
        st.sidebar.success(f"Logado: {u['email']} ({u['role']})")
        if st.sidebar.button("Sair"):
            st.session_state["auth_user"] = None
            st.rerun()
        return

    with st.sidebar.form("login_form"):
        email = st.text_input("Email").strip().lower()
        pwd = st.text_input("Senha", type="password")
        ok = st.form_submit_button("Entrar")

    if ok:
        session = SessionLocal()
        try:
            user = session.query(User).filter(User.email == email, User.active == True).first()
            if not user or not _verify_password(pwd, user.password_hash):
                st.sidebar.error("Credenciais inválidas.")
                return
            st.session_state["auth_user"] = {"id": user.id, "email": user.email, "role": user.role, "name": user.name}
            st.rerun()
        finally:
            session.close()


# =========================
# HELPERS
# =========================
def format_currency(v: Optional[float]) -> str:
    if v is None:
        return ""
    try:
        return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(v)


def signed_amount(tx_type: str, amount: float) -> float:
    return float(amount) if tx_type == "entrada" else -abs(float(amount))


def st_plotly(fig, height: Optional[int] = None):
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)


def month_add(d: date, n_months: int) -> date:
    if HAS_DATEUTIL:
        return (d + relativedelta(months=n_months))
    y = d.year + (d.month - 1 + n_months) // 12
    m = (d.month - 1 + n_months) % 12 + 1
    day = min(d.day, 28)
    return date(y, m, day)


def tx_effective_date(base: str, t: Transaction) -> Optional[date]:
    if base == "Realizado":
        return t.paid_date if (t.paid and t.paid_date) else None
    if base == "Previsto":
        if t.paid:
            return None
        return t.due_date or t.date
    # Tudo
    if t.paid and t.paid_date:
        return t.paid_date
    return t.due_date or t.date


def get_period_transactions(session, start: date, end: date, base: str) -> List[Transaction]:
    txs = session.query(Transaction).order_by(Transaction.date.asc(), Transaction.id.asc()).all()
    out: List[Transaction] = []
    for t in txs:
        dref = tx_effective_date(base, t)
        if dref is None:
            continue
        if start <= dref <= end:
            out.append(t)
    return out


def build_cashflow_series(session, start: date, end: date, gran: str = "Monthly") -> pd.DataFrame:
    txs = session.query(Transaction).all()
    pred = []
    real = []

    for t in txs:
        if t.paid and t.paid_date:
            if start <= t.paid_date <= end:
                real.append((t.paid_date, signed_amount(t.type, t.amount)))
        else:
            d = t.due_date or t.date
            if d and start <= d <= end:
                pred.append((d, signed_amount(t.type, t.amount)))

    def to_period_label(dt: date, g: str) -> str:
        if g == "Monthly":
            return dt.strftime("%Y-%m")
        if g == "Weekly":
            iso = dt.isocalendar()
            return f"{iso[0]}-W{iso[1]:02d}"
        return dt.strftime("%Y-%m-%d")

    period_index = []
    cur = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    if gran == "Monthly":
        cur = cur.replace(day=1)
        while cur <= end_ts:
            period_index.append(cur.strftime("%Y-%m"))
            cur = pd.to_datetime((cur + pd.DateOffset(months=1)).to_pydatetime()).replace(day=1)
    elif gran == "Weekly":
        cur = cur - pd.Timedelta(days=int(cur.weekday()))
        while cur <= end_ts:
            period_index.append(f"{cur.isocalendar()[0]}-W{cur.isocalendar()[1]:02d}")
            cur = cur + pd.Timedelta(weeks=1)
    else:
        while cur <= end_ts:
            period_index.append(cur.strftime("%Y-%m-%d"))
            cur = cur + pd.Timedelta(days=1)

    pred_map = {p: 0.0 for p in period_index}
    real_map = {p: 0.0 for p in period_index}

    for d, v in pred:
        p = to_period_label(d, gran)
        if p in pred_map:
            pred_map[p] += v

    for d, v in real:
        p = to_period_label(d, gran)
        if p in real_map:
            real_map[p] += v

    rows = []
    cum_pred = 0.0
    cum_real = 0.0
    for p in period_index:
        pv = float(pred_map.get(p, 0.0))
        rv = float(real_map.get(p, 0.0))
        cum_pred += pv
        cum_real += rv
        rows.append({
            "period": p,
            "previsto": pv,
            "realizado": rv,
            "dif": rv - pv,
            "cum_previsto": cum_pred,
            "cum_realizado": cum_real,
        })
    return pd.DataFrame(rows)


def breakdown_by(session, txs: List[Transaction], key: str) -> pd.DataFrame:
    rows = []
    for t in txs:
        if key == "Categoria":
            name = t.category.name if t.category else "Sem categoria"
        elif key == "Centro de Custo":
            name = t.cost_center.name if t.cost_center else "Sem centro de custo"
        elif key == "Conta":
            name = t.account.name if t.account else "Sem conta"
        else:
            name = "N/A"
        rows.append({"Item": name, "Valor": signed_amount(t.type, t.amount)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.groupby("Item", as_index=False)["Valor"].sum().sort_values("Valor")
    return df


def kpis_for_period(txs: List[Transaction]) -> dict:
    entradas = sum(t.amount for t in txs if t.type == "entrada")
    saidas = sum(t.amount for t in txs if t.type == "saida")
    saldo = sum(signed_amount(t.type, t.amount) for t in txs)
    return {"Entradas": entradas, "Saídas": saidas, "Saldo (líquido)": saldo}


def df_download_link_csv(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Baixar CSV</a>'


def df_download_link_xlsx(sheets: List[tuple], filename: str) -> str:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, df in sheets:
            safe = (sheet[:31]) if sheet else "Planilha"
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                pd.DataFrame().to_excel(writer, sheet_name=safe, index=False)
            else:
                df.to_excel(writer, sheet_name=safe, index=False)
    data = output.getvalue()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">📥 Baixar Excel</a>'


def build_status(t: Transaction) -> str:
    if t.paid:
        return "Realizado"
    if t.due_date:
        delta = (t.due_date - date.today()).days
        if delta < 0:
            return "Atrasado"
        if delta <= 7:
            return "A vencer"
        return "Previsto"
    return "Previsto"


# =========================
# RECORRÊNCIA E TRANSFERÊNCIA
# =========================
def create_recurrences(session, base_tx: Transaction, count: int, period: str) -> List[int]:
    created = []
    if count <= 1:
        return created

    group = base_tx.recurrence_group or f"recur_{uuid.uuid4().hex[:10]}"
    base_tx.recurrence_group = group

    for i in range(1, count):
        if period == "weekly":
            d2 = base_tx.date + timedelta(weeks=i)
            due2 = base_tx.due_date + timedelta(weeks=i) if base_tx.due_date else None
        elif period == "biweekly":
            d2 = base_tx.date + timedelta(weeks=2 * i)
            due2 = base_tx.due_date + timedelta(weeks=2 * i) if base_tx.due_date else None
        elif period == "monthly":
            d2 = month_add(base_tx.date, i)
            due2 = month_add(base_tx.due_date, i) if base_tx.due_date else None
        elif period == "yearly":
            if HAS_DATEUTIL:
                d2 = base_tx.date + relativedelta(years=i)
                due2 = base_tx.due_date + relativedelta(years=i) if base_tx.due_date else None
            else:
                d2 = base_tx.date + timedelta(days=365 * i)
                due2 = base_tx.due_date + timedelta(days=365 * i) if base_tx.due_date else None
        else:
            d2 = month_add(base_tx.date, i)
            due2 = month_add(base_tx.due_date, i) if base_tx.due_date else None

        tx2 = Transaction(
            date=d2,
            due_date=due2,
            paid=False,
            paid_date=None,
            description=base_tx.description,
            amount=base_tx.amount,
            type=base_tx.type,
            account_id=base_tx.account_id,
            category_id=base_tx.category_id,
            client_id=base_tx.client_id,
            supplier_id=base_tx.supplier_id,
            cost_center_id=base_tx.cost_center_id,
            is_transfer=base_tx.is_transfer,
            transfer_pair_id=base_tx.transfer_pair_id,
            recurrence_group=group,
            reference=base_tx.reference,
            notes=base_tx.notes,
        )
        session.add(tx2)
        session.flush()
        created.append(tx2.id)

    return created


def create_transfer_pair(
    session,
    tx_date: date,
    amount: float,
    from_account_id: int,
    to_account_id: int,
    description: str,
    paid: bool,
    paid_date: Optional[date],
    due_date: Optional[date],
    category_id: Optional[int],
    cost_center_id: Optional[int],
    reference: Optional[str],
    notes: Optional[str],
) -> tuple:
    pair_id = f"tr_{uuid.uuid4().hex[:12]}"

    tx_out = Transaction(
        date=tx_date,
        due_date=due_date,
        paid=paid,
        paid_date=paid_date if paid else None,
        description=description,
        amount=float(amount),
        type="saida",
        account_id=from_account_id,
        category_id=category_id,
        cost_center_id=cost_center_id,
        is_transfer=True,
        transfer_pair_id=pair_id,
        reference=reference,
        notes=notes,
    )
    session.add(tx_out)
    session.flush()

    tx_in = Transaction(
        date=tx_date,
        due_date=due_date,
        paid=paid,
        paid_date=paid_date if paid else None,
        description=description,
        amount=float(amount),
        type="entrada",
        account_id=to_account_id,
        category_id=category_id,
        cost_center_id=cost_center_id,
        is_transfer=True,
        transfer_pair_id=pair_id,
        reference=reference,
        notes=notes,
    )
    session.add(tx_in)
    session.flush()

    return tx_out.id, tx_in.id


# =========================
# UI: HOME
# =========================
def compute_account_balances(session) -> pd.DataFrame:
    accounts = session.query(Account).order_by(Account.name).all()
    if not accounts:
        return pd.DataFrame()

    paid_txs = session.query(Transaction).filter(Transaction.paid == True).all()
    by_acc = {a.id: float(a.initial_balance or 0.0) for a in accounts}

    for t in paid_txs:
        if t.account_id is None:
            continue
        by_acc[t.account_id] = by_acc.get(t.account_id, 0.0) + signed_amount(t.type, t.amount)

    rows = []
    for a in accounts:
        rows.append({
            "Conta": a.name,
            "Banco": a.bank or "",
            "Moeda": a.currency or "BRL",
            "Ativa": "Sim" if a.active else "Não",
            "Saldo Inicial": float(a.initial_balance or 0.0),
            "Saldo Atual (Realizado)": float(by_acc.get(a.id, float(a.initial_balance or 0.0))),
        })

    df = pd.DataFrame(rows)
    df["Saldo Inicial"] = df["Saldo Inicial"].map(format_currency)
    df["Saldo Atual (Realizado)"] = df["Saldo Atual (Realizado)"].map(format_currency)
    return df


def home_ui(SessionLocal):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='big-title'>Home</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Visão rápida do mês</div>", unsafe_allow_html=True)

    session = SessionLocal()
    try:
        today = date.today()
        month_start = today.replace(day=1)
        month_end = (month_add(month_start, 1) - timedelta(days=1))

        all_txs = session.query(Transaction).all()
        prev = []
        real = []
        for t in all_txs:
            if t.paid and t.paid_date:
                if month_start <= t.paid_date <= month_end:
                    real.append(t)
            else:
                d = t.due_date or t.date
                if month_start <= d <= month_end:
                    prev.append(t)

        previsto_entrada = sum(t.amount for t in prev if t.type == "entrada")
        previsto_saida = sum(t.amount for t in prev if t.type == "saida")
        realizado_entrada = sum(t.amount for t in real if t.type == "entrada")
        realizado_saida = sum(t.amount for t in real if t.type == "saida")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='stat-card'><div class='metric-label'>Previsto (Entrada)</div><div class='metric-value'>{format_currency(previsto_entrada)}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='stat-card'><div class='metric-label'>Previsto (Saída)</div><div class='metric-value'>{format_currency(previsto_saida)}</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='stat-card'><div class='metric-label'>Realizado (Entrada)</div><div class='metric-value'>{format_currency(realizado_entrada)}</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='stat-card'><div class='metric-label'>Realizado (Saída)</div><div class='metric-value'>{format_currency(realizado_saida)}</div></div>", unsafe_allow_html=True)

        st.markdown("### Atrasados (não realizados)")
        st.markdown("### Próximos 15 dias (a pagar / a receber)")
        win_end = today + timedelta(days=15)

        upcoming = session.query(Transaction).filter(
            Transaction.paid == False,
            Transaction.due_date != None,
            Transaction.due_date >= today,
            Transaction.due_date <= win_end
        ).order_by(Transaction.due_date.asc()).all()

        if not upcoming:
            st.write("Sem itens previstos para os próximos 15 dias.")
        else:
            df_u = pd.DataFrame([{
                "Vencimento": t.due_date,
                "Descrição": t.description,
                "Tipo": "A Receber" if t.type == "entrada" else "A Pagar",
                "Valor": float(t.amount),
                "Conta": t.account.name if t.account else "",
            } for t in upcoming])
            c_up1, c_up2 = st.columns(2)
            with c_up1:
                st.markdown("**A Receber (15 dias)**")
                df_r = df_u[df_u["Tipo"]=="A Receber"].copy()
                st.metric("Total", format_currency(df_r["Valor"].sum() if not df_r.empty else 0.0))
                if not df_r.empty:
                    df_r["Valor"] = df_r["Valor"].map(format_currency)
                    st.dataframe(df_r.drop(columns=["Tipo"]), use_container_width=True, height=220)
            with c_up2:
                st.markdown("**A Pagar (15 dias)**")
                df_p = df_u[df_u["Tipo"]=="A Pagar"].copy()
                st.metric("Total", format_currency(df_p["Valor"].sum() if not df_p.empty else 0.0))
                if not df_p.empty:
                    df_p["Valor"] = df_p["Valor"].map(format_currency)
                    st.dataframe(df_p.drop(columns=["Tipo"]), use_container_width=True, height=220)

        st.markdown("### Receitas do mês (realizado)")
        df_real_m = pd.DataFrame([{
            "Data": t.paid_date,
            "Descrição": t.description,
            "Valor": float(t.amount),
            "Cliente": t.client.name if t.client else "",
            "Conta": t.account.name if t.account else "",
        } for t in real if t.type=="entrada"])
        if df_real_m.empty:
            st.caption("Sem receitas realizadas neste mês.")
        else:
            df_real_m["Valor"] = df_real_m["Valor"].map(format_currency)
            st.dataframe(df_real_m, use_container_width=True, height=220)

        overdue = session.query(Transaction).filter(
            Transaction.paid == False,
            Transaction.due_date != None,
            Transaction.due_date < today
        ).order_by(Transaction.due_date.asc()).limit(10).all()

        if not overdue:
            st.write("Sem itens atrasados.")
        else:
            df = pd.DataFrame([{
                "Vencimento": t.due_date,
                "Descrição": t.description,
                "Mov": t.type,
                "Valor": format_currency(t.amount),
                "Conta": t.account.name if t.account else "",
            } for t in overdue])
            st.dataframe(df, use_container_width=True)

        st.markdown("### Saldos por conta (realizado)")
        df_bal = compute_account_balances(session)
        if df_bal.empty:
            st.info("Cadastre ao menos uma conta para visualizar saldos.")
        else:
            st.dataframe(df_bal, use_container_width=True)

        st.markdown("### Fluxo mensal (últimos 9 meses)")
        start = month_add(month_start, -8)
        df_flow = build_cashflow_series(session, start, month_end, "Monthly")
        if not df_flow.empty:
            fig = px.bar(df_flow, x="period", y=["previsto", "realizado"], barmode="group", title="Previsto x Realizado (mensal)")
            st_plotly(fig, height=380)

    except Exception as e:
        st.error(f"Erro na Home: {e}")
    finally:
        session.close()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# UI: CRUD CONTAS
# =========================
def accounts_ui(SessionLocal):
    st.header("🏦 Contas / Bancos")
    session = SessionLocal()

    try:
        st.subheader("Cadastrar / Atualizar")
        with st.form("acc_form"):
            acc_id = st.number_input("ID (para editar, deixe 0 para novo)", min_value=0, step=1, value=0)
            name = st.text_input("Nome da conta *")
            bank = st.text_input("Banco")
            initial_balance = st.number_input("Saldo inicial", value=0.0, step=100.0)
            currency = st.text_input("Moeda", value="BRL")
            active = st.checkbox("Ativa", value=True)
            notes = st.text_area("Observações")
            ok = st.form_submit_button("Salvar")

        if ok:
            if not name.strip():
                st.error("Nome é obrigatório.")
            else:
                if acc_id and acc_id > 0:
                    acc = session.query(Account).get(int(acc_id))
                    if not acc:
                        st.error("ID não encontrado.")
                    else:
                        before = {"name": acc.name, "bank": acc.bank, "initial_balance": acc.initial_balance, "currency": acc.currency, "active": acc.active, "notes": acc.notes}
                        acc.name = name.strip()
                        acc.bank = bank.strip() or None
                        acc.initial_balance = float(initial_balance)
                        acc.currency = currency.strip() or "BRL"
                        acc.active = bool(active)
                        acc.notes = notes.strip() or None
                        audit(session, "UPDATE", "accounts", acc.id, before, {"name": acc.name})
                        session.commit()
                        st.success("Conta atualizada.")
                else:
                    acc = Account(
                        name=name.strip(),
                        bank=bank.strip() or None,
                        initial_balance=float(initial_balance),
                        currency=currency.strip() or "BRL",
                        active=bool(active),
                        notes=notes.strip() or None
                    )
                    session.add(acc)
                    session.flush()
                    audit(session, "CREATE", "accounts", acc.id, None, {"name": acc.name})
                    session.commit()
                    st.success("Conta criada.")

        st.markdown("---")
        st.subheader("Lista de Contas")
        accs = session.query(Account).order_by(Account.active.desc(), Account.name.asc()).all()
        if not accs:
            st.info("Nenhuma conta cadastrada.")
        else:
            df = pd.DataFrame([{
                "ID": a.id,
                "Nome": a.name,
                "Banco": a.bank or "",
                "Ativa": "Sim" if a.active else "Não",
                "Saldo inicial": format_currency(a.initial_balance or 0.0),
                "Moeda": a.currency or "BRL",
                "Obs": (a.notes or "")[:60],
            } for a in accs])
            st.dataframe(df, use_container_width=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                del_id = st.number_input("ID para excluir", min_value=0, step=1, value=0, key="acc_del_id")
            with col2:
                if st.button("Excluir conta (cuidado)", type="secondary"):
                    if del_id and del_id > 0:
                        a = session.query(Account).get(int(del_id))
                        if not a:
                            st.error("ID não encontrado.")
                        else:
                            before = {"name": a.name}
                            session.delete(a)
                            audit(session, "DELETE", "accounts", del_id, before, None)
                            session.commit()
                            st.success("Excluída.")
                            st.rerun()

    except Exception as e:
        session.rollback()
        st.error(f"Erro em Contas: {e}")
    finally:
        session.close()


# =========================
# UI: CRUD CATEGORIAS
# =========================
def categories_ui(SessionLocal):
    st.header("📂 Categorias")
    session = SessionLocal()

    try:
        cats = session.query(Category).order_by(Category.parent_id.asc().nullsfirst(), Category.name.asc()).all()
        cat_map = {c.id: c for c in cats}
        cat_options = ["(Sem pai)"] + [f"{c.id} - {c.name}" for c in cats]

        st.subheader("Cadastrar / Atualizar")
        with st.form("cat_form"):
            cat_id = st.number_input("ID (para editar, deixe 0 para novo)", min_value=0, step=1, value=0)
            name = st.text_input("Nome *")
            mov_type = st.selectbox("Tipo", ["both", "entrada", "saida"], index=0)
            parent_sel = st.selectbox("Categoria Pai", cat_options, index=0)
            notes = st.text_area("Observações")
            ok = st.form_submit_button("Salvar")

        if ok:
            if not name.strip():
                st.error("Nome é obrigatório.")
            else:
                parent_id = None
                if parent_sel != "(Sem pai)":
                    parent_id = int(parent_sel.split(" - ", 1)[0])

                if cat_id and cat_id > 0:
                    c = session.query(Category).get(int(cat_id))
                    if not c:
                        st.error("ID não encontrado.")
                    else:
                        before = {"name": c.name, "mov_type": c.mov_type, "parent_id": c.parent_id}
                        c.name = name.strip()
                        c.mov_type = mov_type
                        c.parent_id = parent_id
                        c.notes = notes.strip() or None
                        audit(session, "UPDATE", "categories", c.id, before, {"name": c.name})
                        session.commit()
                        st.success("Categoria atualizada.")
                else:
                    c = Category(name=name.strip(), mov_type=mov_type, parent_id=parent_id, notes=notes.strip() or None)
                    session.add(c)
                    session.flush()
                    audit(session, "CREATE", "categories", c.id, None, {"name": c.name})
                    session.commit()
                    st.success("Categoria criada.")

        st.markdown("---")
        st.subheader("Lista de Categorias")
        cats = session.query(Category).order_by(Category.parent_id.asc().nullsfirst(), Category.name.asc()).all()
        if not cats:
            st.info("Nenhuma categoria cadastrada.")
        else:
            def parent_name(pid):
                return cat_map[pid].name if pid in cat_map else ""

            df = pd.DataFrame([{
                "ID": c.id,
                "Nome": c.name,
                "Pai": parent_name(c.parent_id) if c.parent_id else "",
                "Tipo": c.mov_type,
                "Obs": (c.notes or "")[:60],
            } for c in cats])
            st.dataframe(df, use_container_width=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                del_id = st.number_input("ID para excluir", min_value=0, step=1, value=0, key="cat_del_id")
            with col2:
                if st.button("Excluir categoria (cuidado)", type="secondary"):
                    if del_id and del_id > 0:
                        c = session.query(Category).get(int(del_id))
                        if not c:
                            st.error("ID não encontrado.")
                        else:
                            if session.query(Category).filter(Category.parent_id == c.id).first():
                                st.error("Não pode excluir: esta categoria tem subcategorias.")
                            else:
                                before = {"name": c.name}
                                session.delete(c)
                                audit(session, "DELETE", "categories", del_id, before, None)
                                session.commit()
                                st.success("Excluída.")
                                st.rerun()

    except Exception as e:
        session.rollback()
        st.error(f"Erro em Categorias: {e}")
    finally:
        session.close()


# =========================
# UI: CLIENTES / PRESTADORES
# =========================
def clients_suppliers_ui(SessionLocal):
    st.header("👥 Clientes / Prestadores")
    session = SessionLocal()

    try:
        tabs = st.tabs(["Clientes", "Prestadores/Fornecedores"])

        with tabs[0]:
            st.subheader("Cadastrar / Atualizar Cliente")
            with st.form("client_form"):
                cid = st.number_input("ID (editar, 0=novo)", min_value=0, step=1, value=0, key="cid")
                name = st.text_input("Nome *", key="cname")
                doc = st.text_input("Documento", key="cdoc")
                notes = st.text_area("Observações", key="cnotes")
                ok = st.form_submit_button("Salvar")

            if ok:
                if not name.strip():
                    st.error("Nome é obrigatório.")
                else:
                    if cid and cid > 0:
                        c = session.query(Client).get(int(cid))
                        if not c:
                            st.error("ID não encontrado.")
                        else:
                            before = {"name": c.name, "document": c.document}
                            c.name = name.strip()
                            c.document = doc.strip() or None
                            c.notes = notes.strip() or None
                            audit(session, "UPDATE", "clients", c.id, before, {"name": c.name})
                            session.commit()
                            st.success("Cliente atualizado.")
                    else:
                        c = Client(name=name.strip(), document=doc.strip() or None, notes=notes.strip() or None)
                        session.add(c)
                        session.flush()
                        audit(session, "CREATE", "clients", c.id, None, {"name": c.name})
                        session.commit()
                        st.success("Cliente criado.")

            st.markdown("---")
            st.subheader("Lista de Clientes")
            clients = session.query(Client).order_by(Client.name.asc()).all()
            df = pd.DataFrame([{"ID": c.id, "Nome": c.name, "Documento": c.document or "", "Obs": (c.notes or "")[:60]} for c in clients])
            st.dataframe(df, use_container_width=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                del_id = st.number_input("ID para excluir", min_value=0, step=1, value=0, key="client_del")
            with col2:
                if st.button("Excluir cliente", type="secondary"):
                    if del_id and del_id > 0:
                        c = session.query(Client).get(int(del_id))
                        if not c:
                            st.error("ID não encontrado.")
                        else:
                            before = {"name": c.name}
                            session.delete(c)
                            audit(session, "DELETE", "clients", del_id, before, None)
                            session.commit()
                            st.success("Excluído.")
                            st.rerun()

        with tabs[1]:
            st.subheader("Cadastrar / Atualizar Prestador/Fornecedor")
            with st.form("sup_form"):
                sid = st.number_input("ID (editar, 0=novo)", min_value=0, step=1, value=0, key="sid")
                name = st.text_input("Nome *", key="sname")
                doc = st.text_input("Documento", key="sdoc")
                notes = st.text_area("Observações", key="snotes")
                ok = st.form_submit_button("Salvar")

            if ok:
                if not name.strip():
                    st.error("Nome é obrigatório.")
                else:
                    if sid and sid > 0:
                        s = session.query(Supplier).get(int(sid))
                        if not s:
                            st.error("ID não encontrado.")
                        else:
                            before = {"name": s.name, "document": s.document}
                            s.name = name.strip()
                            s.document = doc.strip() or None
                            s.notes = notes.strip() or None
                            audit(session, "UPDATE", "suppliers", s.id, before, {"name": s.name})
                            session.commit()
                            st.success("Fornecedor atualizado.")
                    else:
                        s = Supplier(name=name.strip(), document=doc.strip() or None, notes=notes.strip() or None)
                        session.add(s)
                        session.flush()
                        audit(session, "CREATE", "suppliers", s.id, None, {"name": s.name})
                        session.commit()
                        st.success("Fornecedor criado.")

            st.markdown("---")
            st.subheader("Lista de Prestadores/Fornecedores")
            sups = session.query(Supplier).order_by(Supplier.name.asc()).all()
            df = pd.DataFrame([{"ID": s.id, "Nome": s.name, "Documento": s.document or "", "Obs": (s.notes or "")[:60]} for s in sups])
            st.dataframe(df, use_container_width=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                del_id = st.number_input("ID para excluir", min_value=0, step=1, value=0, key="sup_del")
            with col2:
                if st.button("Excluir fornecedor", type="secondary"):
                    if del_id and del_id > 0:
                        s = session.query(Supplier).get(int(del_id))
                        if not s:
                            st.error("ID não encontrado.")
                        else:
                            before = {"name": s.name}
                            session.delete(s)
                            audit(session, "DELETE", "suppliers", del_id, before, None)
                            session.commit()
                            st.success("Excluído.")
                            st.rerun()

    except Exception as e:
        session.rollback()
        st.error(f"Erro em Clientes/Prestadores: {e}")
    finally:
        session.close()


# =========================
# UI: CENTROS DE CUSTO
# =========================
def cost_centers_ui(SessionLocal):
    st.header("🏷️ Centros de Custo")
    session = SessionLocal()

    try:
        st.subheader("Cadastrar / Atualizar")
        with st.form("cc_form"):
            cc_id = st.number_input("ID (editar, 0=novo)", min_value=0, step=1, value=0)
            name = st.text_input("Nome *")
            notes = st.text_area("Observações")
            ok = st.form_submit_button("Salvar")

        if ok:
            if not name.strip():
                st.error("Nome é obrigatório.")
            else:
                if cc_id and cc_id > 0:
                    cc = session.query(CostCenter).get(int(cc_id))
                    if not cc:
                        st.error("ID não encontrado.")
                    else:
                        before = {"name": cc.name}
                        cc.name = name.strip()
                        cc.notes = notes.strip() or None
                        audit(session, "UPDATE", "cost_centers", cc.id, before, {"name": cc.name})
                        session.commit()
                        st.success("Atualizado.")
                else:
                    cc = CostCenter(name=name.strip(), notes=notes.strip() or None)
                    session.add(cc)
                    session.flush()
                    audit(session, "CREATE", "cost_centers", cc.id, None, {"name": cc.name})
                    session.commit()
                    st.success("Criado.")

        st.markdown("---")
        st.subheader("Lista")
        ccs = session.query(CostCenter).order_by(CostCenter.name.asc()).all()
        df = pd.DataFrame([{"ID": c.id, "Nome": c.name, "Obs": (c.notes or "")[:60]} for c in ccs])
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            del_id = st.number_input("ID para excluir", min_value=0, step=1, value=0, key="cc_del")
        with col2:
            if st.button("Excluir centro de custo", type="secondary"):
                if del_id and del_id > 0:
                    cc = session.query(CostCenter).get(int(del_id))
                    if not cc:
                        st.error("ID não encontrado.")
                    else:
                        before = {"name": cc.name}
                        session.delete(cc)
                        audit(session, "DELETE", "cost_centers", del_id, before, None)
                        session.commit()
                        st.success("Excluído.")
                        st.rerun()

    except Exception as e:
        session.rollback()
        st.error(f"Erro em Centros de Custo: {e}")
    finally:
        session.close()


# =========================
# UI: METAS + ORÇAMENTO
# =========================
def goals_ui(SessionLocal):
    st.header("🎯 Metas / Orçamento")
    session = SessionLocal()

    try:
        tabs = st.tabs(["Metas", "Orçamentos (Mensal)"])

        cats = session.query(Category).order_by(Category.name.asc()).all()
        cat_opts = ["(Sem categoria)"] + [f"{c.id} - {c.name}" for c in cats]

        with tabs[0]:
            st.subheader("Cadastrar / Atualizar Meta")
            with st.form("goal_form"):
                gid = st.number_input("ID (editar, 0=novo)", min_value=0, step=1, value=0, key="gid")
                name = st.text_input("Nome *", key="gname")
                target = st.number_input("Valor alvo *", value=0.0, step=100.0, key="gtarget")
                sd = st.date_input("Início", value=date.today().replace(day=1), key="gsd")
                ed = st.date_input("Fim", value=month_add(date.today().replace(day=1), 3), key="ged")
                cat_sel = st.selectbox("Categoria", cat_opts, index=0, key="gcat")
                notes = st.text_area("Observações", key="gnotes")
                ok = st.form_submit_button("Salvar")

            if ok:
                if not name.strip():
                    st.error("Nome é obrigatório.")
                else:
                    cat_id = None
                    if cat_sel != "(Sem categoria)":
                        cat_id = int(cat_sel.split(" - ", 1)[0])

                    if gid and gid > 0:
                        g = session.query(Goal).get(int(gid))
                        if not g:
                            st.error("ID não encontrado.")
                        else:
                            before = {"name": g.name, "target_amount": g.target_amount}
                            g.name = name.strip()
                            g.target_amount = float(target)
                            g.start_date = sd
                            g.end_date = ed
                            g.category_id = cat_id
                            g.notes = notes.strip() or None
                            audit(session, "UPDATE", "goals", g.id, before, {"name": g.name})
                            session.commit()
                            st.success("Meta atualizada.")
                    else:
                        g = Goal(
                            name=name.strip(),
                            target_amount=float(target),
                            start_date=sd,
                            end_date=ed,
                            category_id=cat_id,
                            notes=notes.strip() or None,
                        )
                        session.add(g)
                        session.flush()
                        audit(session, "CREATE", "goals", g.id, None, {"name": g.name})
                        session.commit()
                        st.success("Meta criada.")

            st.markdown("---")
            st.subheader("Lista de Metas")
            goals = session.query(Goal).order_by(Goal.id.desc()).all()
            if goals:
                rows = []
                for g in goals:
                    rows.append({
                        "ID": g.id,
                        "Nome": g.name,
                        "Valor Alvo": format_currency(g.target_amount),
                        "Início": g.start_date,
                        "Fim": g.end_date,
                        "Categoria": g.category.name if g.category else "",
                        "Obs": (g.notes or "")[:60],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("Sem metas cadastradas.")

            col1, col2 = st.columns([1, 2])
            with col1:
                del_id = st.number_input("ID para excluir", min_value=0, step=1, value=0, key="goal_del")
            with col2:
                if st.button("Excluir meta", type="secondary"):
                    if del_id and del_id > 0:
                        g = session.query(Goal).get(int(del_id))
                        if not g:
                            st.error("ID não encontrado.")
                        else:
                            before = {"name": g.name}
                            session.delete(g)
                            audit(session, "DELETE", "goals", del_id, before, None)
                            session.commit()
                            st.success("Excluída.")
                            st.rerun()

        with tabs[1]:
            st.subheader("Cadastrar / Atualizar Orçamento (Categoria x Mês)")
            years = list(range(date.today().year - 1, date.today().year + 3))
            months = list(range(1, 13))

            with st.form("budget_form"):
                bid = st.number_input("ID (editar, 0=novo)", min_value=0, step=1, value=0, key="bid")
                cat_sel = st.selectbox("Categoria *", cat_opts, index=0, key="bcat")
                year = st.selectbox("Ano", years, index=1, key="byear")
                month = st.selectbox("Mês", months, index=date.today().month - 1, key="bmonth")
                amount = st.number_input("Valor (R$)", value=0.0, step=100.0, key="bamt")
                ok = st.form_submit_button("Salvar")

            if ok:
                if cat_sel == "(Sem categoria)":
                    st.error("Escolha uma categoria.")
                else:
                    cat_id = int(cat_sel.split(" - ", 1)[0])
                    if bid and bid > 0:
                        b = session.query(Budget).get(int(bid))
                        if not b:
                            st.error("ID não encontrado.")
                        else:
                            before = {"category_id": b.category_id, "year": b.year, "month": b.month, "amount": b.amount}
                            b.category_id = cat_id
                            b.year = int(year)
                            b.month = int(month)
                            b.amount = float(amount)
                            audit(session, "UPDATE", "budgets", b.id, before, {"amount": b.amount})
                            session.commit()
                            st.success("Orçamento atualizado.")
                    else:
                        exists = session.query(Budget).filter(
                            Budget.category_id == cat_id,
                            Budget.year == int(year),
                            Budget.month == int(month),
                        ).first()
                        if exists:
                            before = {"amount": exists.amount}
                            exists.amount = float(amount)
                            audit(session, "UPDATE", "budgets", exists.id, before, {"amount": exists.amount})
                            session.commit()
                            st.success("Orçamento atualizado (registro existente).")
                        else:
                            b = Budget(category_id=cat_id, year=int(year), month=int(month), amount=float(amount))
                            session.add(b)
                            session.flush()
                            audit(session, "CREATE", "budgets", b.id, None, {"amount": b.amount})
                            session.commit()
                            st.success("Orçamento criado.")

            st.markdown("---")
            st.subheader("Lista de Orçamentos")
            buds = session.query(Budget).order_by(Budget.year.desc(), Budget.month.desc()).limit(500).all()
            if buds:
                df = pd.DataFrame([{
                    "ID": b.id,
                    "Ano": b.year,
                    "Mês": b.month,
                    "Categoria": b.category.name if b.category else "",
                    "Valor": format_currency(b.amount),
                } for b in buds])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Sem orçamentos cadastrados.")

            col1, col2 = st.columns([1, 2])
            with col1:
                del_id = st.number_input("ID para excluir", min_value=0, step=1, value=0, key="bud_del")
            with col2:
                if st.button("Excluir orçamento", type="secondary"):
                    if del_id and del_id > 0:
                        b = session.query(Budget).get(int(del_id))
                        if not b:
                            st.error("ID não encontrado.")
                        else:
                            before = {"amount": b.amount}
                            session.delete(b)
                            audit(session, "DELETE", "budgets", del_id, before, None)
                            session.commit()
                            st.success("Excluído.")
                            st.rerun()

    except Exception as e:
        session.rollback()
        st.error(f"Erro em Metas/Orçamento: {e}")
    finally:
        session.close()


# =========================
# UI: LANÇAMENTOS (CADASTRO + LISTA + EDITAR + PAGAR + ANEXO)
# =========================
def add_transaction_ui(SessionLocal):
    st.header("➕ Registrar Lançamento")
    session = SessionLocal()

    try:
        accounts = session.query(Account).filter(Account.active == True).order_by(Account.name.asc()).all()
        categories = session.query(Category).order_by(Category.name.asc()).all()
        clients = session.query(Client).order_by(Client.name.asc()).all()
        suppliers = session.query(Supplier).order_by(Supplier.name.asc()).all()
        ccs = session.query(CostCenter).order_by(CostCenter.name.asc()).all()

        acc_opts = ["(Sem conta)"] + [f"{a.id} - {a.name}" for a in accounts]
        cat_opts = ["(Sem categoria)"] + [f"{c.id} - {c.name}" for c in categories]
        cli_opts = ["(Sem cliente)"] + [f"{c.id} - {c.name}" for c in clients]
        sup_opts = ["(Sem fornecedor)"] + [f"{s.id} - {s.name}" for s in suppliers]
        cc_opts = ["(Sem centro de custo)"] + [f"{c.id} - {c.name}" for c in ccs]

        st.subheader("Novo lançamento")
        with st.form("tx_form"):
            tx_type = st.selectbox("Tipo", ["entrada", "saida", "transferencia"], index=1)
            tx_date = st.date_input("Data (competência)", value=date.today())
            due_date = st.date_input("Vencimento (opcional)", value=date.today())
            use_due = st.checkbox("Usar vencimento", value=True)

            paid = st.checkbox("Já está realizado?", value=False)
            paid_date = st.date_input("Data de pagamento/recebimento", value=date.today())

            description = st.text_input("Descrição *")
            amount = st.number_input("Valor *", value=0.0, step=100.0)

            if tx_type == "transferencia":
                from_acc = st.selectbox("De (Conta)", acc_opts, index=1 if len(acc_opts) > 1 else 0)
                to_acc = st.selectbox("Para (Conta)", acc_opts, index=1 if len(acc_opts) > 1 else 0, key="to_acc")
            else:
                account_sel = st.selectbox("Conta", acc_opts, index=1 if len(acc_opts) > 1 else 0)

            category_sel = st.selectbox("Categoria", cat_opts, index=0)
            cost_center_sel = st.selectbox("Centro de Custo", cc_opts, index=0)

            client_sel = st.selectbox("Cliente (para entradas)", cli_opts, index=0)
            supplier_sel = st.selectbox("Fornecedor (para saídas)", sup_opts, index=0)

            reference = st.text_input("Referência (NF, boleto, contrato)")
            notes = st.text_area("Observações")

            st.markdown("**Recorrência (opcional)**")
            rec_enable = st.checkbox("Criar recorrência", value=False)
            rec_count = st.number_input("Quantidade total", min_value=1, value=1, step=1)
            rec_period = st.selectbox("Periodicidade", ["monthly", "weekly", "biweekly", "yearly"], index=0)

            ok = st.form_submit_button("Salvar lançamento")

        if ok:
            if not description.strip():
                st.error("Descrição é obrigatória.")
            elif amount <= 0:
                st.error("Valor deve ser maior que 0.")
            else:
                due = due_date if use_due else None
                pd_dt = paid_date if paid else None

                cat_id = int(category_sel.split(" - ", 1)[0]) if category_sel != "(Sem categoria)" else None
                cc_id = int(cost_center_sel.split(" - ", 1)[0]) if cost_center_sel != "(Sem centro de custo)" else None
                cli_id = int(client_sel.split(" - ", 1)[0]) if client_sel != "(Sem cliente)" else None
                sup_id = int(supplier_sel.split(" - ", 1)[0]) if supplier_sel != "(Sem fornecedor)" else None

                if tx_type == "transferencia":
                    if from_acc == "(Sem conta)" or to_acc == "(Sem conta)":
                        st.error("Transferência precisa de conta origem e destino.")
                    else:
                        from_id = int(from_acc.split(" - ", 1)[0])
                        to_id = int(to_acc.split(" - ", 1)[0])
                        if from_id == to_id:
                            st.error("Conta origem e destino não podem ser a mesma.")
                        else:
                            tx_out_id, tx_in_id = create_transfer_pair(
                                session=session,
                                tx_date=tx_date,
                                amount=float(amount),
                                from_account_id=from_id,
                                to_account_id=to_id,
                                description=description.strip(),
                                paid=bool(paid),
                                paid_date=pd_dt,
                                due_date=due,
                                category_id=cat_id,
                                cost_center_id=cc_id,
                                reference=reference.strip() or None,
                                notes=notes.strip() or None,
                            )
                            audit(session, "CREATE", "transactions", tx_out_id, None, {"transfer": True})
                            audit(session, "CREATE", "transactions", tx_in_id, None, {"transfer": True})
                            session.commit()
                            st.success(f"Transferência registrada (IDs {tx_out_id} e {tx_in_id}).")
                else:
                    acc_id = int(account_sel.split(" - ", 1)[0]) if account_sel != "(Sem conta)" else None
                    tx = Transaction(
                        date=tx_date,
                        due_date=due,
                        paid=bool(paid),
                        paid_date=pd_dt,
                        description=description.strip(),
                        amount=float(amount),
                        type=tx_type,
                        account_id=acc_id,
                        category_id=cat_id,
                        client_id=cli_id if tx_type == "entrada" else None,
                        supplier_id=sup_id if tx_type == "saida" else None,
                        cost_center_id=cc_id,
                        is_transfer=False,
                        reference=reference.strip() or None,
                        notes=notes.strip() or None,
                    )
                    session.add(tx)
                    session.flush()

                    if rec_enable and int(rec_count) > 1:
                        tx.recurrence_group = f"recur_{uuid.uuid4().hex[:10]}"
                        created = create_recurrences(session, tx, int(rec_count), rec_period)
                        audit(session, "CREATE", "transactions", tx.id, None, {"recurrence": True, "created_more": len(created)})
                    else:
                        audit(session, "CREATE", "transactions", tx.id, None, {"description": tx.description})

                    session.commit()
                    st.success(f"Lançamento registrado (ID {tx.id}).")

        st.markdown("---")
        st.subheader("Pesquisar / Listar lançamentos")

        f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
        with f1:
            start = st.date_input("Data inicial", value=date.today().replace(day=1), key="tx_list_start")
        with f2:
            end = st.date_input("Data final", value=date.today(), key="tx_list_end")
        with f3:
            base = st.selectbox("Base", ["Tudo", "Realizado", "Previsto"], index=0, key="tx_list_base")
        with f4:
            q = st.text_input("Busca (descrição/ref)", value="", key="tx_list_q")

        if end < start:
            st.warning("A data final deve ser maior ou igual à data inicial.")
            return

        all_txs = session.query(Transaction).order_by(Transaction.date.desc(), Transaction.id.desc()).limit(3000).all()

        filtered = []
        for t in all_txs:
            dref = tx_effective_date(base, t) if base != "Tudo" else tx_effective_date("Tudo", t)
            if dref is None:
                continue
            if not (start <= dref <= end):
                continue
            if q.strip():
                qq = q.strip().lower()
                if qq not in (t.description or "").lower() and qq not in (t.reference or "").lower():
                    continue
            filtered.append(t)

        st.caption(f"Mostrando {len(filtered)} registros (limite interno 3000).")

        rows = []
        for t in filtered:
            rows.append({
                "ID": t.id,
                "Status": build_status(t),
                "Data Ref.": tx_effective_date(base if base != "Tudo" else "Tudo", t),
                "Competência": t.date,
                "Vencimento": t.due_date or "",
                "Pagamento": t.paid_date or "",
                "Descrição": t.description,
                "Mov": t.type,
                "Valor": float(t.amount),
                "Conta": t.account.name if t.account else "",
                "Categoria": t.category.name if t.category else "",
                "Centro de Custo": t.cost_center.name if t.cost_center else "",
                "Cliente": t.client.name if t.client else "",
                "Fornecedor": t.supplier.name if t.supplier else "",
                "Ref": t.reference or "",
                "Transfer?": "Sim" if t.is_transfer else "Não",
                "Grupo Recorrência": t.recurrence_group or "",
            })

        df_ext = pd.DataFrame(rows)
        if df_ext.empty:
            st.info("Sem lançamentos no filtro.")
        else:
            df_show = df_ext.copy()
            df_show["Valor"] = df_show["Valor"].map(format_currency)
            st.dataframe(df_show, use_container_width=True)

            cexp1, cexp2 = st.columns([1, 1])
            with cexp1:
                st.markdown(df_download_link_csv(df_ext, "extrato.csv"), unsafe_allow_html=True)
            with cexp2:
                st.markdown(df_download_link_xlsx([("Extrato", df_ext)], "extrato.xlsx"), unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Ações rápidas (Editar / Marcar pago / Anexos / Excluir)")

        colA, colB = st.columns([1, 2])
        with colA:
            action_id = st.number_input("ID do lançamento", min_value=0, step=1, value=0, key="act_id")
        with colB:
            action = st.selectbox("Ação", ["Editar", "Marcar pago", "Marcar não pago", "Anexar arquivo", "Ver anexos", "Excluir"], index=0)

        if action_id and action_id > 0:
            t = session.query(Transaction).get(int(action_id))
            if not t:
                st.error("ID não encontrado.")
            else:
                if action == "Editar":
                    st.info("Edite abaixo e salve.")
                    with st.form("edit_tx_form"):
                        e_date = st.date_input("Competência", value=t.date)
                        e_due = st.date_input("Vencimento", value=t.due_date or t.date)
                        e_use_due = st.checkbox("Usar vencimento", value=bool(t.due_date))
                        e_paid = st.checkbox("Realizado", value=bool(t.paid))
                        e_paid_date = st.date_input("Pagamento/Recebimento", value=t.paid_date or date.today())
                        e_desc = st.text_input("Descrição", value=t.description)
                        e_amount = st.number_input("Valor", value=float(t.amount), step=100.0)
                        e_type = st.selectbox("Mov", ["entrada", "saida"], index=0 if t.type == "entrada" else 1)

                        acc_all = session.query(Account).order_by(Account.name.asc()).all()
                        acc_opts2 = ["(Sem conta)"] + [f"{a.id} - {a.name}" for a in acc_all]
                        idx_acc = 0
                        if t.account_id and t.account:
                            target = f"{t.account_id} - {t.account.name}"
                            if target in acc_opts2:
                                idx_acc = acc_opts2.index(target)
                        e_acc = st.selectbox("Conta", acc_opts2, index=idx_acc)

                        cat_all = session.query(Category).order_by(Category.name.asc()).all()
                        cat_opts2 = ["(Sem categoria)"] + [f"{c.id} - {c.name}" for c in cat_all]
                        idx_cat = 0
                        if t.category_id and t.category:
                            target = f"{t.category_id} - {t.category.name}"
                            if target in cat_opts2:
                                idx_cat = cat_opts2.index(target)
                        e_cat = st.selectbox("Categoria", cat_opts2, index=idx_cat)

                        cc_all = session.query(CostCenter).order_by(CostCenter.name.asc()).all()
                        cc_opts2 = ["(Sem centro de custo)"] + [f"{c.id} - {c.name}" for c in cc_all]
                        idx_cc = 0
                        if t.cost_center_id and t.cost_center:
                            target = f"{t.cost_center_id} - {t.cost_center.name}"
                            if target in cc_opts2:
                                idx_cc = cc_opts2.index(target)
                        e_cc = st.selectbox("Centro de Custo", cc_opts2, index=idx_cc)

                        e_ref = st.text_input("Referência", value=t.reference or "")
                        e_notes = st.text_area("Observações", value=t.notes or "")
                        ok2 = st.form_submit_button("Salvar alterações")

                    if ok2:
                        before = {"description": t.description, "amount": t.amount, "paid": t.paid, "paid_date": str(t.paid_date) if t.paid_date else None}
                        t.date = e_date
                        t.due_date = (e_due if e_use_due else None)
                        t.paid = bool(e_paid)
                        t.paid_date = (e_paid_date if e_paid else None)
                        t.description = e_desc.strip()
                        t.amount = float(e_amount)
                        t.type = e_type

                        if e_acc != "(Sem conta)":
                            t.account_id = int(e_acc.split(" - ", 1)[0])
                        else:
                            t.account_id = None

                        t.category_id = int(e_cat.split(" - ", 1)[0]) if e_cat != "(Sem categoria)" else None
                        t.cost_center_id = int(e_cc.split(" - ", 1)[0]) if e_cc != "(Sem centro de custo)" else None
                        t.reference = e_ref.strip() or None
                        t.notes = e_notes.strip() or None

                        audit(session, "UPDATE", "transactions", t.id, before, {"description": t.description})
                        session.commit()
                        st.success("Atualizado.")
                        st.rerun()

                elif action == "Marcar pago":
                    before = {"paid": t.paid, "paid_date": str(t.paid_date) if t.paid_date else None}
                    t.paid = True
                    t.paid_date = date.today()
                    audit(session, "UPDATE", "transactions", t.id, before, {"paid": True, "paid_date": str(t.paid_date)})
                    session.commit()
                    st.success("Marcado como pago/recebido (hoje).")
                    st.rerun()

                elif action == "Marcar não pago":
                    before = {"paid": t.paid, "paid_date": str(t.paid_date) if t.paid_date else None}
                    t.paid = False
                    t.paid_date = None
                    audit(session, "UPDATE", "transactions", t.id, before, {"paid": False})
                    session.commit()
                    st.success("Marcado como NÃO pago.")
                    st.rerun()

                elif action == "Anexar arquivo":
                    up = st.file_uploader("Selecione o arquivo", key="tx_attach_up")
                    if up is not None:
                        if st.button("Salvar anexo", type="primary"):
                            data = up.read()
                            att = Attachment(
                                transaction_id=t.id,
                                filename=up.name,
                                content_type=getattr(up, "type", None),
                                data=data,
                            )
                            session.add(att)
                            audit(session, "CREATE", "attachments", None, None, {"tx_id": t.id, "filename": up.name})
                            session.commit()
                            st.success("Anexo salvo.")
                            st.rerun()

                elif action == "Ver anexos":
                    atts = session.query(Attachment).filter(Attachment.transaction_id == t.id).order_by(Attachment.uploaded_at.desc()).all()
                    if not atts:
                        st.info("Sem anexos.")
                    else:
                        for a in atts:
                            st.markdown(f"**{a.filename}** — {a.uploaded_at}")
                            st.download_button(
                                label=f"Baixar {a.filename}",
                                data=a.data,
                                file_name=a.filename,
                                mime=a.content_type or "application/octet-stream",
                                key=f"dl_{a.id}"
                            )
                            if st.button("Excluir anexo", key=f"del_att_{a.id}", type="secondary"):
                                before = {"filename": a.filename, "tx_id": a.transaction_id}
                                session.delete(a)
                                audit(session, "DELETE", "attachments", a.id, before, None)
                                session.commit()
                                st.success("Anexo excluído.")
                                st.rerun()
                            st.markdown("---")

                elif action == "Excluir":
                    st.error("Confirme a exclusão (irreversível).")
                    if st.button("CONFIRMAR EXCLUSÃO", type="primary"):
                        before = {"description": t.description, "amount": t.amount}
                        session.query(Attachment).filter(Attachment.transaction_id == t.id).delete()
                        session.delete(t)
                        audit(session, "DELETE", "transactions", t.id, before, None)
                        session.commit()
                        st.success("Lançamento excluído.")
                        st.rerun()

    except Exception as e:
        session.rollback()
        st.error(f"Erro em Lançamentos: {e}")
    finally:
        session.close()


# =========================
# UI: DASHBOARDS & RELATÓRIOS
# =========================
def dashboards_ui(SessionLocal):
    session = SessionLocal()
    st.header("📊 Dashboards & Relatórios")

    try:
        sub = st.tabs(["📈 Dashboard", "🧾 Relatórios"])

        with sub[0]:
            c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
            with c1:
                start = st.date_input("Data inicial", value=date.today().replace(day=1), key="dash_start")
            with c2:
                end = st.date_input("Data final", value=date.today(), key="dash_end")
            with c3:
                gran = st.selectbox("Granularidade", options=["Monthly", "Weekly", "Daily"], key="dash_gran")
            with c4:
                base = st.selectbox("Base dos números", options=["Realizado", "Previsto", "Tudo"], index=2, key="dash_base")

            if end < start:
                st.warning("A data final deve ser maior ou igual à data inicial.")
                return

            txs_period = get_period_transactions(session, start, end, base)
            k = kpis_for_period(txs_period)

            saldo_inicial_total = sum(float(a.initial_balance or 0.0) for a in session.query(Account).all())
            paid_txs = session.query(Transaction).filter(Transaction.paid == True).all()
            saldo_atual_total = saldo_inicial_total + sum(
                signed_amount(t.type, t.amount) for t in paid_txs if t.account_id is not None
            )

            kc1, kc2, kc3, kc4 = st.columns(4)
            with kc1:
                st.markdown(f"<div class='stat-card'><div class='metric-label'>Saldo Inicial (Contas)</div><div class='metric-value'>{format_currency(saldo_inicial_total)}</div></div>", unsafe_allow_html=True)
            with kc2:
                st.markdown(f"<div class='stat-card'><div class='metric-label'>Saldo Atual (Realizado)</div><div class='metric-value'>{format_currency(saldo_atual_total)}</div></div>", unsafe_allow_html=True)
            with kc3:
                st.markdown(f"<div class='stat-card'><div class='metric-label'>Entradas ({base})</div><div class='metric-value'>{format_currency(k['Entradas'])}</div></div>", unsafe_allow_html=True)
            with kc4:
                st.markdown(f"<div class='stat-card'><div class='metric-label'>Saídas ({base})</div><div class='metric-value'>{format_currency(k['Saídas'])}</div></div>", unsafe_allow_html=True)

            st.markdown("### Fluxo de Caixa (Previsto x Realizado)")
            df_period = build_cashflow_series(session, start, end, gran)
            if df_period.empty:
                st.info("Nenhuma movimentação para o período.")
            else:
                show_mode = st.radio(
                    "Exibição",
                    options=["Comparativo (Previsto x Realizado)", "Somente Realizado", "Somente Previsto"],
                    horizontal=True
                )

                fig = go.Figure()
                if show_mode in ("Comparativo (Previsto x Realizado)", "Somente Previsto"):
                    fig.add_trace(go.Bar(x=df_period["period"], y=df_period["previsto"], name="Previsto"))
                if show_mode in ("Comparativo (Previsto x Realizado)", "Somente Realizado"):
                    fig.add_trace(go.Bar(x=df_period["period"], y=df_period["realizado"], name="Realizado"))

                fig.add_trace(go.Scatter(x=df_period["period"], y=df_period["cum_previsto"], mode="lines+markers", name="Acumulado Previsto", yaxis="y2"))
                fig.add_trace(go.Scatter(x=df_period["period"], y=df_period["cum_realizado"], mode="lines+markers", name="Acumulado Realizado", yaxis="y2"))

                fig.update_layout(
                    title="Fluxo de Caixa (com acumulados)",
                    xaxis_title="Período",
                    yaxis_title="Valor",
                    legend=dict(orientation="h"),
                    yaxis2=dict(title="Saldo acumulado", overlaying="y", side="right")
                )
                st_plotly(fig, height=440)

                show = df_period.copy()
                for c in ["previsto", "realizado", "dif", "cum_previsto", "cum_realizado"]:
                    show[c] = show[c].map(format_currency)
                st.subheader("Tabela (Fluxo de Caixa)")
                st.dataframe(show, use_container_width=True)

            st.markdown("### Visão por Centro de Custo")
            df_cc = breakdown_by(session, txs_period, "Centro de Custo")
            if df_cc.empty:
                st.write("Sem dados para o período/base selecionados.")
            else:
                df_cc["Abs"] = df_cc["Valor"].abs()
                df_cc2 = df_cc.sort_values("Abs", ascending=False).drop(columns=["Abs"]).head(20)
                figcc = px.bar(df_cc2, x="Valor", y="Item", orientation="h", title=f"Centro de Custo ({base}) - Top 20")
                st_plotly(figcc, height=520)

            st.markdown("### Saldos por Conta (Realizado)")
            df_bal = compute_account_balances(session)
            if df_bal.empty:
                st.info("Cadastre contas para ver saldos.")
            else:
                st.dataframe(df_bal, use_container_width=True)

        with sub[1]:
            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                start_r = st.date_input("Data inicial (Relatórios)", value=date.today().replace(day=1), key="rep_start")
            with c2:
                end_r = st.date_input("Data final (Relatórios)", value=date.today(), key="rep_end")
            with c3:
                base_r = st.selectbox("Base", options=["Realizado", "Previsto", "Tudo"], index=2, key="rep_base")

            if end_r < start_r:
                st.warning("A data final deve ser maior ou igual à data inicial.")
                return

            txs_r = get_period_transactions(session, start_r, end_r, base_r)

            st.subheader("Relatório por Categoria")
            df_cat = breakdown_by(session, txs_r, "Categoria")
            if df_cat.empty:
                st.write("Sem dados.")
            else:
                df_cat["Abs"] = df_cat["Valor"].abs()
                df_top = df_cat.sort_values("Abs", ascending=False).drop(columns=["Abs"]).head(20)
                fig2 = px.bar(df_top, x="Valor", y="Item", orientation="h", title=f"Top Categorias ({base_r})")
                st_plotly(fig2, height=520)
                st.dataframe(df_cat.rename(columns={"Item": "Categoria"}), use_container_width=True)

            st.markdown("---")
            st.subheader("Relatório por Centro de Custo")
            df_cc2 = breakdown_by(session, txs_r, "Centro de Custo")
            if df_cc2.empty:
                st.write("Sem dados.")
            else:
                df_cc2["Abs"] = df_cc2["Valor"].abs()
                df_topcc = df_cc2.sort_values("Abs", ascending=False).drop(columns=["Abs"]).head(20)
                fig3 = px.bar(df_topcc, x="Valor", y="Item", orientation="h", title=f"Top Centros de Custo ({base_r})")
                st_plotly(fig3, height=520)
                st.dataframe(df_cc2.rename(columns={"Item": "Centro de Custo"}), use_container_width=True)

            st.markdown("---")
            st.subheader("Extrato (para exportação)")

            rows = []
            for t in txs_r:
                rows.append({
                    "Base": base_r,
                    "Data Ref.": tx_effective_date(base_r if base_r != "Tudo" else "Tudo", t),
                    "Competência": t.date,
                    "Vencimento": t.due_date or "",
                    "Pagamento": t.paid_date or "",
                    "Descrição": t.description,
                    "Mov": t.type,
                    "Valor": float(t.amount),
                    "Conta": t.account.name if t.account else "",
                    "Categoria": t.category.name if t.category else "",
                    "Centro de Custo": t.cost_center.name if t.cost_center else "",
                    "Cliente": t.client.name if t.client else "",
                    "Fornecedor": t.supplier.name if t.supplier else "",
                    "Realizado?": "Sim" if t.paid else "Não",
                })

            df_ext = pd.DataFrame(rows)
            if df_ext.empty:
                st.write("Sem dados.")
            else:
                df_show = df_ext.copy()
                df_show["Valor"] = df_show["Valor"].map(format_currency)
                st.dataframe(df_show, use_container_width=True)

                cexp1, cexp2 = st.columns([1, 1])
                with cexp1:
                    st.markdown(df_download_link_csv(df_ext, "relatorio_extrato.csv"), unsafe_allow_html=True)
                with cexp2:
                    st.markdown(df_download_link_xlsx([
                        ("Extrato", df_ext),
                        ("Por Categoria", df_cat.rename(columns={"Item": "Categoria"}) if not df_cat.empty else pd.DataFrame()),
                        ("Por Centro de Custo", df_cc2.rename(columns={"Item": "Centro de Custo"}) if not df_cc2.empty else pd.DataFrame()),
                    ], "relatorios.xlsx"), unsafe_allow_html=True)

    except Exception as e:
        session.rollback()
        st.error(f"Erro dashboards/relatórios: {e}")
    finally:
        session.close()


# =========================
# MAIN
# =========================
def main():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='big-title'>BK - Gestão Financeira</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Lançamentos, dashboards, relatórios, metas e orçamento</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    engine, SessionLocal = get_db()
    st.sidebar.caption(f"DB: {engine.dialect.name}")

    login_ui(SessionLocal)
    require_login()

    tab_defs = [
        ("🏠 Home", "leitura"),
        ("➕ Registrar Lançamento", "financeiro"),
        ("📊 Painéis e Relatórios", "diretoria"),
        ("🏦 Contas / Bancos", "admin"),
        ("📂 Categorias", "admin"),
        ("👥 Clientes / Prestadores", "admin"),
        ("🏷️ Centros de Custo", "admin"),
        ("🎯 Metas / Orçamento", "admin"),
    ]

    visible = [(name, need) for (name, need) in tab_defs if can_view(need)]
    tabs = st.tabs([v[0] for v in visible])

    for i, (tab_name, _) in enumerate(visible):
        with tabs[i]:
            if tab_name == "🏠 Home":
                home_ui(SessionLocal)
            elif tab_name == "➕ Registrar Lançamento":
                add_transaction_ui(SessionLocal)
            elif tab_name == "📊 Painéis e Relatórios":
                dashboards_ui(SessionLocal)
            elif tab_name == "🏦 Contas / Bancos":
                accounts_ui(SessionLocal)
            elif tab_name == "📂 Categorias":
                categories_ui(SessionLocal)
            elif tab_name == "👥 Clientes / Prestadores":
                clients_suppliers_ui(SessionLocal)
            elif tab_name == "🏷️ Centros de Custo":
                cost_centers_ui(SessionLocal)
            elif tab_name == "🎯 Metas / Orçamento":
                goals_ui(SessionLocal)

    st.markdown("<div class='footer'>Desenvolvido pela BK Engenharia e Tecnologia</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()



# -------------------------------------------------------------------
# Compat helpers para BK_ERP (Cadastros)
# O BK_ERP separa Clientes e Fornecedores em abas e espera estas funções.
# -------------------------------------------------------------------

def clients_ui(SessionLocal):
    """UI de cadastro/listagem de Clientes (tabela: clients)."""
    st.subheader("👥 Clientes")
    session = SessionLocal()
    try:
        with st.form("bk_erp_client_form", clear_on_submit=False):
            cid = st.number_input("ID (editar, 0=novo)", min_value=0, step=1, value=0)
            name = st.text_input("Nome *")
            doc = st.text_input("Documento (CPF/CNPJ)")
            notes = st.text_area("Observações", height=80)
            ok = st.form_submit_button("Salvar")

        if ok:
            if not name.strip():
                st.error("Nome é obrigatório.")
            else:
                if cid and int(cid) > 0:
                    c = session.query(Client).get(int(cid))
                    if not c:
                        st.error("ID não encontrado.")
                    else:
                        c.name = name.strip()
                        c.document = (doc.strip() or None)
                        c.notes = (notes.strip() or None)
                        session.commit()
                        st.success("Cliente atualizado.")
                else:
                    c = Client(name=name.strip(), document=(doc.strip() or None), notes=(notes.strip() or None))
                    session.add(c)
                    session.commit()
                    st.success(f"Cliente criado (ID {c.id}).")

        # Listagem
        df = pd.read_sql(text("SELECT id, name, document, notes FROM clients ORDER BY name"), session.bind)
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("🗑️ Excluir cliente"):
            did = st.number_input("ID para excluir", min_value=0, step=1, value=0, key="del_client_id")
            if st.button("Excluir", key="btn_del_client"):
                if did and int(did) > 0:
                    c = session.query(Client).get(int(did))
                    if not c:
                        st.error("ID não encontrado.")
                    else:
                        session.delete(c)
                        session.commit()
                        st.success("Cliente excluído.")
                        st.rerun()
    finally:
        session.close()


def suppliers_ui(SessionLocal):
    """UI de cadastro/listagem de Fornecedores/Prestadores (tabela: suppliers)."""
    st.subheader("🏭 Fornecedores / Prestadores")
    session = SessionLocal()
    try:
        with st.form("bk_erp_supplier_form", clear_on_submit=False):
            sid = st.number_input("ID (editar, 0=novo)", min_value=0, step=1, value=0)
            name = st.text_input("Nome *")
            doc = st.text_input("Documento (CPF/CNPJ)")
            notes = st.text_area("Observações", height=80)
            ok = st.form_submit_button("Salvar")

        if ok:
            if not name.strip():
                st.error("Nome é obrigatório.")
            else:
                if sid and int(sid) > 0:
                    s = session.query(Supplier).get(int(sid))
                    if not s:
                        st.error("ID não encontrado.")
                    else:
                        s.name = name.strip()
                        s.document = (doc.strip() or None)
                        s.notes = (notes.strip() or None)
                        session.commit()
                        st.success("Fornecedor atualizado.")
                else:
                    s = Supplier(name=name.strip(), document=(doc.strip() or None), notes=(notes.strip() or None))
                    session.add(s)
                    session.commit()
                    st.success(f"Fornecedor criado (ID {s.id}).")

        df = pd.read_sql(text("SELECT id, name, document, notes FROM suppliers ORDER BY name"), session.bind)
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("🗑️ Excluir fornecedor"):
            did = st.number_input("ID para excluir", min_value=0, step=1, value=0, key="del_supplier_id")
            if st.button("Excluir", key="btn_del_supplier"):
                if did and int(did) > 0:
                    s = session.query(Supplier).get(int(did))
                    if not s:
                        st.error("ID não encontrado.")
                    else:
                        session.delete(s)
                        session.commit()
                        st.success("Fornecedor excluído.")
                        st.rerun()
    finally:
        session.close()
