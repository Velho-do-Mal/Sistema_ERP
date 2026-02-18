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

# Tema unificado do ERP (sem alterar fontes/ícones/relatórios)
try:
    from bk_erp_shared.theme import apply_theme  # type: ignore
except Exception:  # pragma: no cover
    apply_theme = None  # type: ignore

# Exportação XLSX (Streamlit Cloud precisa do openpyxl no requirements)
try:
    import openpyxl  # noqa: F401
    HAS_OPENPYXL = True
except Exception:  # pragma: no cover
    HAS_OPENPYXL = False

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except Exception:
        tomllib = None  # type: ignore


from sqlalchemy import (
    create_engine, Column, Integer, String, Date, DateTime, Float,
    LargeBinary, Text, ForeignKey, Boolean, Index, text, func
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


# (CSS interno removido: usamos o tema unificado do ERP)

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
    st.plotly_chart(fig, width="stretch")


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
    """Gera link de download XLSX (precisa openpyxl).
    Se openpyxl não estiver instalado, retorna aviso e não quebra a tela.
    """
    if not HAS_OPENPYXL:
        return "<span style='color:#b45309'>⚠️ Exportação Excel indisponível: instale <b>openpyxl</b> no requirements.</span>"

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
    return (
        f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" '
        f'download="{filename}">📥 Baixar Excel</a>'
    )


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
                    st.dataframe(df_r.drop(columns=["Tipo"]), width="stretch", height=220)
            with c_up2:
                st.markdown("**A Pagar (15 dias)**")
                df_p = df_u[df_u["Tipo"]=="A Pagar"].copy()
                st.metric("Total", format_currency(df_p["Valor"].sum() if not df_p.empty else 0.0))
                if not df_p.empty:
                    df_p["Valor"] = df_p["Valor"].map(format_currency)
                    st.dataframe(df_p.drop(columns=["Tipo"]), width="stretch", height=220)

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
            st.dataframe(df_real_m, width="stretch", height=220)

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
            st.dataframe(df, width="stretch")

        st.markdown("### Saldos por conta (realizado)")
        df_bal = compute_account_balances(session)
        if df_bal.empty:
            st.info("Cadastre ao menos uma conta para visualizar saldos.")
        else:
            st.dataframe(df_bal, width="stretch")

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
            st.dataframe(df, width="stretch")

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
    """Categorias com: filtro por tipo, subcategorias vinculadas, tabela editável."""
    session = SessionLocal()
    try:
        # ── Carregar todas as categorias ──
        all_cats = session.query(Category).order_by(
            Category.parent_id.asc().nullsfirst(), Category.name.asc()
        ).all()
        cat_map = {c.id: c for c in all_cats}

        # ── Abas: Categorias Pai | Subcategorias ──
        tab_cat, tab_sub = st.tabs(["📂 Categorias", "🗂️ Subcategorias"])

        # ══════════════════════════════════════
        # ABA 1: CATEGORIAS PAI
        # ══════════════════════════════════════
        with tab_cat:
            st.markdown('<div class="bk-section">➕ Nova Categoria</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([2, 1, 3])
            with c1:
                new_cat_name = st.text_input("Nome *", key="new_cat_name")
            with c2:
                new_mov_type = st.selectbox(
                    "Tipo de movimento",
                    options=["both", "entrada", "saida"],
                    format_func=lambda x: {"both": "Ambos (entrada/saída)", "entrada": "Entrada", "saida": "Saída"}[x],
                    key="new_cat_type"
                )
            with c3:
                new_cat_notes = st.text_input("Observação (opcional)", key="new_cat_notes")

            if st.button("💾 Criar Categoria", key="btn_create_cat", type="primary"):
                if not new_cat_name.strip():
                    st.error("Nome é obrigatório.")
                else:
                    c = Category(
                        name=new_cat_name.strip(),
                        mov_type=new_mov_type,
                        parent_id=None,
                        notes=new_cat_notes.strip() or None
                    )
                    session.add(c)
                    session.flush()
                    audit(session, "CREATE", "categories", c.id, None, {"name": c.name})
                    session.commit()
                    st.success(f"Categoria '{c.name}' criada.")
                    st.rerun()

            st.markdown("---")

            # Filtro por tipo
            filtro_tipo = st.radio(
                "Filtrar por tipo:",
                options=["Todos", "Entrada", "Saída"],
                horizontal=True,
                key="cat_filtro_tipo"
            )

            # Pais (sem parent_id)
            pais = [c for c in all_cats if c.parent_id is None]
            if filtro_tipo == "Entrada":
                pais = [c for c in pais if c.mov_type in ("entrada", "both")]
            elif filtro_tipo == "Saída":
                pais = [c for c in pais if c.mov_type in ("saida", "both")]

            if not pais:
                st.info("Nenhuma categoria encontrada para o filtro selecionado.")
            else:
                tipo_label = {"both": "Ambos", "entrada": "Entrada", "saida": "Saída"}
                df_cats = pd.DataFrame([{
                    "id": c.id,
                    "Nome": c.name,
                    "Tipo": tipo_label.get(c.mov_type, c.mov_type),
                    "Obs": (c.notes or "")[:80],
                } for c in pais])

                edited = st.data_editor(
                    df_cats,
                    use_container_width=True,
                    hide_index=True,
                    num_rows="fixed",
                    key="editor_cats",
                    column_config={
                        "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                        "Nome": st.column_config.TextColumn("Nome", width="medium"),
                        "Tipo": st.column_config.SelectboxColumn(
                            "Tipo",
                            options=["Ambos", "Entrada", "Saída"],
                            width="small"
                        ),
                        "Obs": st.column_config.TextColumn("Observação", width="large"),
                    }
                )

                if st.button("💾 Salvar edições de Categorias", key="btn_save_cats"):
                    tipo_rev = {"Ambos": "both", "Entrada": "entrada", "Saída": "saida"}
                    changed = 0
                    for _, row in edited.iterrows():
                        c = cat_map.get(int(row["id"]))
                        if c:
                            before = {"name": c.name, "mov_type": c.mov_type, "notes": c.notes}
                            c.name = str(row["Nome"]).strip()
                            c.mov_type = tipo_rev.get(str(row["Tipo"]), c.mov_type)
                            c.notes = str(row["Obs"]).strip() or None
                            audit(session, "UPDATE", "categories", c.id, before, {"name": c.name})
                            changed += 1
                    session.commit()
                    st.success(f"{changed} categoria(s) salva(s).")
                    st.rerun()

                st.markdown("---")
                st.markdown("**🗑️ Excluir categoria:**")
                del_opts = {"— selecione —": None} | {f"{c.id} — {c.name}": c.id for c in pais}
                del_sel = st.selectbox("Selecione para excluir", list(del_opts.keys()), key="cat_del_sel")
                if st.button("Excluir categoria selecionada", type="secondary", key="btn_del_cat"):
                    del_id = del_opts.get(del_sel)
                    if del_id:
                        c = session.query(Category).get(int(del_id))
                        if c:
                            if session.query(Category).filter(Category.parent_id == c.id).first():
                                st.error("Não é possível excluir: esta categoria tem subcategorias. Exclua-as primeiro.")
                            else:
                                before = {"name": c.name}
                                session.delete(c)
                                audit(session, "DELETE", "categories", del_id, before, None)
                                session.commit()
                                st.success("Categoria excluída.")
                                st.rerun()

        # ══════════════════════════════════════
        # ABA 2: SUBCATEGORIAS
        # ══════════════════════════════════════
        with tab_sub:
            st.markdown('<div class="bk-section">➕ Nova Subcategoria</div>', unsafe_allow_html=True)

            # Filtro de tipo para mostrar só as categorias-pai relevantes
            sc1, sc2 = st.columns([1, 2])
            with sc1:
                sub_tipo_filtro = st.selectbox(
                    "Tipo de movimento",
                    options=["Todos", "Entrada", "Saída"],
                    key="sub_tipo_filtro"
                )

            # Filtrar pais para o dropdown
            pais_all = [c for c in all_cats if c.parent_id is None]
            if sub_tipo_filtro == "Entrada":
                pais_filtrados = [c for c in pais_all if c.mov_type in ("entrada", "both")]
            elif sub_tipo_filtro == "Saída":
                pais_filtrados = [c for c in pais_all if c.mov_type in ("saida", "both")]
            else:
                pais_filtrados = pais_all

            pai_opts = {"— selecione a categoria pai —": None} | {
                f"{c.id} — {c.name}": c.id for c in pais_filtrados
            }

            with sc2:
                pai_sel_label = st.selectbox("Categoria Pai *", list(pai_opts.keys()), key="sub_pai_sel")

            s1, s2 = st.columns([2, 3])
            with s1:
                sub_name = st.text_input("Nome da subcategoria *", key="sub_name")
            with s2:
                sub_notes = st.text_input("Observação (opcional)", key="sub_notes")

            if st.button("💾 Criar Subcategoria", key="btn_create_sub", type="primary"):
                pai_id = pai_opts.get(pai_sel_label)
                if not sub_name.strip():
                    st.error("Nome é obrigatório.")
                elif pai_id is None:
                    st.error("Selecione a categoria pai.")
                else:
                    pai_obj = cat_map.get(pai_id)
                    sub = Category(
                        name=sub_name.strip(),
                        mov_type=pai_obj.mov_type if pai_obj else "both",
                        parent_id=pai_id,
                        notes=sub_notes.strip() or None
                    )
                    session.add(sub)
                    session.flush()
                    audit(session, "CREATE", "categories", sub.id, None, {"name": sub.name, "parent_id": pai_id})
                    session.commit()
                    st.success(f"Subcategoria '{sub.name}' criada em '{pai_obj.name if pai_obj else ''}'.")
                    st.rerun()

            st.markdown("---")

            # Lista de subcategorias filtrada pela categoria-pai selecionada
            st.markdown("**Visualizar / Editar subcategorias por categoria:**")
            pai_view_opts = {"— todas —": None} | {
                f"{c.id} — {c.name}": c.id for c in pais_all
            }
            pai_view_sel = st.selectbox("Filtrar por categoria:", list(pai_view_opts.keys()), key="sub_view_pai")
            pai_view_id = pai_view_opts.get(pai_view_sel)

            subs = [c for c in all_cats if c.parent_id is not None]
            if pai_view_id is not None:
                subs = [c for c in subs if c.parent_id == pai_view_id]

            if not subs:
                st.info("Nenhuma subcategoria encontrada.")
            else:
                df_subs = pd.DataFrame([{
                    "id": s.id,
                    "Nome": s.name,
                    "Categoria Pai": cat_map[s.parent_id].name if s.parent_id in cat_map else "",
                    "Obs": (s.notes or "")[:80],
                } for s in subs])

                edited_subs = st.data_editor(
                    df_subs,
                    use_container_width=True,
                    hide_index=True,
                    num_rows="fixed",
                    key="editor_subs",
                    column_config={
                        "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                        "Nome": st.column_config.TextColumn("Nome", width="medium"),
                        "Categoria Pai": st.column_config.TextColumn("Categoria Pai", disabled=True, width="medium"),
                        "Obs": st.column_config.TextColumn("Observação", width="large"),
                    }
                )

                if st.button("💾 Salvar edições de Subcategorias", key="btn_save_subs"):
                    sub_map = {s.id: s for s in subs}
                    changed = 0
                    for _, row in edited_subs.iterrows():
                        s = sub_map.get(int(row["id"]))
                        if s:
                            before = {"name": s.name, "notes": s.notes}
                            s.name = str(row["Nome"]).strip()
                            s.notes = str(row["Obs"]).strip() or None
                            audit(session, "UPDATE", "categories", s.id, before, {"name": s.name})
                            changed += 1
                    session.commit()
                    st.success(f"{changed} subcategoria(s) salva(s).")
                    st.rerun()

                st.markdown("---")
                st.markdown("**🗑️ Excluir subcategoria:**")
                sub_del_opts = {"— selecione —": None} | {f"{s.id} — {s.name}": s.id for s in subs}
                sub_del_sel = st.selectbox("Selecione para excluir", list(sub_del_opts.keys()), key="sub_del_sel")
                if st.button("Excluir subcategoria selecionada", type="secondary", key="btn_del_sub"):
                    sub_del_id = sub_del_opts.get(sub_del_sel)
                    if sub_del_id:
                        s = session.query(Category).get(int(sub_del_id))
                        if s:
                            before = {"name": s.name}
                            session.delete(s)
                            audit(session, "DELETE", "categories", sub_del_id, before, None)
                            session.commit()
                            st.success("Subcategoria excluída.")
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
            st.dataframe(df, width="stretch")

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
            st.dataframe(df, width="stretch")

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
    """Centros de Custo — formulário de criação + tabela editável + exclusão por seleção."""
    session = SessionLocal()
    try:
        # ── Criar novo ──
        st.markdown('<div class="bk-section">➕ Novo Centro de Custo</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 3])
        with c1:
            new_cc_name = st.text_input("Nome *", key="new_cc_name")
        with c2:
            new_cc_notes = st.text_input("Observação (opcional)", key="new_cc_notes")

        if st.button("💾 Criar Centro de Custo", key="btn_create_cc", type="primary"):
            if not new_cc_name.strip():
                st.error("Nome é obrigatório.")
            else:
                cc = CostCenter(name=new_cc_name.strip(), notes=new_cc_notes.strip() or None)
                session.add(cc)
                session.flush()
                audit(session, "CREATE", "cost_centers", cc.id, None, {"name": cc.name})
                session.commit()
                st.success(f"Centro de custo '{cc.name}' criado.")
                st.rerun()

        st.markdown("---")

        ccs = session.query(CostCenter).order_by(CostCenter.name.asc()).all()
        if not ccs:
            st.info("Nenhum centro de custo cadastrado ainda.")
        else:
            cc_map = {c.id: c for c in ccs}
            df_cc = pd.DataFrame([{
                "id": c.id,
                "Nome": c.name,
                "Observação": (c.notes or ""),
            } for c in ccs])

            edited = st.data_editor(
                df_cc,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="editor_cc",
                column_config={
                    "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                    "Nome": st.column_config.TextColumn("Nome", width="medium"),
                    "Observação": st.column_config.TextColumn("Observação", width="large"),
                }
            )

            if st.button("💾 Salvar edições", key="btn_save_cc"):
                changed = 0
                for _, row in edited.iterrows():
                    cc = cc_map.get(int(row["id"]))
                    if cc:
                        before = {"name": cc.name, "notes": cc.notes}
                        cc.name = str(row["Nome"]).strip()
                        cc.notes = str(row["Observação"]).strip() or None
                        audit(session, "UPDATE", "cost_centers", cc.id, before, {"name": cc.name})
                        changed += 1
                session.commit()
                st.success(f"{changed} registro(s) salvo(s).")
                st.rerun()

            st.markdown("---")
            st.markdown("**🗑️ Excluir centro de custo:**")
            del_opts = {"— selecione —": None} | {f"{c.id} — {c.name}": c.id for c in ccs}
            del_sel = st.selectbox("Selecione para excluir", list(del_opts.keys()), key="cc_del_sel")
            if st.button("Excluir selecionado", type="secondary", key="btn_del_cc"):
                del_id = del_opts.get(del_sel)
                if del_id:
                    cc = session.query(CostCenter).get(int(del_id))
                    if cc:
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
                st.dataframe(pd.DataFrame(rows), width="stretch")
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
            st.subheader("Orçamento (12 meses) — Previsto x Realizado")
            years = list(range(date.today().year - 1, date.today().year + 3))
            year = st.selectbox("Ano", years, index=1, key="budget_table_year")

            st.caption("Edite o **Previsto** por categoria (pai) e mês. O **Realizado** é calculado automaticamente a partir dos lançamentos marcados como realizados (pagos/recebidos).")

            MONTH_LABELS = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

            money_cols = {
                m: st.column_config.NumberColumn(m, format="R$ %.2f", step=100.0)
                for m in MONTH_LABELS
            }

            # Categorias pai
            parents = [c for c in cats if c.parent_id is None]
            by_id = {c.id: c for c in cats}

            def _root_cat_id(cat_id: int) -> int:
                cur = by_id.get(cat_id)
                while cur is not None and cur.parent_id is not None and by_id.get(cur.parent_id) is not None:
                    cur = by_id.get(cur.parent_id)
                return int(cur.id) if cur is not None else int(cat_id)

            # Carrega orçamentos previstos (year)
            buds = session.query(Budget).filter(Budget.year == int(year)).all()
            bud_map = {(int(b.category_id), int(b.month)): float(b.amount or 0.0) for b in buds}

            def _make_editor_df(cat_list):
                rows = []
                for c in cat_list:
                    row = {"ID": int(c.id), "Categoria": c.name}
                    for mi, mlabel in enumerate(MONTH_LABELS, start=1):
                        row[mlabel] = float(bud_map.get((int(c.id), mi), 0.0))
                    rows.append(row)
                return pd.DataFrame(rows)

            def _save_editor_df(df_edit: pd.DataFrame) -> int:
                changed = 0
                for _, r in df_edit.iterrows():
                    cid = int(r["ID"])
                    for mi, mlabel in enumerate(MONTH_LABELS, start=1):
                        val = r.get(mlabel, 0.0)
                        try:
                            amt = float(val or 0.0)
                        except Exception:
                            amt = 0.0

                        b = session.query(Budget).filter(
                            Budget.category_id == cid,
                            Budget.year == int(year),
                            Budget.month == int(mi),
                        ).first()

                        if amt <= 0:
                            if b is not None:
                                before = {"category_id": b.category_id, "year": b.year, "month": b.month, "amount": b.amount}
                                session.delete(b)
                                audit(session, "DELETE", "budgets", int(b.id), before, None)
                                changed += 1
                            continue

                        if b is None:
                            b = Budget(category_id=cid, year=int(year), month=int(mi), amount=amt)
                            session.add(b)
                            session.flush()
                            audit(session, "CREATE", "budgets", int(b.id), None, {"category_id": cid, "year": int(year), "month": int(mi), "amount": amt})
                            changed += 1
                        else:
                            if float(b.amount or 0.0) != float(amt):
                                before = {"amount": b.amount}
                                b.amount = amt
                                audit(session, "UPDATE", "budgets", int(b.id), before, {"amount": amt})
                                changed += 1
                if changed:
                    session.commit()
                return changed

            entradas = [c for c in parents if (c.mov_type or "both") == "entrada"]
            saidas = [c for c in parents if (c.mov_type or "both") == "saida"]

            st.markdown("### <span style='color:#2563eb'>Entradas (Previsto)</span>", unsafe_allow_html=True)
            df_in = _make_editor_df(entradas)
            df_in_edit = st.data_editor(
                df_in,
                hide_index=True,
                disabled=["ID", "Categoria"],
                column_config=money_cols,
                key="budget_in_editor",
                width="stretch",
            )

            st.markdown("### <span style='color:#dc2626'>Saídas (Previsto)</span>", unsafe_allow_html=True)
            df_out = _make_editor_df(saidas)
            df_out_edit = st.data_editor(
                df_out,
                hide_index=True,
                disabled=["ID", "Categoria"],
                column_config=money_cols,
                key="budget_out_editor",
                width="stretch",
            )

            if st.button("💾 Salvar orçamento previsto", key="btn_save_budget_table"):
                changed = _save_editor_df(df_in_edit) + _save_editor_df(df_out_edit)
                st.success(f"Orçamento salvo. Alterações aplicadas: {changed}.")
                st.rerun()

            # Realizado por mês (pagos/recebidos)
            txs = session.query(Transaction).filter(Transaction.paid == True).all()
            actual_in = {m: 0.0 for m in range(1, 13)}
            actual_out = {m: 0.0 for m in range(1, 13)}
            for t in txs:
                dref = t.paid_date or t.date
                if not dref or int(dref.year) != int(year):
                    continue
                if t.type == "entrada":
                    actual_in[int(dref.month)] += float(t.amount or 0.0)
                elif t.type == "saida":
                    actual_out[int(dref.month)] += float(t.amount or 0.0)

            # Previsto por mês (somatório da tabela)
            planned_in = {m: 0.0 for m in range(1, 13)}
            planned_out = {m: 0.0 for m in range(1, 13)}
            for mi, mlabel in enumerate(MONTH_LABELS, start=1):
                planned_in[mi] = float(df_in_edit[mlabel].sum()) if not df_in_edit.empty else 0.0
                planned_out[mi] = float(df_out_edit[mlabel].sum()) if not df_out_edit.empty else 0.0

            planned_net = [planned_in[m] - planned_out[m] for m in range(1, 13)]
            actual_net = [actual_in[m] - actual_out[m] for m in range(1, 13)]

            tot_plan_in = sum(planned_in.values())
            tot_plan_out = sum(planned_out.values())
            tot_act_in = sum(actual_in.values())
            tot_act_out = sum(actual_out.values())

            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Entradas (Previsto)", format_currency(tot_plan_in))
                st.caption(f"Realizado: {format_currency(tot_act_in)}")
            with k2:
                st.metric("Saídas (Previsto)", format_currency(tot_plan_out))
                st.caption(f"Realizado: {format_currency(tot_act_out)}")
            with k3:
                st.metric("Saldo (Previsto)", format_currency(tot_plan_in - tot_plan_out))
                st.caption(f"Realizado: {format_currency(tot_act_in - tot_act_out)}")

            # Gráfico: colunas (saldo mensal) + linhas (saldo acumulado)
            fig = go.Figure()
            fig.add_bar(x=MONTH_LABELS, y=planned_net, name="Previsto (saldo mensal)")
            fig.add_bar(x=MONTH_LABELS, y=actual_net, name="Realizado (saldo mensal)")
            fig.add_scatter(x=MONTH_LABELS, y=list(pd.Series(planned_net).cumsum()), name="Saldo acumulado previsto", mode="lines+markers", yaxis="y2")
            fig.add_scatter(x=MONTH_LABELS, y=list(pd.Series(actual_net).cumsum()), name="Saldo acumulado realizado", mode="lines+markers", yaxis="y2")
            fig.update_layout(
                barmode="group",
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis_title="Saldo mensal (R$)",
                yaxis2=dict(title="Saldo acumulado (R$)", overlaying="y", side="right"),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, width="stretch")

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

        st.markdown('<div class="bk-section">➕ Novo Lançamento</div>', unsafe_allow_html=True)

        # ── Seleção de categoria e subcategoria FORA do form (reativas) ──
        all_categories = session.query(Category).order_by(
            Category.parent_id.asc().nullsfirst(), Category.name.asc()
        ).all()

        # Tipo de movimento — controla filtro de categorias
        tx_type_pre = st.selectbox(
            "Tipo de movimento",
            ["entrada", "saida", "transferencia"],
            index=1,
            key="tx_type_pre",
            format_func=lambda x: {"entrada": "🟢 Entrada", "saida": "🔴 Saída", "transferencia": "🔄 Transferência"}[x]
        )

        # Categorias pai filtradas pelo tipo
        cats_pai = [c for c in all_categories if c.parent_id is None]
        if tx_type_pre == "entrada":
            cats_pai = [c for c in cats_pai if c.mov_type in ("entrada", "both")]
        elif tx_type_pre == "saida":
            cats_pai = [c for c in cats_pai if c.mov_type in ("saida", "both")]

        cat_pai_opts = {"(Sem categoria)": None} | {f"{c.name}": c.id for c in cats_pai}

        tc1, tc2 = st.columns(2)
        with tc1:
            cat_pai_sel = st.selectbox("Categoria", list(cat_pai_opts.keys()), key="tx_cat_pai")
        cat_pai_id = cat_pai_opts.get(cat_pai_sel)

        # Subcategorias filtradas pela categoria pai selecionada
        subs_disponiveis = [
            c for c in all_categories
            if c.parent_id is not None and (cat_pai_id is None or c.parent_id == cat_pai_id)
        ]
        sub_opts = {"(Sem subcategoria)": None} | {f"{s.name}": s.id for s in subs_disponiveis}
        with tc2:
            sub_sel = st.selectbox("Subcategoria", list(sub_opts.keys()), key="tx_subcat",
                                   disabled=(len(subs_disponiveis) == 0))

        # Anexos FORA do form (st.file_uploader não funciona bem dentro de st.form)
        attachments = st.file_uploader(
            "📎 Anexos (boletos, comprovantes, contratos, NFs)",
            accept_multiple_files=True,
            key="tx_new_attachments",
            help="Aceita PDF, imagens e documentos. Você pode selecionar múltiplos arquivos."
        )

        with st.form("tx_form"):
            r1c1, r1c2, r1c3 = st.columns([2, 2, 1])
            with r1c1:
                tx_date = st.date_input("Data competência", value=date.today())
            with r1c2:
                due_date = st.date_input("Vencimento", value=date.today())
            with r1c3:
                use_due = st.checkbox("Usar vencimento", value=True)

            r2c1, r2c2 = st.columns([1, 1])
            with r2c1:
                paid = st.checkbox("Já está realizado?", value=False)
            with r2c2:
                # Sempre renderizado — evita bug de estado no Streamlit dentro de st.form
                paid_date = st.date_input("Data realização (se pago)", value=date.today())

            description = st.text_input("Descrição *")
            amount = st.number_input("Valor (R$) *", value=0.0, step=100.0, format="%.2f")

            # ── Contas: SEMPRE os três campos renderizados ──
            # (evita NameError quando tx_type muda entre renders)
            fa1, fa2, fa3 = st.columns(3)
            with fa1:
                account_sel = st.selectbox("Conta (entrada/saída)", acc_opts,
                                           index=1 if len(acc_opts) > 1 else 0)
            with fa2:
                from_acc = st.selectbox("De (só transferência)", acc_opts,
                                        index=1 if len(acc_opts) > 1 else 0, key="from_acc")
            with fa3:
                to_acc = st.selectbox("Para (só transferência)", acc_opts,
                                      index=min(2, len(acc_opts) - 1), key="to_acc")

            cost_center_sel = st.selectbox("Centro de Custo", cc_opts, index=0)

            # ── Cliente e Fornecedor: SEMPRE renderizados ──
            cf1, cf2 = st.columns(2)
            with cf1:
                client_sel  = st.selectbox("Cliente", cli_opts, index=0)
            with cf2:
                supplier_sel = st.selectbox("Fornecedor", sup_opts, index=0)

            reference = st.text_input("Referência (NF, boleto, contrato, PO)")
            notes = st.text_area("Observações", height=80)

            st.markdown("**📅 Recorrência (opcional)**")
            rec_enable = st.checkbox("Criar recorrência", value=False)
            rec_count = st.number_input("Quantidade de parcelas", min_value=1, value=1, step=1)
            rec_period = st.selectbox(
                "Periodicidade", ["monthly", "weekly", "biweekly", "yearly"], index=0,
                format_func=lambda x: {
                    "monthly": "Mensal", "weekly": "Semanal",
                    "biweekly": "Quinzenal", "yearly": "Anual"
                }[x]
            )
            ok = st.form_submit_button("💾 Salvar lançamento", type="primary", use_container_width=True)

        # tx_type vem do seletor externo (reativo, fora do form)
        tx_type = tx_type_pre

        if ok:
            if not description.strip():
                st.error("Descrição é obrigatória.")
            elif amount <= 0:
                st.error("Valor deve ser maior que 0.")
            else:
                due = due_date if use_due else None
                pd_dt = paid_date if paid else None  # paid_date widget sempre renderizado; só usa se paid=True

                # Anexos (opcional): boletos, comprovantes, recibos, etc.
                uploaded_payloads = []
                for up in (attachments or []):
                    try:
                        data = up.getvalue() if hasattr(up, "getvalue") else up.read()
                    except Exception:
                        data = b""
                    if not data:
                        continue
                    uploaded_payloads.append((up.name, getattr(up, "type", None), data))

                def _save_attachments_for(tx_id: int):
                    for fname, ctype, data in uploaded_payloads:
                        att = Attachment(
                            transaction_id=int(tx_id),
                            filename=str(fname),
                            content_type=ctype,
                            data=data,
                        )
                        session.add(att)
                        audit(session, "CREATE", "attachments", None, None, {"tx_id": int(tx_id), "filename": str(fname)})

                # Categoria: subcategoria tem prioridade sobre a pai
                _sub_id = sub_opts.get(sub_sel) if sub_sel else None
                cat_id = _sub_id if _sub_id is not None else cat_pai_id
                cc_id = (int(cost_center_sel.split(" - ", 1)[0])
                         if cost_center_sel != "(Sem centro de custo)" else None)
                # Cliente só para entrada; Fornecedor só para saída
                cli_id = (int(client_sel.split(" - ", 1)[0])
                          if tx_type == "entrada" and client_sel != "(Sem cliente)" else None)
                sup_id = (int(supplier_sel.split(" - ", 1)[0])
                          if tx_type == "saida" and supplier_sel != "(Sem fornecedor)" else None)

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
                            # Anexa os mesmos arquivos aos dois registros da transferência
                            if uploaded_payloads:
                                _save_attachments_for(int(tx_out_id))
                                _save_attachments_for(int(tx_in_id))

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

        f1, f2, f3, f4, f5 = st.columns([2, 2, 1, 1, 2])
        with f1:
            start = st.date_input("Data inicial", value=date.today().replace(day=1), key="tx_list_start")
        with f2:
            end = st.date_input("Data final", value=date.today(), key="tx_list_end")
        with f3:
            base = st.selectbox("Base", ["Tudo", "Realizado", "Previsto"], index=0, key="tx_list_base")
        with f4:
            tipo_filtro = st.selectbox("Tipo", ["Todos", "entrada", "saida", "transferencia"], index=0, key="tx_list_tipo")
        with f5:
            q = st.text_input("Busca (descrição/ref/qualquer)", value="", key="tx_list_q")

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
            if tipo_filtro != "Todos" and t.type != tipo_filtro:
                continue
            if q.strip():
                qq = q.strip().lower()
                # Busca em descrição, referência, conta, categoria, cliente, fornecedor
                campos = " ".join([
                    (t.description or ""),
                    (t.reference or ""),
                    (t.account.name if t.account else ""),
                    (t.category.name if t.category else ""),
                    (t.client.name if t.client else ""),
                    (t.supplier.name if t.supplier else ""),
                ]).lower()
                if qq not in campos:
                    continue
            filtered.append(t)

        st.caption(f"Mostrando {len(filtered)} registros (limite interno 3000).")

        # Contagem de anexos por movimentação (para exibir na tabela)
        att_count = {}
        if filtered:
            ids = [int(t.id) for t in filtered if getattr(t, "id", None) is not None]
            if ids:
                try:
                    q_att = session.query(Attachment.transaction_id, func.count(Attachment.id)).filter(Attachment.transaction_id.in_(ids)).group_by(Attachment.transaction_id).all()
                    att_count = {int(tid): int(cnt) for tid, cnt in q_att}
                except Exception:
                    att_count = {}

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
                "Anexos": int(att_count.get(int(t.id), 0)),
                "Transfer?": "Sim" if t.is_transfer else "Não",
                "Grupo Recorrência": t.recurrence_group or "",
            })

        df_ext = pd.DataFrame(rows)
        if df_ext.empty:
            st.info("Sem lançamentos no filtro.")
        else:
            df_show = df_ext.copy()
            df_show["Valor"] = df_show["Valor"].map(format_currency)
            st.dataframe(df_show, width="stretch")

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
                        e_paid_date = None
                        if e_paid:
                            e_paid_date = st.date_input("Pagamento/Recebimento", value=t.paid_date or date.today())
                        else:
                            st.caption("Pagamento/Recebimento: será preenchido quando marcar como realizado.")
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
                st.dataframe(show, width="stretch")

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
                st.dataframe(df_bal, width="stretch")

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
                st.dataframe(df_cat.rename(columns={"Item": "Categoria"}), width="stretch")

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
                st.dataframe(df_cc2.rename(columns={"Item": "Centro de Custo"}), width="stretch")

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
                st.dataframe(df_show, width="stretch")

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
    # Aplica o tema unificado do ERP (mantém padrão visual entre páginas)
    if apply_theme:
        apply_theme()

    st.title("💰 Financeiro")
    st.caption("Lançamentos, dashboards, relatórios, metas e orçamento")

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
    """UI de cadastro/listagem de Clientes — formulário + tabela editável."""
    session = SessionLocal()
    try:
        # ── Criar novo ──
        st.markdown('<div class="bk-section">➕ Novo Cliente</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            new_cli_name = st.text_input("Nome *", key="new_cli_name")
        with col2:
            new_cli_doc = st.text_input("CPF/CNPJ", key="new_cli_doc")
        with col3:
            new_cli_notes = st.text_input("Observação", key="new_cli_notes")

        if st.button("💾 Criar Cliente", key="btn_create_cli", type="primary"):
            if not new_cli_name.strip():
                st.error("Nome é obrigatório.")
            else:
                c = Client(
                    name=new_cli_name.strip(),
                    document=new_cli_doc.strip() or None,
                    notes=new_cli_notes.strip() or None
                )
                session.add(c)
                session.flush()
                audit(session, "CREATE", "clients", c.id, None, {"name": c.name})
                session.commit()
                st.success(f"Cliente '{c.name}' criado.")
                st.rerun()

        st.markdown("---")

        clients = session.query(Client).order_by(Client.name.asc()).all()
        if not clients:
            st.info("Nenhum cliente cadastrado ainda.")
        else:
            cli_map = {c.id: c for c in clients}
            df_cli = pd.DataFrame([{
                "id": c.id,
                "Nome": c.name,
                "CPF/CNPJ": c.document or "",
                "Observação": (c.notes or ""),
            } for c in clients])

            edited = st.data_editor(
                df_cli,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="editor_clients",
                column_config={
                    "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                    "Nome": st.column_config.TextColumn("Nome", width="medium"),
                    "CPF/CNPJ": st.column_config.TextColumn("CPF/CNPJ", width="medium"),
                    "Observação": st.column_config.TextColumn("Observação", width="large"),
                }
            )

            if st.button("💾 Salvar edições", key="btn_save_cli"):
                changed = 0
                for _, row in edited.iterrows():
                    c = cli_map.get(int(row["id"]))
                    if c:
                        before = {"name": c.name, "document": c.document}
                        c.name = str(row["Nome"]).strip()
                        c.document = str(row["CPF/CNPJ"]).strip() or None
                        c.notes = str(row["Observação"]).strip() or None
                        audit(session, "UPDATE", "clients", c.id, before, {"name": c.name})
                        changed += 1
                session.commit()
                st.success(f"{changed} cliente(s) salvo(s).")
                st.rerun()

            st.markdown("---")
            st.markdown("**🗑️ Excluir cliente:**")
            del_opts = {"— selecione —": None} | {f"{c.id} — {c.name}": c.id for c in clients}
            del_sel = st.selectbox("Selecione para excluir", list(del_opts.keys()), key="cli_del_sel")
            if st.button("Excluir selecionado", type="secondary", key="btn_del_cli"):
                del_id = del_opts.get(del_sel)
                if del_id:
                    c = session.query(Client).get(int(del_id))
                    if c:
                        before = {"name": c.name}
                        session.delete(c)
                        audit(session, "DELETE", "clients", del_id, before, None)
                        session.commit()
                        st.success("Cliente excluído.")
                        st.rerun()
    except Exception as e:
        session.rollback()
        st.error(f"Erro em Clientes: {e}")
    finally:
        session.close()


def suppliers_ui(SessionLocal):
    """UI de cadastro/listagem de Fornecedores — formulário + tabela editável."""
    session = SessionLocal()
    try:
        st.markdown('<div class="bk-section">➕ Novo Fornecedor / Prestador</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            new_sup_name = st.text_input("Nome *", key="new_sup_name")
        with col2:
            new_sup_doc = st.text_input("CPF/CNPJ", key="new_sup_doc")
        with col3:
            new_sup_notes = st.text_input("Observação", key="new_sup_notes")

        if st.button("💾 Criar Fornecedor", key="btn_create_sup", type="primary"):
            if not new_sup_name.strip():
                st.error("Nome é obrigatório.")
            else:
                s = Supplier(
                    name=new_sup_name.strip(),
                    document=new_sup_doc.strip() or None,
                    notes=new_sup_notes.strip() or None
                )
                session.add(s)
                session.flush()
                audit(session, "CREATE", "suppliers", s.id, None, {"name": s.name})
                session.commit()
                st.success(f"Fornecedor '{s.name}' criado.")
                st.rerun()

        st.markdown("---")

        suppliers = session.query(Supplier).order_by(Supplier.name.asc()).all()
        if not suppliers:
            st.info("Nenhum fornecedor cadastrado ainda.")
        else:
            sup_map = {s.id: s for s in suppliers}
            df_sup = pd.DataFrame([{
                "id": s.id,
                "Nome": s.name,
                "CPF/CNPJ": s.document or "",
                "Observação": (s.notes or ""),
            } for s in suppliers])

            edited = st.data_editor(
                df_sup,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                key="editor_suppliers",
                column_config={
                    "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                    "Nome": st.column_config.TextColumn("Nome", width="medium"),
                    "CPF/CNPJ": st.column_config.TextColumn("CPF/CNPJ", width="medium"),
                    "Observação": st.column_config.TextColumn("Observação", width="large"),
                }
            )

            if st.button("💾 Salvar edições", key="btn_save_sup"):
                changed = 0
                for _, row in edited.iterrows():
                    s = sup_map.get(int(row["id"]))
                    if s:
                        before = {"name": s.name, "document": s.document}
                        s.name = str(row["Nome"]).strip()
                        s.document = str(row["CPF/CNPJ"]).strip() or None
                        s.notes = str(row["Observação"]).strip() or None
                        audit(session, "UPDATE", "suppliers", s.id, before, {"name": s.name})
                        changed += 1
                session.commit()
                st.success(f"{changed} fornecedor(es) salvo(s).")
                st.rerun()

            st.markdown("---")
            st.markdown("**🗑️ Excluir fornecedor:**")
            del_opts = {"— selecione —": None} | {f"{s.id} — {s.name}": s.id for s in suppliers}
            del_sel = st.selectbox("Selecione para excluir", list(del_opts.keys()), key="sup_del_sel")
            if st.button("Excluir selecionado", type="secondary", key="btn_del_sup"):
                del_id = del_opts.get(del_sel)
                if del_id:
                    s = session.query(Supplier).get(int(del_id))
                    if s:
                        before = {"name": s.name}
                        session.delete(s)
                        audit(session, "DELETE", "suppliers", del_id, before, None)
                        session.commit()
                        st.success("Fornecedor excluído.")
                        st.rerun()
    except Exception as e:
        session.rollback()
        st.error(f"Erro em Fornecedores: {e}")
    finally:
        session.close()
