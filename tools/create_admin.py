"""
BK_ERP - Criar admin inicial diretamente no banco (Neon)

Uso:
  python tools/create_admin.py "email" "senha"

Cria se não existir.
"""
from __future__ import annotations

import os, sys, base64, hashlib
from pathlib import Path
from sqlalchemy import create_engine, text

try:
    import tomllib
except Exception:
    tomllib = None

def pbkdf2_hash_password(password: str, salt_b64: str | None = None, iterations: int = 180_000) -> str:
    if not salt_b64:
        salt = os.urandom(16)
        salt_b64 = base64.b64encode(salt).decode("utf-8")
    else:
        import base64 as _b
        salt = _b.b64decode(salt_b64.encode("utf-8"))
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    dk_b64 = base64.b64encode(dk).decode("utf-8")
    return f"pbkdf2_sha256${iterations}${salt_b64}${dk_b64}"

def get_db_url() -> str:
    db = os.getenv("DATABASE_URL", "").strip()
    if db:
        return db
    p = Path(".streamlit") / "secrets.toml"
    if p.exists() and tomllib is not None:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
        return str(data.get("general", {}).get("database_url", "")).strip()
    raise SystemExit("Não achei DATABASE_URL nem .streamlit/secrets.toml")

def main():
    if len(sys.argv) < 3:
        raise SystemExit('Uso: python tools/create_admin.py "email" "senha"')

    email = sys.argv[1].strip().lower()
    pwd = sys.argv[2]

    engine = create_engine(get_db_url(), pool_pre_ping=True)
    ph = pbkdf2_hash_password(pwd)

    with engine.begin() as conn:
        exists = conn.execute(text("SELECT id FROM users WHERE lower(email)=:e"), {"e": email}).fetchone()
        if exists:
            print("Já existe usuário:", email)
            return
        conn.execute(text("""
            INSERT INTO users (name, email, role, password_hash, active)
            VALUES (:name, :email, 'admin', :ph, true)
        """), {"name": "Administrador", "email": email, "ph": ph})
    print("OK: admin criado:", email)

if __name__ == "__main__":
    main()
