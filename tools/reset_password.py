"""
BK_ERP - Reset de senha (usa a mesma hash PBKDF2 do app)

Uso:
  python tools/reset_password.py "email" "nova_senha"

Ele tenta obter DATABASE_URL nesta ordem:
  1) env DATABASE_URL
  2) .streamlit/secrets.toml -> [general].database_url
"""
from __future__ import annotations

import os, sys, base64, hashlib, hmac
from pathlib import Path
from sqlalchemy import create_engine, text

try:
    import tomllib  # py 3.11+
except Exception:  # pragma: no cover
    tomllib = None

def pbkdf2_hash_password(password: str, salt_b64: str | None = None, iterations: int = 180_000) -> str:
    if not salt_b64:
        salt = os.urandom(16)
        salt_b64 = base64.b64encode(salt).decode("utf-8")
    else:
        salt = base64.b64decode(salt_b64.encode("utf-8"))
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
        raise SystemExit('Uso: python tools/reset_password.py "email" "nova_senha"')

    email = sys.argv[1].strip().lower()
    pwd = sys.argv[2]
    db_url = get_db_url()
    engine = create_engine(db_url, pool_pre_ping=True)

    new_hash = pbkdf2_hash_password(pwd)

    with engine.begin() as conn:
        res = conn.execute(text("UPDATE users SET password_hash=:h WHERE lower(email)=:e"), {"h": new_hash, "e": email})
        if res.rowcount == 0:
            raise SystemExit(f"Usuário não encontrado: {email}")

    print("OK: senha atualizada para", email)

if __name__ == "__main__":
    main()
