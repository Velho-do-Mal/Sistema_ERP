
"""
BK_ERP - Notificador (E-mail e WhatsApp)

Uso:
    python notifier.py

Recomendação para produção:
- Rodar em um cron (ex.: a cada 1h) ou serviço (Cloud Run / container).
- O app Streamlit não é ideal para jobs em background.

Variáveis de ambiente:
- DATABASE_URL (Neon/Postgres)  [obrigatório]
- SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM
- NOTIFY_TO (lista separada por vírgula)
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM, WHATSAPP_TO

WhatsApp é opcional; se faltar credencial, ele apenas ignora.
"""
from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from datetime import date, timedelta, datetime

from sqlalchemy import text
from sqlalchemy import create_engine


def get_engine():
    db_url = (os.getenv("DATABASE_URL") or "").strip()
    if not db_url:
        raise RuntimeError("Defina DATABASE_URL.")
    return create_engine(db_url, pool_pre_ping=True)


def send_email(subject: str, html: str) -> None:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT") or "587")
    user = os.getenv("SMTP_USER")
    passwd = os.getenv("SMTP_PASS")
    from_addr = os.getenv("SMTP_FROM") or user
    to_list = [x.strip() for x in (os.getenv("NOTIFY_TO") or "").split(",") if x.strip()]

    if not host or not from_addr or not to_list:
        return

    msg = MIMEText(html, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_list)

    with smtplib.SMTP(host, port) as s:
        s.starttls()
        if user and passwd:
            s.login(user, passwd)
        s.sendmail(from_addr, to_list, msg.as_string())


def send_whatsapp(text_msg: str) -> None:
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    w_from = os.getenv("TWILIO_WHATSAPP_FROM")  # ex: whatsapp:+14155238886
    w_to = os.getenv("WHATSAPP_TO")  # ex: whatsapp:+55...

    if not sid or not token or not w_from or not w_to:
        return

    # Dependência opcional (não obrigatória no Streamlit)
    try:
        from twilio.rest import Client
    except Exception:
        return

    client = Client(sid, token)
    client.messages.create(from_=w_from, to=w_to, body=text_msg)


def already_sent(conn, entity: str, entity_id: str, event_type: str, channel: str) -> bool:
    r = conn.execute(
        text("""
            SELECT id FROM notification_events
            WHERE entity=:entity AND entity_id=:entity_id AND event_type=:event_type AND channel=:channel
            LIMIT 1
        """),
        dict(entity=entity, entity_id=entity_id, event_type=event_type, channel=channel),
    ).fetchone()
    return bool(r)


def mark_sent(conn, entity: str, entity_id: str, event_type: str, channel: str, status: str, info: str = "") -> None:
    conn.execute(
        text("""
            INSERT INTO notification_events (entity, entity_id, event_type, last_sent_at, channel, status, info)
            VALUES (:entity,:entity_id,:event_type,:last_sent_at,:channel,:status,:info)
        """),
        dict(entity=entity, entity_id=entity_id, event_type=event_type,
             last_sent_at=datetime.utcnow(), channel=channel, status=status, info=info),
    )


def run():
    engine = get_engine()
    today = date.today()
    win_end = today + timedelta(days=15)

    with engine.begin() as conn:
        # garantir tabela
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS notification_events (
                id SERIAL PRIMARY KEY,
                entity TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                last_sent_at TIMESTAMP,
                channel TEXT,
                status TEXT,
                info TEXT
            );
        """))

        # transações a vencer e vencidas
        txs = conn.execute(text("""
            SELECT id, type, amount, description, due_date, paid, paid_date
            FROM transactions
            WHERE due_date IS NOT NULL
        """)).fetchall()

        due_soon = []
        overdue = []
        paid_new = []

        for t in txs:
            tx_id = str(t.id)
            if not t.paid and t.due_date:
                if today <= t.due_date <= win_end:
                    due_soon.append(t)
                elif t.due_date < today:
                    overdue.append(t)
            if t.paid:
                # evento "paid" (para saída) e "received" (para entrada)
                ev = "received" if t.type == "entrada" else "paid"
                # manda uma vez
                if not already_sent(conn, "transaction", tx_id, ev, "email"):
                    paid_new.append((t, ev))

        # E-mail: vencendo
        if due_soon:
            lines = "".join([f"<li>#{t.id} • {t.description} • {t.due_date} • R$ {t.amount:,.2f}</li>" for t in due_soon[:50]])
            html = f"<h3>BK_ERP - Itens a vencer (15 dias)</h3><ul>{lines}</ul>"
            send_email("BK_ERP • A vencer (15 dias)", html)
            for t in due_soon:
                if not already_sent(conn, "transaction", str(t.id), "due_soon", "email"):
                    mark_sent(conn, "transaction", str(t.id), "due_soon", "email", "sent")

        # E-mail: atrasados
        if overdue:
            lines = "".join([f"<li>#{t.id} • {t.description} • {t.due_date} • R$ {t.amount:,.2f}</li>" for t in overdue[:50]])
            html = f"<h3>BK_ERP - Itens atrasados</h3><ul>{lines}</ul>"
            send_email("BK_ERP • Atrasados", html)
            for t in overdue:
                if not already_sent(conn, "transaction", str(t.id), "overdue", "email"):
                    mark_sent(conn, "transaction", str(t.id), "overdue", "email", "sent")

        # E-mail: pagos/recebidos
        for t, ev in paid_new:
            html = f"<h3>BK_ERP - Evento financeiro</h3><p>Transação #{t.id} ({'Recebido' if ev=='received' else 'Pago'}): <b>{t.description}</b><br/>Valor: R$ {t.amount:,.2f}</p>"
            send_email(f"BK_ERP • {'Recebido' if ev=='received' else 'Pago'} • #{t.id}", html)
            mark_sent(conn, "transaction", str(t.id), ev, "email", "sent")

        # WhatsApp (resumo)
        # Envia apenas resumo do dia (evita spam); marca como "daily_summary"
        key = today.strftime("%Y-%m-%d")
        if not already_sent(conn, "system", key, "daily_summary", "whatsapp"):
            msg = f"BK_ERP • {key}\nA vencer (15d): {len(due_soon)} | Atrasados: {len(overdue)} | Pagos/Recebidos novos: {len(paid_new)}"
            send_whatsapp(msg)
            mark_sent(conn, "system", key, "daily_summary", "whatsapp", "sent")


if __name__ == "__main__":
    run()
