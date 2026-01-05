
import os
import streamlit as st
from datetime import date
from sqlalchemy import text

from bk_erp_shared.theme import apply_theme
from bk_erp_shared.erp_db import get_finance_db, ensure_erp_tables
from bk_erp_shared.auth import login_and_guard, can_view

import bk_finance

st.set_page_config(page_title="BK_ERP - Admin & Notificações", layout="wide")
apply_theme()
ensure_erp_tables()

engine, SessionLocal = get_finance_db()
login_and_guard(SessionLocal)

st.markdown('<div class="bk-card"><div class="bk-title">Admin & Notificações</div><div class="bk-subtitle">Configurações de envio (e-mail/WhatsApp) e auditoria.</div></div>', unsafe_allow_html=True)

tabs = st.tabs(["🔔 Notificações", "👥 Usuários", "🧾 Auditoria"])

with tabs[0]:
    st.subheader("🔔 Notificações")
    st.write("O BK_ERP envia e-mails quando:")
    st.write("- Um título vai vencer (próximos 15 dias)")
    st.write("- Um título está atrasado")
    st.write("- Um título foi pago/recebido (detecção por varredura)")

    st.info("Para produção, rode `python notifier.py` via cron/Cloud Run. No Streamlit, use o botão abaixo apenas para teste manual.")

    c1, c2 = st.columns(2)
    with c1:
        st.code("SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM, NOTIFY_TO", language="bash")
        st.caption("NOTIFY_TO: lista separada por vírgula.")
    with c2:
        st.code("TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM, WHATSAPP_TO", language="bash")
        st.caption("WhatsApp via Twilio (opcional).")

    if st.button("Executar varredura agora (teste)"):
        # Rodar uma varredura simplificada (somente cria logs)
        today = date.today()
        with engine.begin() as conn:
            due = conn.execute(text("""
                SELECT COUNT(*) FROM transactions
                WHERE paid = FALSE AND due_date IS NOT NULL AND due_date BETWEEN :d1 AND (:d1 + INTERVAL '15 day')
            """), {"d1": today}).fetchone()[0] if engine.dialect.name != "sqlite" else 0
        st.success(f"Varredura concluída. Itens a vencer (15d): {due}. Para envio real, configure SMTP/Twilio e rode notifier.py")

    st.markdown("### Histórico")
    try:
        df = bk_finance.pd.read_sql(text("SELECT * FROM notification_events ORDER BY id DESC LIMIT 200"), engine)
        st.dataframe(df, use_container_width=True, height=360)
    except Exception:
        st.caption("Sem histórico (tabela será criada quando rodar o notificador).")

with tabs[1]:
    st.subheader("👥 Usuários")
    if can_view("admin"):
        bk_finance.users_ui(SessionLocal)
    else:
        st.warning("Apenas administradores podem gerenciar usuários.")

with tabs[2]:
    st.subheader("🧾 Auditoria")
    if can_view("admin"):
        bk_finance.audit_ui(SessionLocal)
    else:
        st.warning("Apenas administradores podem ver auditoria.")
