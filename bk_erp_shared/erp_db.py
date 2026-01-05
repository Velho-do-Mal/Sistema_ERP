
"""
BK_ERP - Camada de dados compartilhada

- Reaproveita conexão e autenticação do módulo financeiro (bk_finance.py).
- Cria tabelas adicionais (Compras, Vendas/Propostas, Documentos, Projetos, Notificações)
  sem quebrar o módulo Financeiro.

Observação: este arquivo não contém senhas. Tudo vem de variáveis de ambiente:
DATABASE_URL (+ PGPASSWORD opcional) e sslmode=require.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Tuple

import streamlit as st
from sqlalchemy import text

import bk_finance  # módulo financeiro refatorado (mantém User, Transaction etc.)


@st.cache_resource
def get_finance_db() -> Tuple[object, object]:
    """Retorna (engine, SessionLocal) do módulo financeiro."""
    return bk_finance.get_db()


def ensure_erp_tables() -> None:
    engine, _ = get_finance_db()
    dialect = engine.dialect.name

    # Tipos por dialeto
    if dialect == "sqlite":
        id_col = "INTEGER PRIMARY KEY AUTOINCREMENT"
        ts = "DATETIME"
        bool_t = "BOOLEAN"
        money = "FLOAT"
        floaty = "FLOAT"
        blob = "BLOB"
    else:
        id_col = "SERIAL PRIMARY KEY"
        ts = "TIMESTAMP"
        bool_t = "BOOLEAN"
        money = "DOUBLE PRECISION"
        floaty = "DOUBLE PRECISION"
        blob = "BYTEA"

    statements = [
        # -------------------------
        # PROJETOS (estado em JSON)
        # -------------------------
        f"""
        CREATE TABLE IF NOT EXISTS projects (
            id {id_col},
            data TEXT,
            nome TEXT,
            status TEXT,
            dataInicio TEXT,
            gerente TEXT,
            patrocinador TEXT,
            encerrado {bool_t} DEFAULT FALSE,
            client_id INTEGER NULL
        );
        """,
        
        # -------------------------
        # BOLETIM DE MEDIÇÕES (ENGENHARIA)
        # - Itens do Contrato por Projeto
        # - Boletins por período (mensal ou por janela de datas)
        # -------------------------
        f"""
        CREATE TABLE IF NOT EXISTS project_contract_items (
            id {id_col},
            project_id INTEGER NOT NULL,
            item TEXT NOT NULL,
            atividade_geral TEXT,
            unidade TEXT DEFAULT 'un',
            qtde_contratada {floaty} DEFAULT 0,
            valor_unit {money} DEFAULT 0,
            valor_total {money} DEFAULT 0,
            ativo {bool_t} DEFAULT TRUE,
            created_at {ts} DEFAULT CURRENT_TIMESTAMP,
            updated_at {ts}
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS measurements (
            id {id_col},
            project_id INTEGER NOT NULL,
            period_start DATE,
            period_end DATE,
            reference TEXT,
            status TEXT DEFAULT 'rascunho', -- rascunho | enviado | aprovado | rejeitado
            delay_responsibility TEXT DEFAULT 'N/A', -- BK | CLIENTE | N/A
            notes TEXT,
            approved_by INTEGER,
            approved_at {ts},
            created_at {ts} DEFAULT CURRENT_TIMESTAMP,
            updated_at {ts}
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS measurement_items (
            id {id_col},
            measurement_id INTEGER NOT NULL,
            contract_item_id INTEGER,
            descricao TEXT,
            unidade TEXT,
            qtde_periodo {floaty} DEFAULT 0,
            valor_unit {money} DEFAULT 0,
            valor_periodo {money} DEFAULT 0,
            qtde_acumulada {floaty} DEFAULT 0,
            valor_acumulado {money} DEFAULT 0
        );
        """,
# -------------------------
        # LEADS / CLIENTES (CRM)
        # -------------------------
        f"""
        CREATE TABLE IF NOT EXISTS leads (
            id {id_col},
            name TEXT NOT NULL,
            company TEXT,
            email TEXT,
            phone TEXT,
            source TEXT,
            stage TEXT,
            value_estimate {money} DEFAULT 0,
            status TEXT DEFAULT 'novo',
            notes TEXT,
            created_at {ts} DEFAULT CURRENT_TIMESTAMP
        );
        """,
        # -------------------------
        # PROPOSTAS / VENDAS
        # -------------------------
        f"""
        CREATE TABLE IF NOT EXISTS proposals (
            id {id_col},
            code TEXT,
            title TEXT NOT NULL,
            client_id INTEGER NULL,
            lead_id INTEGER NULL,
            value_total {money} DEFAULT 0,
            status TEXT DEFAULT 'rascunho',
            created_at {ts} DEFAULT CURRENT_TIMESTAMP,
            valid_until TEXT,
            notes TEXT
        );
        """,
        # -------------------------
        # CATÁLOGO DE SERVIÇOS/PRODUTOS
        # -------------------------
        f"""
        CREATE TABLE IF NOT EXISTS product_services (
            id {id_col},
            code TEXT,
            type TEXT DEFAULT 'servico',
            name TEXT NOT NULL,
            description TEXT,
            default_unit TEXT,
            default_unit_price {money} DEFAULT 0,
            active {bool_t} DEFAULT TRUE,
            created_at {ts} DEFAULT CURRENT_TIMESTAMP
        );
        """,
        # -------------------------
        # ITENS DA PROPOSTA
        # -------------------------
        f"""
        CREATE TABLE IF NOT EXISTS proposal_items (
            id {id_col},
            proposal_id INTEGER NOT NULL,
            product_service_id INTEGER NULL,
            description TEXT NOT NULL,
            unit TEXT,
            qty {floaty} DEFAULT 1,
            unit_price {money} DEFAULT 0,
            total {money} DEFAULT 0,
            sort_order INTEGER DEFAULT 0,
            created_at {ts} DEFAULT CURRENT_TIMESTAMP
        );
        """,
        # -------------------------
        # CAMPOS COMPLEMENTARES NA PROPOSTA (idempotente)
        # -------------------------

        f"""
        CREATE TABLE IF NOT EXISTS sales_orders (
            id {id_col},
            proposal_id INTEGER NULL,
            client_id INTEGER NULL,
            order_date TEXT,
            value_total {money} DEFAULT 0,
            status TEXT DEFAULT 'aberta',
            notes TEXT
        );
        """,
        # -------------------------
        # COMPRAS
        # -------------------------
        f"""
        CREATE TABLE IF NOT EXISTS purchase_orders (
            id {id_col},
            code TEXT,
            supplier_id INTEGER NULL,
            project_id INTEGER NULL,
            order_date TEXT,
            expected_date TEXT,
            value_total {money} DEFAULT 0,
            status TEXT DEFAULT 'aberta',
            notes TEXT
        );
        """,
        # -------------------------
        # DOCUMENTOS (anexos gerais)
        # -------------------------
        f"""
        CREATE TABLE IF NOT EXISTS documents (
            id {id_col},
            title TEXT NOT NULL,
            doc_type TEXT,
            project_id INTEGER NULL,
            client_id INTEGER NULL,
            supplier_id INTEGER NULL,
            uploaded_by TEXT,
            uploaded_at {ts} DEFAULT CURRENT_TIMESTAMP,
            filename TEXT,
            content_type TEXT,
            data {blob},
            tags TEXT,
            notes TEXT
        );
        """,
        # -------------------------
        # NOTIFICAÇÕES
        # -------------------------
        f"""
        CREATE TABLE IF NOT EXISTS notification_events (
            id {id_col},
            entity TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            last_sent_at {ts},
            channel TEXT,
            status TEXT,
            info TEXT
        );
        """,
    ]

    # Campos complementares na proposta (migração idempotente):
    # - Postgres suporta ADD COLUMN IF NOT EXISTS e múltiplas colunas no mesmo ALTER
    # - SQLite NÃO suporta (dependendo da versão) 'IF NOT EXISTS' e não aceita múltiplos ADD COLUMN em um único ALTER
    if dialect != "sqlite":
        statements.append(f"""
        ALTER TABLE proposals
            ADD COLUMN IF NOT EXISTS objective TEXT,
            ADD COLUMN IF NOT EXISTS scope TEXT,
            ADD COLUMN IF NOT EXISTS resp_contratante TEXT,
            ADD COLUMN IF NOT EXISTS resp_contratado TEXT,
            ADD COLUMN IF NOT EXISTS payment_terms TEXT,
            ADD COLUMN IF NOT EXISTS delivery_terms TEXT,
            ADD COLUMN IF NOT EXISTS reference TEXT,
            ADD COLUMN IF NOT EXISTS observations TEXT,
            ADD COLUMN IF NOT EXISTS updated_at {ts};
        """)


    with engine.begin() as conn:
        for sql in statements:
            conn.execute(text(sql))



    # Migração SQLite: adicionar colunas da proposta se ainda não existirem
    if dialect == "sqlite":
        cols = {
            "objective": "TEXT",
            "scope": "TEXT",
            "resp_contratante": "TEXT",
            "resp_contratado": "TEXT",
            "payment_terms": "TEXT",
            "delivery_terms": "TEXT",
            "reference": "TEXT",
            "observations": "TEXT",
            "updated_at": ts,
        }
        with engine.begin() as conn:
            existing = [r[1] for r in conn.execute(text("PRAGMA table_info(proposals)"))]
            for col, ctype in cols.items():
                if col not in existing:
                    conn.execute(text(f'ALTER TABLE proposals ADD COLUMN "{col}" {ctype}'))


    # Migração: garantir colunas do pipeline (leads) para Vendas/Propostas
    if dialect == "sqlite":
        with engine.begin() as conn:
            existing = {row[1] for row in conn.execute(text("PRAGMA table_info(leads)"))}
            lead_cols = {
                "source": "TEXT",
                "stage": "TEXT",
                "value_estimate": money,
            }
            for col, ctype in lead_cols.items():
                if col not in existing:
                    conn.execute(text(f'ALTER TABLE leads ADD COLUMN "{col}" {ctype}'))
            # Harmonizar stage com status quando aplicável
            try:
                conn.execute(text("UPDATE leads SET stage = COALESCE(stage, status) WHERE stage IS NULL"))
            except Exception:
                pass
            try:
                conn.execute(text("UPDATE leads SET value_estimate = 0 WHERE value_estimate IS NULL"))
            except Exception:
                pass

    else:
        with engine.begin() as conn:
            try:
                conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS source TEXT"))
                conn.execute(text("ALTER TABLE leads ADD COLUMN IF NOT EXISTS stage TEXT"))
                conn.execute(text(f"ALTER TABLE leads ADD COLUMN IF NOT EXISTS value_estimate {money}"))
                conn.execute(text("UPDATE leads SET stage = COALESCE(stage, status) WHERE stage IS NULL"))
                conn.execute(text("UPDATE leads SET value_estimate = 0 WHERE value_estimate IS NULL"))
            except Exception:
                pass
    # Ajustes de colunas (quando a tabela já existe, ex.: criada pelo módulo de Projetos)
    if dialect != "sqlite":
        with engine.begin() as conn:
            try:
                conn.execute(text("ALTER TABLE projects ADD COLUMN IF NOT EXISTS client_id INTEGER"))
            except Exception:
                pass


    # Migração/normalização: Boletim de Medições (garantir colunas/tabelas no Postgres e SQLite)
    if dialect == "sqlite":
        with engine.begin() as conn:
            # project_contract_items
            try:
                existing = {row[1] for row in conn.execute(text("PRAGMA table_info(project_contract_items)"))}
                cols = {
                    "atividade_geral": "TEXT",
                    "unidade": "TEXT",
                    "qtde_contratada": floaty,
                    "valor_unit": money,
                    "valor_total": money,
                    "ativo": bool_t,
                    "updated_at": ts,
                }
                for col, ctype in cols.items():
                    if col not in existing:
                        conn.execute(text(f'ALTER TABLE project_contract_items ADD COLUMN "{col}" {ctype}'))
            except Exception:
                pass

            # measurements
            try:
                existing = {row[1] for row in conn.execute(text("PRAGMA table_info(measurements)"))}
                cols = {
                    "period_start": "DATE",
                    "period_end": "DATE",
                    "reference": "TEXT",
                    "status": "TEXT",
                    "delay_responsibility": "TEXT",
                    "notes": "TEXT",
                    "approved_by": "INTEGER",
                    "approved_at": ts,
                    "updated_at": ts,
                }
                for col, ctype in cols.items():
                    if col not in existing:
                        conn.execute(text(f'ALTER TABLE measurements ADD COLUMN "{col}" {ctype}'))
            except Exception:
                pass

            # measurement_items
            try:
                existing = {row[1] for row in conn.execute(text("PRAGMA table_info(measurement_items)"))}
                cols = {
                    "contract_item_id": "INTEGER",
                    "descricao": "TEXT",
                    "unidade": "TEXT",
                    "qtde_periodo": floaty,
                    "valor_unit": money,
                    "valor_periodo": money,
                    "qtde_acumulada": floaty,
                    "valor_acumulado": money,
                }
                for col, ctype in cols.items():
                    if col not in existing:
                        conn.execute(text(f'ALTER TABLE measurement_items ADD COLUMN "{col}" {ctype}'))
            except Exception:
                pass
    else:
        with engine.begin() as conn:
            try:
                conn.execute(text("ALTER TABLE project_contract_items ADD COLUMN IF NOT EXISTS atividade_geral TEXT"))
                conn.execute(text("ALTER TABLE project_contract_items ADD COLUMN IF NOT EXISTS unidade TEXT"))
                conn.execute(text(f"ALTER TABLE project_contract_items ADD COLUMN IF NOT EXISTS qtde_contratada {floaty}"))
                conn.execute(text(f"ALTER TABLE project_contract_items ADD COLUMN IF NOT EXISTS valor_unit {money}"))
                conn.execute(text(f"ALTER TABLE project_contract_items ADD COLUMN IF NOT EXISTS valor_total {money}"))
                conn.execute(text("ALTER TABLE project_contract_items ADD COLUMN IF NOT EXISTS ativo BOOLEAN"))
                conn.execute(text(f"ALTER TABLE project_contract_items ADD COLUMN IF NOT EXISTS updated_at {ts}"))
            except Exception:
                pass
            try:
                conn.execute(text("ALTER TABLE measurements ADD COLUMN IF NOT EXISTS period_start DATE"))
                conn.execute(text("ALTER TABLE measurements ADD COLUMN IF NOT EXISTS period_end DATE"))
                conn.execute(text("ALTER TABLE measurements ADD COLUMN IF NOT EXISTS reference TEXT"))
                conn.execute(text("ALTER TABLE measurements ADD COLUMN IF NOT EXISTS status TEXT"))
                conn.execute(text("ALTER TABLE measurements ADD COLUMN IF NOT EXISTS delay_responsibility TEXT"))
                conn.execute(text("ALTER TABLE measurements ADD COLUMN IF NOT EXISTS notes TEXT"))
                conn.execute(text("ALTER TABLE measurements ADD COLUMN IF NOT EXISTS approved_by INTEGER"))
                conn.execute(text(f"ALTER TABLE measurements ADD COLUMN IF NOT EXISTS approved_at {ts}"))
                conn.execute(text(f"ALTER TABLE measurements ADD COLUMN IF NOT EXISTS updated_at {ts}"))
            except Exception:
                pass
            try:
                conn.execute(text("ALTER TABLE measurement_items ADD COLUMN IF NOT EXISTS contract_item_id INTEGER"))
                conn.execute(text("ALTER TABLE measurement_items ADD COLUMN IF NOT EXISTS descricao TEXT"))
                conn.execute(text("ALTER TABLE measurement_items ADD COLUMN IF NOT EXISTS unidade TEXT"))
                conn.execute(text(f"ALTER TABLE measurement_items ADD COLUMN IF NOT EXISTS qtde_periodo {floaty}"))
                conn.execute(text(f"ALTER TABLE measurement_items ADD COLUMN IF NOT EXISTS valor_unit {money}"))
                conn.execute(text(f"ALTER TABLE measurement_items ADD COLUMN IF NOT EXISTS valor_periodo {money}"))
                conn.execute(text(f"ALTER TABLE measurement_items ADD COLUMN IF NOT EXISTS qtde_acumulada {floaty}"))
                conn.execute(text(f"ALTER TABLE measurement_items ADD COLUMN IF NOT EXISTS valor_acumulado {money}"))
            except Exception:
                pass

def utcnow_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")