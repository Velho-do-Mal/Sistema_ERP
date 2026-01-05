
"""
BK_ERP - Autenticação compartilhada

Reaproveita o esquema de usuários do Financeiro (tabela users).
"""
from __future__ import annotations

import streamlit as st
import bk_finance


def login_and_guard(SessionLocal) -> None:
    """
    Mostra login na sidebar e bloqueia a página se não logado.
    """
    bk_finance.login_ui(SessionLocal)
    bk_finance.require_login()


def current_user():
    return bk_finance.current_user()


def can_view(role_needed: str) -> bool:
    return bk_finance.can_view(role_needed)
