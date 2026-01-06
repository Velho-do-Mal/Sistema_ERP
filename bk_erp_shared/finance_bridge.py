# bk_erp_shared/finance_bridge.py
# Ponte entre Compras/Vendas/Propostas e o módulo Financeiro (bk_finance).
# --------------------------------------------------------------------------------
# - create_installments(tx_type, description, installments, client_id=..., supplier_id=..., project_id=...)
#     cria lançamentos no módulo financeiro (uma Transaction por parcela) e agrupa
#     todos os lançamentos por recurrence_group (uuid).
#
# - link_order_to_finance(order_type, order_id, description, total_amount, entry_amount, parcels, ...)
#     helper que monta a lista de parcelas (entrada + parcelas) e chama create_installments.
#
# Observação:
# - Este arquivo usa a sessão do financeiro via bk_erp_shared.erp_db.get_finance_db()
# - Não toca em tabelas do domínio de propostas/compras: apenas cria transações.
# --------------------------------------------------------------------------------

from __future__ import annotations
import uuid
from datetime import date
from typing import List, Dict, Optional

# Reaproveita a função que obtém o engine e SessionLocal do módulo financeiro
from bk_erp_shared.erp_db import get_finance_db
import bk_finance  # modelo ORM do financeiro (Transaction, User, etc.)

def create_installments(tx_type: str,
                        description: str,
                        installments: List[Dict],
                        client_id: Optional[int] = None,
                        supplier_id: Optional[int] = None,
                        project_id: Optional[int] = None) -> List[int]:
    """
    Cria lançamentos (Transactions) no financeiro para cada item em "installments".

    Parâmetros:
    - tx_type: 'entrada' ou 'saida' (string).
    - description: descrição textual do lançamento (ex: "Venda #123").
    - installments: lista de dicts com {'amount': float, 'due_date': date}.
    - client_id / supplier_id / project_id: referencias opcionais.

    Retorna:
    - lista de ids (inteiros) das transactions criadas.
    """
    engine, SessionLocal = get_finance_db()
    recurrence_group = uuid.uuid4().hex  # agrupa as parcelas para edição futura
    created_ids: List[int] = []

    # Abre sessão ORM do módulo financeiro
    with SessionLocal() as session:
        # cria cada parcela como uma Transaction ORM do bk_finance
        for inst in installments:
            amt = float(inst.get("amount", 0))
            if amt <= 0:
                # ignora parcelas com valor zero ou negativo (defensivo)
                continue
            due = inst.get("due_date") or date.today()

            # Monta o objeto Transaction do bk_finance (schema já presente no módulo)
            tx = bk_finance.Transaction(
                date=due,  # competência: usamos a data de vencimento por padrão
                due_date=due,
                paid=False,
                description=description,
                amount=amt,
                type=tx_type,               # 'entrada' ou 'saida'
                client_id=client_id,
                supplier_id=supplier_id,
                recurrence_group=recurrence_group,
                reference=str(project_id) if project_id is not None else None,
            )
            session.add(tx)

        # confirma inserções
        session.commit()

        # recupera os ids criados (filtra pelo recurrence_group)
        rows = session.query(bk_finance.Transaction).filter_by(recurrence_group=recurrence_group).all()
        created_ids = [r.id for r in rows]

    return created_ids


def link_order_to_finance(order_type: str,
                          order_id: int,
                          description: str,
                          total_amount: float,
                          entry_amount: float,
                          parcels: List[Dict],
                          client_id: Optional[int] = None,
                          supplier_id: Optional[int] = None,
                          project_id: Optional[int] = None) -> List[int]:
    """
    Helper para ligar uma ordem (proposta / venda / compra) ao financeiro.

    Parâmetros:
    - order_type: 'sale'|'purchase' ou explicitamente 'entrada'|'saida'
    - order_id: id da ordem (apenas para referência no description se desejar)
    - description: texto para os lançamentos (ex: "Venda/Proposta #123")
    - total_amount: valor total da ordem (soma da entrada + parcelas)
    - entry_amount: valor de entrada (float, 0 se não houver)
    - parcels: lista [{'amount':float, 'due_date': date}, ...] representando as parcelas
    - client_id / supplier_id / project_id: ids opcionais para vinculação
    Retorna lista de transaction ids criadas.
    """
    # Normaliza tx_type
    if order_type in ("entrada", "saida"):
        tx_type = order_type
    elif order_type == "sale":
        tx_type = "entrada"
    else:
        tx_type = "saida"

    # Monta lista de parcelas (entrada + parcelas fornecidas)
    insts: List[Dict] = []
    entry_amount = float(entry_amount or 0)
    if entry_amount > 0:
        insts.append({"amount": entry_amount, "due_date": date.today()})

    # Adiciona as parcelas passadas pelo caller
    for p in parcels:
        insts.append({"amount": float(p.get("amount", 0)), "due_date": p.get("due_date")})

    # Validação da soma (não bloqueante - só advertência se divergente)
    s = sum(float(i["amount"]) for i in insts)
    if round(s, 2) != round(float(total_amount or 0), 2):
        # Aqui não lançamos exceção por compatibilidade, mas idealmente a UI
        # deveria ajustar a 1ª parcela para cobrir diferença de arredondamento.
        # Você pode logar/alertar no frontend se desejar.
        pass

    return create_installments(tx_type, description, insts,
                               client_id=client_id, supplier_id=supplier_id, project_id=project_id)
