# examples/proposals_finance_ui_example.py
# Exemplo de UI para gerar parcelas editáveis (entrada + N parcelas) e criar lançamentos no financeiro.
# --------------------------------------------------------------------------------
# - Usa st.experimental_data_editor() para edição das parcelas antes de confirmar.
# - Chama bk_erp_shared.finance_bridge.link_order_to_finance(...) para criar as transactions.
# - Integre este código no módulo real de propostas/vendas do seu app.
# --------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

# Importa a função helper que criamos
from bk_erp_shared.finance_bridge import link_order_to_finance

def ui_generate_finance_from_proposal(proposal_id: int, proposal_value: float, client_id: int = None, project_id: int = None):
    """
    Exemplo de função que renderiza o formulário para gerar lançamentos financeiros
    a partir de uma proposta/venda.
    - proposal_id: id da proposta (usado na descrição)
    - proposal_value: valor total da proposta (float)
    - client_id/project_id: ids opcionais para vincular a transação
    """
    st.subheader(f"Gerar conta no Financeiro • Proposta #{proposal_id}")

    # Layout em 3 colunas (número parcelas, valor entrada, primeiro vencimento)
    cols = st.columns(3)
    with cols[0]:
        parcelas = st.number_input("Número de parcelas", min_value=1, value=3, help="Quantas parcelas além da entrada (se houver)")
    with cols[1]:
        entrada = st.number_input("Valor da entrada", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    with cols[2]:
        primeiro_venc = st.date_input("Primeiro vencimento", value=date.today())

    # Calcula parcelas automáticas (divisão simples, ajusta diferença no 1º)
    remaining = float(proposal_value) - float(entrada)
    per = round(remaining / parcelas, 2)
    rows = []
    for i in range(parcelas):
        venc = primeiro_venc + relativedelta(months=i)
        rows.append({"parcela": i+1, "valor": per, "vencimento": venc})

    # Ajuste para diferença de arredondamento
    diff = round(remaining - per * parcelas, 2)
    if diff != 0:
        rows[0]["valor"] = rows[0]["valor"] + diff

    df = pd.DataFrame(rows)[["parcela", "valor", "vencimento"]]

    # Data editor experimental do Streamlit permite edição inline das parcelas
    # (observação: a API é experimental, mas é apropriada para este caso)
    edited = st.experimental_data_editor(df, num_rows="fixed")

    # Botão para confirmar e gerar os lançamentos no financeiro
    if st.button("Confirmar e gerar no financeiro"):
        parcels_payload = []
        for _, r in edited.iterrows():
            parcels_payload.append({"amount": float(r["valor"]), "due_date": r["vencimento"]})
        # Linka a ordem com o financeiro -> cria transactions
        ids = link_order_to_finance("sale", proposal_id,
                                    f"Venda/Proposta #{proposal_id}",
                                    proposal_value,
                                    entrada,
                                    parcels_payload,
                                    client_id=client_id,
                                    project_id=project_id)
        st.success(f"Criado(s) lançamento(s) financeiro(s): {ids}")

# Execução direta para teste rápido
if __name__ == "__main__":
    ui_generate_finance_from_proposal(1, 1000.0, client_id=1, project_id=1)
