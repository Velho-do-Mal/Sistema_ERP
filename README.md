# BK_ERP

ERP em Streamlit + Neon/Postgres, focado em empresas de engenharia, empreiteiras e construtoras.

## Módulos
- **Home (executivo)**: resumo financeiro (fluxo de caixa e vencimentos), projetos, compras, propostas e documentos.
- **Financeiro**: app completo (usuários, auditoria, contas, lançamentos, relatórios e dashboards).
- **Projetos (PMBOK)**: TAP, EAP, Gantt, Curva S, riscos, lições e relatórios HTML.
- **Cadastros**: clientes, fornecedores, categorias, centros de custo + Leads.
- **Compras**: pedidos de compra integrados (fornecedor/projeto).
- **Vendas/Propostas**: propostas + ordens de venda e botão para gerar conta a receber no Financeiro.
- **Documentos**: upload/download e associação com projeto/cliente/fornecedor.
- **Admin & Notificações**: configurações e histórico.

## Como rodar localmente
1. Crie um `.env` baseado no `.env.example` e preencha `DATABASE_URL`.
2. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Rode:
   ```bash
   streamlit run Home.py
   ```

## Banco (Neon)
O app cria as tabelas automaticamente na primeira execução.

## Notificações (E-mail/WhatsApp)
O Streamlit não é ideal para jobs em background. Em produção, agende:
```bash
python notifier.py
```
Sugestão: cron (VM), Cloud Run job, ou GitHub Actions (dependendo do cenário).

## Publicação (pronto para nuvem)
Há um `Dockerfile` para subir em qualquer servidor/container (Cloud Run, ECS, Azure Container Apps etc.):
```bash
docker build -t bk_erp .
docker run -p 8501:8501 --env-file .env bk_erp
```
