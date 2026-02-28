"""
BK_ERP - Renderização de relatórios HTML para propostas.
"""
from __future__ import annotations
from pathlib import Path
import base64


def money_br(value) -> str:
    """Formata valor monetário no padrão brasileiro: R$ 1.234,56"""
    try:
        v = float(value or 0)
        formatted = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {formatted}"
    except Exception:
        return "R$ 0,00"


def _load_logo_b64(logo_path: Path) -> str:
    """Carrega logo como base64 para embutir no HTML."""
    try:
        if not logo_path.exists():
            return ""
        suffix = logo_path.suffix.lower()
        data = logo_path.read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        mime = {".svg": "image/svg+xml", ".png": "image/png",
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix, "image/png")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""


def render_proposta_html(
    template_path: Path,
    logo_path: Path,
    context: dict,
) -> str:
    """
    Renderiza proposta em HTML.
    - Se template_path existir, usa ele substituindo {{CHAVE}} pelo context.
    - Se não existir, gera HTML padrão BK.
    """
    logo_uri = _load_logo_b64(logo_path)
    context["LOGO_URI"] = logo_uri

    if template_path.exists():
        html = template_path.read_text(encoding="utf-8")
        for key, value in context.items():
            html = html.replace(f"{{{{{key}}}}}", str(value))
        return html

    # Template padrão BK (sem arquivo externo)
    items_html = context.get("ITENS_TABELA", "")
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Proposta {context.get('CODIGO','')}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; color: #1a1a2e; }}
  .page {{ max-width: 900px; margin: 20px auto; background: #fff; border-radius: 12px;
           box-shadow: 0 4px 24px rgba(0,0,0,0.1); overflow: hidden; }}
  .header {{ background: linear-gradient(135deg, #1E3A8A, #2563EB); padding: 32px 40px;
             display: flex; align-items: center; gap: 20px; }}
  .header img {{ height: 56px; }}
  .header-text h1 {{ color: #fff; font-size: 22px; font-weight: 900; letter-spacing: -0.02em; }}
  .header-text p {{ color: rgba(255,255,255,0.75); font-size: 13px; margin-top: 4px; }}
  .badge {{ background: rgba(255,255,255,0.15); color: #fff; border-radius: 8px;
            padding: 4px 14px; font-size: 12px; font-weight: 700; margin-left: auto; white-space: nowrap; }}
  .body {{ padding: 36px 40px; }}
  .section {{ margin-bottom: 28px; }}
  .section-title {{ font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
                    color: #2563EB; border-left: 4px solid #2563EB; padding-left: 10px; margin-bottom: 12px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .field {{ background: #f8faff; border-radius: 8px; padding: 12px 16px; border: 1px solid #e2e8f0; }}
  .field-label {{ font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;
                  letter-spacing: 0.05em; margin-bottom: 4px; }}
  .field-value {{ font-size: 14px; color: #1e293b; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 4px; }}
  thead {{ background: linear-gradient(135deg, #1E3A8A, #2563EB); color: #fff; }}
  th {{ padding: 10px 12px; font-size: 12px; font-weight: 700; text-align: left; }}
  td {{ padding: 9px 12px; font-size: 13px; border-bottom: 1px solid #e2e8f0; }}
  tr:nth-child(even) td {{ background: #f8faff; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .total-row td {{ background: linear-gradient(135deg, #1E3A8A, #2563EB) !important;
                   color: #fff !important; font-weight: 700; font-size: 15px; }}
  .footer {{ text-align: center; padding: 20px; color: #94a3b8; font-size: 12px;
             border-top: 1px solid #e2e8f0; margin-top: 32px; }}
  pre {{ white-space: pre-wrap; font-family: inherit; font-size: 13px; color: #1e293b; }}
</style>
</head>
<body>
<div class="page">
  <div class="header">
    {'<img src="' + logo_uri + '" alt="BK"/>' if logo_uri else ''}
    <div class="header-text">
      <h1>Proposta Técnica e Comercial</h1>
      <p>BK Engenharia e Tecnologia</p>
    </div>
    <div class="badge">{context.get('CODIGO','')}</div>
  </div>
  <div class="body">

    <div class="section">
      <div class="section-title">Identificação</div>
      <div class="grid-2">
        <div class="field">
          <div class="field-label">Título</div>
          <div class="field-value">{context.get('TITULO','')}</div>
        </div>
        <div class="field">
          <div class="field-label">Cliente</div>
          <div class="field-value">{context.get('CLIENTE','')}</div>
        </div>
        <div class="field">
          <div class="field-label">Data de Emissão</div>
          <div class="field-value">{context.get('DATA_EMISSAO','')}</div>
        </div>
        <div class="field">
          <div class="field-label">Validade</div>
          <div class="field-value">{context.get('VALIDADE','')}</div>
        </div>
      </div>
    </div>

    {'<div class="section"><div class="section-title">Objetivo</div><pre>' + context.get('OBJETIVO','') + '</pre></div>' if context.get('OBJETIVO') else ''}
    {'<div class="section"><div class="section-title">Escopo</div><pre>' + context.get('ESCOPO','') + '</pre></div>' if context.get('ESCOPO') else ''}

    <div class="section">
      <div class="section-title">Itens / Serviços</div>
      <table>
        <thead><tr><th>#</th><th>Descrição</th><th>Un</th><th class="num">Qtde</th>
                   <th class="num">Preço Unit.</th><th class="num">Total</th></tr></thead>
        <tbody>{items_html}</tbody>
        <tr class="total-row">
          <td colspan="5" style="text-align:right; padding-right:16px;">TOTAL GERAL</td>
          <td class="num">{context.get('TOTAL_GERAL','')}</td>
        </tr>
      </table>
    </div>

    <div class="grid-2" style="margin-bottom:28px;">
      {'<div class="field"><div class="field-label">Condições de Pagamento</div><pre>' + context.get('PAGAMENTO','') + '</pre></div>' if context.get('PAGAMENTO') else ''}
      {'<div class="field"><div class="field-label">Prazo de Entrega</div><pre>' + context.get('ENTREGA','') + '</pre></div>' if context.get('ENTREGA') else ''}
    </div>

    {'<div class="section"><div class="section-title">Observações</div><pre>' + context.get('OBSERVACOES','') + '</pre></div>' if context.get('OBSERVACOES') else ''}

  </div>
  <div class="footer">BK Engenharia e Tecnologia — Proposta {context.get('CODIGO','')} — {context.get('DATA_EMISSAO','')}</div>
</div>
</body>
</html>"""
