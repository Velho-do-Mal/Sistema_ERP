# test_imports.py
# Testa imports dos módulos que adicionamos/alteramos para diagnosticar erros.
import traceback

def main():
    try:
        from bk_erp_shared.theme import apply_theme, load_svg
        print("IMPORT theme OK")
    except Exception:
        print("ERRO ao importar bk_erp_shared.theme:")
        traceback.print_exc()

    try:
        from bk_erp_shared.finance_bridge import link_order_to_finance, create_installments
        print("IMPORT finance_bridge OK")
    except Exception:
        print("ERRO ao importar bk_erp_shared.finance_bridge:")
        traceback.print_exc()

    try:
        from bk_erp_shared.erp_db import ensure_erp_tables, get_finance_db
        print("IMPORT erp_db OK")
    except Exception:
        print("ERRO ao importar bk_erp_shared.erp_db:")
        traceback.print_exc()

if __name__ == '__main__':
    main()
