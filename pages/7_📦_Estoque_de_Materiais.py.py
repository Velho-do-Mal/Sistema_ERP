import pandas as pd
import streamlit as st

from datetime import date
from sqlalchemy import text

from bk_erp_shared.theme import apply_theme
from bk_erp_shared.erp_db import get_finance_db, ensure_erp_tables
from bk_erp_shared.auth import login_and_guard


st.set_page_config(page_title="📦 Estoque de Materiais", layout="wide")


def _to_iso(d: date | None) -> str | None:
    return d.isoformat() if d else None


def _from_iso(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def _table_exists(engine, table_name: str) -> bool:
    try:
        q = text(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_name = :t
            LIMIT 1
            """
        )
        with engine.begin() as conn:
            return conn.execute(q, {"t": table_name}).fetchone() is not None
    except Exception:
        # SQLite fallback / permissive
        try:
            with engine.begin() as conn:
                conn.execute(text(f"SELECT 1 FROM {table_name} LIMIT 1"))
            return True
        except Exception:
            return False


def load_suppliers(engine) -> pd.DataFrame:
    if not _table_exists(engine, "suppliers"):
        return pd.DataFrame(columns=["id", "name"])
    try:
        return pd.read_sql(text("SELECT id, name FROM suppliers ORDER BY name"), engine)
    except Exception:
        return pd.DataFrame(columns=["id", "name"])


def load_projects(engine) -> pd.DataFrame:
    if not _table_exists(engine, "projects"):
        return pd.DataFrame(columns=["id", "nome"])
    try:
        return pd.read_sql(text("SELECT id, nome FROM projects ORDER BY nome"), engine)
    except Exception:
        return pd.DataFrame(columns=["id", "nome"])


def load_stock(engine) -> pd.DataFrame:
    # Tenta trazer nomes (se existirem), mas não falha se a tabela de fornecedores ainda não existir.
    has_sup = _table_exists(engine, "suppliers")
    has_proj = _table_exists(engine, "projects")

    if has_sup and has_proj:
        sql = """
            SELECT
                sm.id,
                sm.material_code AS "Código",
                sm.description   AS "Descrição",
                COALESCE(s.name, '') AS "Fornecedor",
                COALESCE(p.nome, '') AS "Projeto",
                sm.qty_purchased AS "Qtd. Comprada",
                sm.purchase_value AS "Valor da Compra",
                sm.purchase_date AS "Data da Compra",
                sm.validity_date AS "Validade",
                COALESCE(sm.notes,'') AS "Observação",
                COALESCE(sm.used_qty, 0) AS "Utilizado"
            FROM stock_materials sm
            LEFT JOIN suppliers s ON s.id = sm.supplier_id
            LEFT JOIN projects p  ON p.id = sm.project_id
            ORDER BY sm.id DESC
        """
    elif has_sup and not has_proj:
        sql = """
            SELECT
                sm.id,
                sm.material_code AS "Código",
                sm.description   AS "Descrição",
                COALESCE(s.name, '') AS "Fornecedor",
                CAST(COALESCE(sm.project_id,0) AS TEXT) AS "Projeto",
                sm.qty_purchased AS "Qtd. Comprada",
                sm.purchase_value AS "Valor da Compra",
                sm.purchase_date AS "Data da Compra",
                sm.validity_date AS "Validade",
                COALESCE(sm.notes,'') AS "Observação",
                COALESCE(sm.used_qty, 0) AS "Utilizado"
            FROM stock_materials sm
            LEFT JOIN suppliers s ON s.id = sm.supplier_id
            ORDER BY sm.id DESC
        """
    elif (not has_sup) and has_proj:
        sql = """
            SELECT
                sm.id,
                sm.material_code AS "Código",
                sm.description   AS "Descrição",
                CAST(COALESCE(sm.supplier_id,0) AS TEXT) AS "Fornecedor",
                COALESCE(p.nome, '') AS "Projeto",
                sm.qty_purchased AS "Qtd. Comprada",
                sm.purchase_value AS "Valor da Compra",
                sm.purchase_date AS "Data da Compra",
                sm.validity_date AS "Validade",
                COALESCE(sm.notes,'') AS "Observação",
                COALESCE(sm.used_qty, 0) AS "Utilizado"
            FROM stock_materials sm
            LEFT JOIN projects p  ON p.id = sm.project_id
            ORDER BY sm.id DESC
        """
    else:
        sql = """
            SELECT
                sm.id,
                sm.material_code AS "Código",
                sm.description   AS "Descrição",
                CAST(COALESCE(sm.supplier_id,0) AS TEXT) AS "Fornecedor",
                CAST(COALESCE(sm.project_id,0) AS TEXT) AS "Projeto",
                sm.qty_purchased AS "Qtd. Comprada",
                sm.purchase_value AS "Valor da Compra",
                sm.purchase_date AS "Data da Compra",
                sm.validity_date AS "Validade",
                COALESCE(sm.notes,'') AS "Observação",
                COALESCE(sm.used_qty, 0) AS "Utilizado"
            FROM stock_materials sm
            ORDER BY sm.id DESC
        """

    df = pd.read_sql(text(sql), engine)

    # saldo (calculado)
    for col in ["Qtd. Comprada", "Utilizado", "Valor da Compra"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["Saldo"] = df["Qtd. Comprada"] - df["Utilizado"]

    # Datas para exibição
    for c in ["Data da Compra", "Validade"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: str(x)[:10] if x else "")

    return df


def upsert_purchase(engine, payload: dict) -> tuple[str, int | None]:
    """Cria/atualiza compra.

    Regra: se existir um item com MESMO código+projeto+fornecedor+descrição e ainda tiver saldo>0,
    soma a nova quantidade ao registro existente.
    """

    material_code = (payload.get("material_code") or "").strip()
    description = (payload.get("description") or "").strip()
    supplier_id = payload.get("supplier_id")
    project_id = payload.get("project_id")
    qty = float(payload.get("qty_purchased") or 0.0)
    val = float(payload.get("purchase_value") or 0.0)

    if not material_code:
        return "Código do material é obrigatório.", None
    if qty <= 0:
        return "Quantidade comprada deve ser maior que 0.", None

    with engine.begin() as conn:
        # Edição explícita por ID
        edit_id = payload.get("id")
        if edit_id and int(edit_id) > 0:
            conn.execute(
                text(
                    """
                    UPDATE stock_materials
                    SET material_code=:material_code,
                        description=:description,
                        supplier_id=:supplier_id,
                        project_id=:project_id,
                        qty_purchased=:qty_purchased,
                        purchase_value=:purchase_value,
                        purchase_date=:purchase_date,
                        validity_date=:validity_date,
                        notes=:notes,
                        updated_at=NOW()
                    WHERE id=:id
                    """
                ),
                {
                    "id": int(edit_id),
                    "material_code": material_code,
                    "description": description,
                    "supplier_id": supplier_id,
                    "project_id": project_id,
                    "qty_purchased": qty,
                    "purchase_value": val,
                    "purchase_date": payload.get("purchase_date"),
                    "validity_date": payload.get("validity_date"),
                    "notes": payload.get("notes"),
                },
            )
            return "Material atualizado.", int(edit_id)

        # Busca possível item existente com saldo > 0
        existing = conn.execute(
            text(
                """
                SELECT id, qty_purchased, COALESCE(used_qty,0) AS used_qty
                FROM stock_materials
                WHERE material_code=:material_code
                  AND COALESCE(description,'')=:description
                  AND COALESCE(supplier_id,0)=COALESCE(:supplier_id,0)
                  AND COALESCE(project_id,0)=COALESCE(:project_id,0)
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {
                "material_code": material_code,
                "description": description,
                "supplier_id": supplier_id,
                "project_id": project_id,
            },
        ).fetchone()

        if existing is not None:
            ex_id, ex_qty, ex_used = int(existing[0]), float(existing[1] or 0.0), float(existing[2] or 0.0)
            if (ex_qty - ex_used) > 0:
                # soma ao existente
                conn.execute(
                    text(
                        """
                        UPDATE stock_materials
                        SET qty_purchased = COALESCE(qty_purchased,0) + :add_qty,
                            purchase_value = COALESCE(purchase_value,0) + :add_val,
                            purchase_date = COALESCE(:purchase_date, purchase_date),
                            validity_date = COALESCE(:validity_date, validity_date),
                            notes = CASE
                                WHEN COALESCE(notes,'') = '' THEN :notes
                                WHEN COALESCE(:notes,'') = '' THEN notes
                                ELSE notes || E'\n' || :notes
                            END,
                            updated_at = NOW()
                        WHERE id = :id
                        """
                    ),
                    {
                        "id": ex_id,
                        "add_qty": qty,
                        "add_val": val,
                        "purchase_date": payload.get("purchase_date"),
                        "validity_date": payload.get("validity_date"),
                        "notes": payload.get("notes"),
                    },
                )
                return "Compra somada ao material existente (saldo ainda disponível).", ex_id

        # insere novo
        new_id = conn.execute(
            text(
                """
                INSERT INTO stock_materials
                    (material_code, description, supplier_id, project_id,
                     qty_purchased, purchase_value, purchase_date, validity_date,
                     notes, used_qty, created_at, updated_at)
                VALUES
                    (:material_code, :description, :supplier_id, :project_id,
                     :qty_purchased, :purchase_value, :purchase_date, :validity_date,
                     :notes, 0, NOW(), NOW())
                RETURNING id
                """
            ),
            {
                "material_code": material_code,
                "description": description,
                "supplier_id": supplier_id,
                "project_id": project_id,
                "qty_purchased": qty,
                "purchase_value": val,
                "purchase_date": payload.get("purchase_date"),
                "validity_date": payload.get("validity_date"),
                "notes": payload.get("notes"),
            },
        ).fetchone()

        return "Material registrado.", int(new_id[0]) if new_id else None


def delete_material(engine, rid: int) -> bool:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM stock_materials WHERE id=:id"), {"id": int(rid)})
    return True


def update_used_qty(engine, updates: list[dict]) -> int:
    if not updates:
        return 0
    with engine.begin() as conn:
        for u in updates:
            conn.execute(
                text(
                    """
                    UPDATE stock_materials
                    SET used_qty=:used_qty,
                        updated_at=NOW()
                    WHERE id=:id
                    """
                ),
                {"id": int(u["id"]), "used_qty": float(u["used_qty"])},
            )
    return len(updates)


def main():
    apply_theme()

    engine, SessionLocal = get_finance_db()
    ensure_erp_tables(engine)

    login_and_guard(SessionLocal)

    st.title("📦 Estoque de Materiais")
    st.caption("Cadastro de compras, controle de utilização e saldo")

    if not _table_exists(engine, "stock_materials"):
        st.error("Tabela de estoque ainda não existe. (ensure_erp_tables não criou)")
        st.stop()

    # Dados auxiliares
    df_sup = load_suppliers(engine)
    sup_opts = ["(Sem fornecedor)"] + [f"{int(r.id)} - {r.name}" for r in df_sup.itertuples(index=False)]

    df_proj = load_projects(engine)
    proj_opts = ["(Sem projeto)"] + [f"{int(r.id)} - {r.nome}" for r in df_proj.itertuples(index=False)]

    # --- Formulário ---
    st.subheader("Cadastro / Edição")

    df_stock_raw = pd.read_sql(text("SELECT id, material_code, description FROM stock_materials ORDER BY id DESC"), engine)
    edit_choices = ["(Novo)"] + [f"{int(r.id)} - {r.material_code} - {r.description}" for r in df_stock_raw.itertuples(index=False)]

    sel = st.selectbox("Selecionar material para editar", edit_choices, index=0)
    edit_id = 0
    if sel != "(Novo)":
        edit_id = int(sel.split(" - ", 1)[0])

    # Carrega valores
    current = {
        "material_code": "",
        "description": "",
        "supplier_id": None,
        "project_id": None,
        "qty_purchased": 0.0,
        "purchase_value": 0.0,
        "purchase_date": date.today(),
        "validity_date": None,
        "notes": "",
    }

    if edit_id > 0:
        row = pd.read_sql(
            text(
                """
                SELECT id, material_code, description, supplier_id, project_id,
                       qty_purchased, purchase_value, purchase_date, validity_date, notes
                FROM stock_materials
                WHERE id=:id
                """
            ),
            engine,
            params={"id": edit_id},
        )
        if not row.empty:
            r = row.iloc[0].to_dict()
            current.update(
                {
                    "material_code": r.get("material_code") or "",
                    "description": r.get("description") or "",
                    "supplier_id": r.get("supplier_id"),
                    "project_id": r.get("project_id"),
                    "qty_purchased": float(r.get("qty_purchased") or 0.0),
                    "purchase_value": float(r.get("purchase_value") or 0.0),
                    "purchase_date": _from_iso(r.get("purchase_date")) or date.today(),
                    "validity_date": _from_iso(r.get("validity_date")),
                    "notes": r.get("notes") or "",
                }
            )

    # Indexes selects
    def _idx_from_id(opts: list[str], target_id: int | None) -> int:
        if not target_id:
            return 0
        for i, o in enumerate(opts):
            if o.startswith(f"{int(target_id)} - "):
                return i
        return 0

    with st.form("stock_form"):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            material_code = st.text_input("Código do material *", value=current["material_code"])
        with c2:
            description = st.text_input("Descrição *", value=current["description"])
        with c3:
            supplier_sel = st.selectbox(
                "Fornecedor",
                options=sup_opts,
                index=_idx_from_id(sup_opts, current.get("supplier_id")),
            )
        with c4:
            project_sel = st.selectbox(
                "Projeto",
                options=proj_opts,
                index=_idx_from_id(proj_opts, current.get("project_id")),
            )

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            qty_purchased = st.number_input("Quantidade comprada *", min_value=0.0, value=float(current["qty_purchased"]), step=1.0)
        with c6:
            purchase_value = st.number_input("Valor da compra", min_value=0.0, value=float(current["purchase_value"]), step=50.0)
        with c7:
            purchase_date = st.date_input("Data da compra", value=current["purchase_date"])
        with c8:
            validity_date = st.date_input(
                "Validade (opcional)",
                value=current["validity_date"] if current["validity_date"] else date.today(),
            )
            use_validity = st.checkbox("Usar validade", value=bool(current.get("validity_date")))

        notes = st.text_area("Observação", value=current.get("notes") or "")

        colA, colB, colC = st.columns([1, 1, 2])
        with colA:
            btn_save = st.form_submit_button("💾 Salvar")
        with colB:
            btn_delete = st.form_submit_button("🗑️ Excluir", type="secondary")
        with colC:
            if edit_id > 0:
                st.caption(f"Editando ID: {edit_id}")
            else:
                st.caption("Novo material")

    if btn_delete:
        if edit_id <= 0:
            st.warning("Selecione um registro para excluir.")
        else:
            try:
                delete_material(engine, edit_id)
                st.success("Registro excluído.")
                st.rerun()
            except Exception as e:
                st.error(f"Falha ao excluir: {e}")

    if btn_save:
        try:
            supplier_id = int(supplier_sel.split(" - ", 1)[0]) if supplier_sel != "(Sem fornecedor)" else None
            project_id = int(project_sel.split(" - ", 1)[0]) if project_sel != "(Sem projeto)" else None

            payload = {
                "id": int(edit_id) if edit_id > 0 else None,
                "material_code": material_code,
                "description": description,
                "supplier_id": supplier_id,
                "project_id": project_id,
                "qty_purchased": float(qty_purchased),
                "purchase_value": float(purchase_value),
                "purchase_date": _to_iso(purchase_date),
                "validity_date": _to_iso(validity_date) if use_validity else None,
                "notes": notes.strip() or None,
            }
            msg, rid = upsert_purchase(engine, payload)
            st.success(msg)
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao salvar: {e}")

    st.markdown("---")

    # --- Tabela tipo Excel ---
    st.subheader("Tabela de Estoque")
    st.caption("O campo **Utilizado** pode ser editado diretamente na tabela. O **Saldo** é calculado: Qtd. Comprada - Utilizado.")

    df = load_stock(engine)

    if df.empty:
        st.info("Nenhum material cadastrado.")
        return

    # Editor: somente Utilizado editável
    df_editor = df.copy()

    edited = st.data_editor(
        df_editor,
        hide_index=True,
        width="stretch",
        disabled=[
            "id",
            "Código",
            "Descrição",
            "Fornecedor",
            "Projeto",
            "Qtd. Comprada",
            "Valor da Compra",
            "Data da Compra",
            "Validade",
            "Observação",
            "Saldo",
        ],
        key="stock_editor",
    )

    if st.button("💾 Salvar utilizado", key="save_used"):
        # detecta alterações
        updates: list[dict] = []
        try:
            base_map = {int(r.id): float(r["Utilizado"]) for _, r in df.iterrows()}
            for _, r in edited.iterrows():
                rid = int(r["id"])
                new_u = float(pd.to_numeric(r["Utilizado"], errors="coerce") or 0.0)
                old_u = float(base_map.get(rid, 0.0))
                if abs(new_u - old_u) > 1e-9:
                    updates.append({"id": rid, "used_qty": max(new_u, 0.0)})

            n = update_used_qty(engine, updates)
            st.success(f"Atualizado: {n} item(ns).")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao salvar utilizado: {e}")


if __name__ == "__main__":
    main()
