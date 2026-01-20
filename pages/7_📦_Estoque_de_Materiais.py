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
    """Carrega a tabela de estoque (material_stock).

    A tabela **material_stock** é criada pelo `ensure_erp_tables` e é a mesma
    alimentada automaticamente pelo módulo **Compras**. Por isso, ao registrar
    uma compra, o item já aparece aqui automaticamente.

    Observação: nesta tabela guardamos `supplier_name` e `project_name` como texto.
    """
    sql = """
        SELECT
            id,
            material_code,
            description,
            COALESCE(supplier_name, '') AS supplier_name,
            COALESCE(project_name, '') AS project_name,
            qty_purchased,
            unit_price,
            purchase_date,
            validity_date,
            COALESCE(notes, '') AS notes,
            COALESCE(qty_used, 0) AS qty_used
        FROM material_stock
        ORDER BY purchase_date DESC NULLS LAST, id DESC
    """
    with engine.begin() as conn:
        try:
            df = pd.read_sql(text(sql), conn)
        except Exception:
            # SQLite não aceita NULLS LAST
            sql2 = sql.replace(' NULLS LAST', '')
            df = pd.read_sql(text(sql2), conn)

    # Normalizações para o editor/tabela
    for c in ['qty_purchased', 'unit_price', 'qty_used']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    return df


def get_material_by_id(engine, rid: int) -> dict | None:
    """Retorna 1 material por ID (material_stock)."""
    df = pd.read_sql(text("SELECT * FROM material_stock WHERE id = :id"), engine, params={"id": int(rid)})
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def upsert_purchase(engine, payload: dict) -> tuple[str, int | None]:
    """Cria/atualiza compra.

    Regra: se existir um item com MESMO código+projeto+fornecedor+descrição e ainda tiver saldo>0,
    soma a nova quantidade ao registro existente.
    """

    material_code = (payload.get("material_code") or "").strip()
    description = (payload.get("description") or "").strip()
    supplier_name = payload.get("supplier_name")
    project_name = payload.get("project_name")
    qty = float(payload.get("qty_purchased") or 0.0)
    val = float(payload.get("unit_price") or 0.0)

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
                    UPDATE material_stock
                    SET material_code=:material_code,
                        description=:description,
                        supplier_name=:supplier_name,
                        project_name=:project_name,
                        qty_purchased=:qty_purchased,
                        unit_price=:unit_price,
                        purchase_date=:purchase_date,
                        validity_date=:validity_date,
                        notes=:notes,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=:id
                    """
                ),
                {
                    "id": int(edit_id),
                    "material_code": material_code,
                    "description": description,
                    "supplier_name": supplier_name,
                    "project_name": project_name,
                    "qty_purchased": qty,
                    "unit_price": val,
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
                SELECT id, qty_purchased, COALESCE(qty_used,0) AS qty_used
                FROM material_stock
                WHERE material_code=:material_code
                  AND COALESCE(description,'')=:description
                  AND COALESCE(supplier_name,'')=COALESCE(:supplier_name,'')
                  AND COALESCE(project_name,'')=COALESCE(:project_name,'')
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {
                "material_code": material_code,
                "description": description,
                "supplier_name": supplier_name,
                "project_name": project_name,
            },
        ).fetchone()

        if existing is not None:
            ex_id, ex_qty, ex_used = int(existing[0]), float(existing[1] or 0.0), float(existing[2] or 0.0)
            if (ex_qty - ex_used) > 0:
                # soma ao existente
                conn.execute(
                    text(
                        """
                        UPDATE material_stock
                        SET qty_purchased = COALESCE(qty_purchased,0) + :add_qty,
                            unit_price = COALESCE(unit_price,0) + :add_val,
                            purchase_date = COALESCE(:purchase_date, purchase_date),
                            validity_date = COALESCE(:validity_date, validity_date),
                            notes = CASE
                                WHEN COALESCE(notes,'') = '' THEN :notes
                                WHEN COALESCE(:notes,'') = '' THEN notes
                                ELSE notes || E'\n' || :notes
                            END,
                            updated_at = CURRENT_TIMESTAMP
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
                INSERT INTO material_stock
                    (material_code, description, supplier_name, project_name,
                     qty_purchased, unit_price, purchase_date, validity_date,
                     notes, qty_used, created_at, updated_at)
                VALUES
                    (:material_code, :description, :supplier_name, :project_name,
                     :qty_purchased, :unit_price, :purchase_date, :validity_date,
                     :notes, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                RETURNING id
                """
            ),
            {
                "material_code": material_code,
                "description": description,
                "supplier_name": supplier_name,
                "project_name": project_name,
                "qty_purchased": qty,
                "unit_price": val,
                "purchase_date": payload.get("purchase_date"),
                "validity_date": payload.get("validity_date"),
                "notes": payload.get("notes"),
            },
        ).fetchone()

        return "Material registrado.", int(new_id[0]) if new_id else None


def delete_material(engine, rid: int) -> bool:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM material_stock WHERE id=:id"), {"id": int(rid)})
    return True


def update_used_qty(engine, updates: list[dict]) -> int:
    if not updates:
        return 0
    with engine.begin() as conn:
        for u in updates:
            conn.execute(
                text(
                    """
                    UPDATE material_stock
                    SET qty_used=:qty_used,
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=:id
                    """
                ),
                {"id": int(u["id"]), "qty_used": float(u["qty_used"])},
            )
    return len(updates)


def main():
    apply_theme()

    engine, SessionLocal = get_finance_db()
    ensure_erp_tables(engine)

    login_and_guard(SessionLocal)

    st.title("📦 Estoque de Materiais")
    st.caption("Cadastro de compras, controle de utilização e saldo")

    if not _table_exists(engine, "material_stock"):
        st.error("Tabela de estoque ainda não existe. (ensure_erp_tables não criou)")
        st.stop()

    # Dados auxiliares
    df_sup = load_suppliers(engine)
    sup_opts = ["(Sem fornecedor)"] + [f"{int(r.id)} - {r.name}" for r in df_sup.itertuples(index=False)]

    df_proj = load_projects(engine)
    proj_opts = ["(Sem projeto)"] + [f"{int(r.id)} - {r.nome}" for r in df_proj.itertuples(index=False)]

    # --- Formulário ---
    st.subheader("Cadastro / Edição")

    df_stock_raw = pd.read_sql(text("SELECT id, material_code, description FROM material_stock ORDER BY id DESC"), engine)
    edit_choices = ["(Novo)"] + [f"{int(r.id)} - {r.material_code} - {r.description}" for r in df_stock_raw.itertuples(index=False)]

    sel = st.selectbox("Selecionar material para editar", edit_choices, index=0)
    edit_id = 0
    if sel != "(Novo)":
        edit_id = int(sel.split(" - ", 1)[0])

    # Carrega valores
    current = {
        "material_code": "",
        "description": "",
        "supplier_name": None,
        "project_name": None,
        "qty_purchased": 0.0,
        "unit_price": 0.0,
        "purchase_date": date.today(),
        "validity_date": None,
        "notes": "",
    }

    if edit_id > 0:
        row = pd.read_sql(
            text(
                """
                SELECT id, material_code, description, supplier_name, project_name,
                       qty_purchased, unit_price, purchase_date, validity_date, notes
                FROM material_stock
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
                    "supplier_name": r.get("supplier_name"),
                    "project_name": r.get("project_name"),
                    "qty_purchased": float(r.get("qty_purchased") or 0.0),
                    "unit_price": float(r.get("unit_price") or 0.0),
                    "purchase_date": _from_iso(r.get("purchase_date")) or date.today(),
                    "validity_date": _from_iso(r.get("validity_date")),
                    "notes": r.get("notes") or "",
                }
            )
    # Indexes selects
    def _idx_from_name(opts: list[str], target_name: str | None) -> int:
        if not target_name:
            return 0
        for i, o in enumerate(opts):
            if ' - ' in o and o.split(' - ', 1)[1].strip() == str(target_name).strip():
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
                index=_idx_from_name(sup_opts, current.get("supplier_name")),
            )
        with c4:
            project_sel = st.selectbox(
                "Projeto",
                options=proj_opts,
                index=_idx_from_name(proj_opts, current.get("project_name")),
            )

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            qty_purchased = st.number_input("Quantidade comprada *", min_value=0.0, value=float(current["qty_purchased"]), step=1.0)
        with c6:
            unit_price = st.number_input("Valor da compra", min_value=0.0, value=float(current["unit_price"]), step=50.0)
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
            supplier_name = supplier_sel.split(" - ", 1)[1] if (supplier_sel and supplier_sel != "(Sem fornecedor)" and " - " in supplier_sel) else None
            project_name = project_sel.split(" - ", 1)[1] if (project_sel and project_sel != "(Sem projeto)" and " - " in project_sel) else None

            payload = {
                "id": int(edit_id) if edit_id > 0 else None,
                "material_code": material_code,
                "description": description,
                "supplier_name": supplier_name,
                "project_name": project_name,
                "qty_purchased": float(qty_purchased),
                "unit_price": float(unit_price),
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

    # Editor: somente **Utilizado** editável
    df_editor = pd.DataFrame({
        "id": df["id"],
        "Código": df["material_code"],
        "Descrição": df["description"],
        "Fornecedor": df.get("supplier_name", ""),
        "Projeto": df.get("project_name", ""),
        "Qtd. Comprada": pd.to_numeric(df.get("qty_purchased", 0), errors="coerce").fillna(0.0),
        "Valor da Compra": pd.to_numeric(df.get("unit_price", 0), errors="coerce").fillna(0.0),
        "Data da Compra": df.get("purchase_date", ""),
        "Validade": df.get("validity_date", ""),
        "Observação": df.get("notes", ""),
        "Utilizado": pd.to_numeric(df.get("qty_used", 0), errors="coerce").fillna(0.0),
    })
    df_editor["Saldo"] = df_editor["Qtd. Comprada"] - df_editor["Utilizado"]

    disabled_cols = [c for c in df_editor.columns if c != "Utilizado"]

    edited = st.data_editor(
        df_editor,
        hide_index=True,
        width="stretch",
        disabled=disabled_cols,
        key="stock_editor",
    )

    if st.button("💾 Salvar utilizado", key="save_used"):
        # detecta alterações
        updates: list[dict] = []
        try:
            base_map = {int(r["id"]): float((r.get("qty_used") or 0) if (r.get("qty_used") is not None) else 0) for _, r in df.iterrows()}
            for _, r in edited.iterrows():
                rid = int(r["id"])
                val_u = pd.to_numeric(r["Utilizado"], errors="coerce")
                new_u = 0.0 if pd.isna(val_u) else float(val_u)
                old_u = float(base_map.get(rid, 0.0))
                if abs(new_u - old_u) > 1e-9:
                    updates.append({"id": rid, "qty_used": max(new_u, 0.0)})

            n = update_used_qty(engine, updates)
            st.success(f"Atualizado: {n} item(ns).")
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao salvar utilizado: {e}")


if __name__ == "__main__":
    main()
