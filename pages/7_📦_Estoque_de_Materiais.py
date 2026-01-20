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


def _get_table_columns(engine, table_name: str) -> set[str]:
    """Retorna o conjunto de colunas existentes (Postgres ou SQLite)."""
    cols: set[str] = set()
    try:
        with engine.begin() as conn:
            rows = conn.execute(text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = :t
                """
            ), {"t": table_name}).fetchall()
        cols = {str(r[0]) for r in rows}
        if cols:
            return cols
    except Exception:
        pass

    # SQLite / fallback
    try:
        with engine.begin() as conn:
            rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
        cols = {str(r[1]) for r in rows}
    except Exception:
        cols = set()
    return cols


def ensure_material_stock_schema(engine) -> None:
    """Garante que a tabela material_stock tenha colunas esperadas (migração leve)."""
    if not _table_exists(engine, "material_stock"):
        return

    cols = _get_table_columns(engine, "material_stock")

    is_pg = (getattr(engine.dialect, "name", "") or "").startswith("postgres")
    T_INT = "integer" if is_pg else "INTEGER"
    T_NUM = "double precision" if is_pg else "REAL"
    T_TXT = "text" if is_pg else "TEXT"
    T_DATE = "date" if is_pg else "DATE"
    T_TS = "timestamp" if is_pg else "TIMESTAMP"

    wanted: list[tuple[str, str]] = [
        ("material_code", T_TXT),
        ("description", T_TXT),
        ("supplier_id", T_INT),
        ("project_id", T_INT),
        ("qty_purchased", T_NUM),
        ("qty_used", T_NUM),
        ("unit_price", T_NUM),
        ("total_price", T_NUM),
        ("purchase_date", T_DATE),
        ("validity_date", T_DATE),
        ("notes", T_TXT),
        ("supplier_name", T_TXT),
        ("project_name", T_TXT),
        ("created_at", T_TS),
        ("updated_at", T_TS),
    ]

    with engine.begin() as conn:
        for col, typ in wanted:
            if col in cols:
                continue
            try:
                if is_pg:
                    conn.execute(text(f"ALTER TABLE material_stock ADD COLUMN IF NOT EXISTS {col} {typ}"))
                else:
                    conn.execute(text(f"ALTER TABLE material_stock ADD COLUMN {col} {typ}"))
            except Exception:
                pass

        # Backfill unit_price a partir do total_price
        try:
            cols2 = _get_table_columns(engine, "material_stock")
            if ("unit_price" in cols2) and ("total_price" in cols2):
                conn.execute(text(
                    """
                    UPDATE material_stock
                    SET unit_price = CASE
                        WHEN (unit_price IS NULL OR unit_price = 0)
                             AND qty_purchased IS NOT NULL AND qty_purchased <> 0
                        THEN COALESCE(total_price, 0) / qty_purchased
                        ELSE unit_price
                    END
                    """
                ))
        except Exception:
            pass


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
    """Carrega o estoque de forma tolerante a versões diferentes do schema."""
    if not _table_exists(engine, "material_stock"):
        return pd.DataFrame()

    ensure_material_stock_schema(engine)

    cols = _get_table_columns(engine, "material_stock")
    has_supplier_id = "supplier_id" in cols
    has_project_id = "project_id" in cols
    has_supplier_name = "supplier_name" in cols
    has_project_name = "project_name" in cols

    joins: list[str] = []
    supplier_name_expr = "''"
    project_name_expr = "''"

    if has_supplier_id and _table_exists(engine, "suppliers"):
        joins.append("LEFT JOIN suppliers s ON s.id = ms.supplier_id")
        supplier_name_expr = "COALESCE(s.name, '')"
    elif has_supplier_name:
        supplier_name_expr = "COALESCE(ms.supplier_name, '')"

    if has_project_id and _table_exists(engine, "projects"):
        joins.append("LEFT JOIN projects p ON p.id = ms.project_id")
        project_name_expr = "COALESCE(p.nome, COALESCE(p.name, ''))"
    elif has_project_name:
        project_name_expr = "COALESCE(ms.project_name, '')"

    if "unit_price" in cols:
        unit_price_expr = "COALESCE(ms.unit_price, 0)"
    elif "total_price" in cols:
        unit_price_expr = """CASE
            WHEN ms.qty_purchased IS NOT NULL AND ms.qty_purchased <> 0
            THEN COALESCE(ms.total_price, 0) / ms.qty_purchased
            ELSE 0
        END"""
    else:
        unit_price_expr = "0"

    if "total_price" in cols:
        total_price_expr = "COALESCE(ms.total_price, 0)"
    elif "unit_price" in cols:
        total_price_expr = "COALESCE(ms.unit_price, 0) * COALESCE(ms.qty_purchased, 0)"
    else:
        total_price_expr = "0"

    sql = f"""
        SELECT
            ms.id,
            ms.material_code,
            ms.description,
            {supplier_name_expr} AS supplier_name,
            {project_name_expr} AS project_name,
            {('ms.supplier_id' if has_supplier_id else 'NULL')} AS supplier_id,
            {('ms.project_id' if has_project_id else 'NULL')} AS project_id,
            COALESCE(ms.qty_purchased, 0) AS qty_purchased,
            COALESCE(ms.qty_used, 0) AS qty_used,
            {unit_price_expr} AS unit_price,
            {total_price_expr} AS total_price,
            ms.purchase_date,
            ms.validity_date,
            COALESCE(ms.notes, '') AS notes
        FROM material_stock ms
        {' '.join(joins)}
        ORDER BY ms.purchase_date DESC NULLS LAST, ms.id DESC
    """

    if (getattr(engine.dialect, "name", "") or "").startswith("sqlite"):
        sql = sql.replace(" NULLS LAST", "")

    with engine.begin() as conn:
        df = pd.read_sql(text(sql), conn)

    for c in ["qty_purchased", "qty_used", "unit_price", "total_price"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["saldo"] = (df.get("qty_purchased", 0) - df.get("qty_used", 0)).astype(float)
    return df

def get_material_by_id(engine, rid: int) -> dict | None:
    """Retorna 1 material por ID (material_stock)."""
    sql = """
        SELECT
            id,
            material_code,
            description,
            supplier_id,
            project_id,
            qty_purchased,
            COALESCE(qty_used, 0) AS qty_used,
            COALESCE(unit_price, 0) AS unit_price,
            COALESCE(total_price, 0) AS total_price,
            purchase_date,
            validity_date,
            COALESCE(notes, '') AS notes
        FROM material_stock
        WHERE id = :id
    """
    df = pd.read_sql(text(sql), engine, params={"id": int(rid)})
    if df.empty:
        return None
    return df.iloc[0].to_dict()

def upsert_purchase(engine, payload: dict) -> tuple[str, int | None]:
    """Cria/atualiza item no estoque (material_stock).

    - Para consistência com `ensure_erp_tables`, persistimos `supplier_id` e `project_id`.
    - `total_price` representa o **valor total da compra**.
    - `unit_price` é calculado automaticamente quando possível (total / quantidade).
    """

    rid = int(payload.get("id") or 0)

    qty = float(payload.get("qty_purchased") or 0.0)
    total = float(payload.get("total_price") or 0.0)
    unit = float(payload.get("unit_price") or 0.0)
    if total and qty > 0:
        unit = total / qty
    elif unit and qty > 0 and not total:
        total = unit * qty

    # Normaliza FKs
    supplier_id = payload.get("supplier_id")
    project_id = payload.get("project_id")
    supplier_id = int(supplier_id) if supplier_id not in (None, "", 0) else None
    project_id = int(project_id) if project_id not in (None, "", 0) else None

    with engine.begin() as conn:
        if rid > 0:
            conn.execute(
                text(
                    """
                    UPDATE material_stock
                    SET
                        material_code = :material_code,
                        description   = :description,
                        supplier_id   = :supplier_id,
                        project_id    = :project_id,
                        qty_purchased = :qty_purchased,
                        unit_price    = :unit_price,
                        total_price   = :total_price,
                        purchase_date = :purchase_date,
                        validity_date = :validity_date,
                        notes         = :notes,
                        updated_at    = CURRENT_TIMESTAMP
                    WHERE id = :id
                    """
                ),
                {
                    "id": rid,
                    "material_code": (payload.get("material_code") or "").strip(),
                    "description": (payload.get("description") or "").strip(),
                    "supplier_id": supplier_id,
                    "project_id": project_id,
                    "qty_purchased": qty,
                    "unit_price": unit,
                    "total_price": total,
                    "purchase_date": payload.get("purchase_date"),
                    "validity_date": payload.get("validity_date"),
                    "notes": (payload.get("notes") or "").strip() or None,
                },
            )
            return ("Atualizado", rid)

        r = conn.execute(
            text(
                """
                INSERT INTO material_stock (
                    material_code, description, supplier_id, project_id,
                    qty_purchased, qty_used, unit_price, total_price,
                    purchase_date, validity_date, notes,
                    created_at, updated_at
                ) VALUES (
                    :material_code, :description, :supplier_id, :project_id,
                    :qty_purchased, 0, :unit_price, :total_price,
                    :purchase_date, :validity_date, :notes,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                RETURNING id
                """
            ),
            {
                "material_code": (payload.get("material_code") or "").strip(),
                "description": (payload.get("description") or "").strip(),
                "supplier_id": supplier_id,
                "project_id": project_id,
                "qty_purchased": qty,
                "unit_price": unit,
                "total_price": total,
                "purchase_date": payload.get("purchase_date"),
                "validity_date": payload.get("validity_date"),
                "notes": (payload.get("notes") or "").strip() or None,
            },
        )

        new_id = None
        try:
            new_id = int(r.scalar())
        except Exception:
            # SQLite pode não suportar RETURNING dependendo da versão
            try:
                new_id = int(conn.execute(text("SELECT last_insert_rowid()")) .scalar())
            except Exception:
                new_id = None

        return ("Criado", new_id)

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
    ensure_material_stock_schema(engine)

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
        "supplier_id": None,
        "project_id": None,
        "qty_purchased": 0.0,
        "total_price": 0.0,
        "purchase_date": date.today(),
        "validity_date": None,
        "notes": "",
    }

    if edit_id > 0:
        row = get_material_by_id(engine, edit_id)
        if row:
            current.update({
                "material_code": row.get("material_code", "") or "",
                "description": row.get("description", "") or "",
                "supplier_id": row.get("supplier_id"),
                "project_id": row.get("project_id"),
                "qty_purchased": float(row.get("qty_purchased") or 0.0),
                "total_price": float(row.get("total_price") or 0.0),
                "purchase_date": row.get("purchase_date") or date.today(),
                "validity_date": row.get("validity_date"),
                "notes": row.get("notes", "") or "",
            })

    # Indexes selects (strings no formato "id - nome")
    def _idx_from_id(opts: list[str], target_id) -> int:
        if target_id is None:
            return 0
        try:
            tid = int(target_id)
        except Exception:
            return 0
        for i, o in enumerate(opts):
            if ' - ' in o:
                left = o.split(' - ', 1)[0].strip()
                try:
                    if int(left) == tid:
                        return i
                except Exception:
                    continue
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
            total_price = st.number_input("Valor da compra", min_value=0.0, value=float(current.get("total_price") or 0.0), step=50.0)
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
            supplier_id = int(supplier_sel.split(" - ", 1)[0]) if (supplier_sel and supplier_sel != "(Sem fornecedor)" and " - " in supplier_sel) else None
            project_id = int(project_sel.split(" - ", 1)[0]) if (project_sel and project_sel != "(Sem projeto)" and " - " in project_sel) else None

            payload = {
                "id": int(edit_id) if edit_id > 0 else None,
                "material_code": material_code,
                "description": description,
                "supplier_id": supplier_id,
                "project_id": project_id,
                "qty_purchased": float(qty_purchased),
                "total_price": float(total_price),
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
        "Valor da Compra": pd.to_numeric(df.get("total_price", df.get("unit_price", 0)), errors="coerce").fillna(0.0),
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
