from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Dict, Any, Tuple

import pandas as pd
from sqlalchemy import text


def _today_iso() -> str:
    return date.today().isoformat()


def list_projects(conn, only_open: bool = False) -> pd.DataFrame:
    where = "WHERE COALESCE(encerrado, FALSE) = FALSE" if only_open else ""
    # Campos alinhados ao app Power Apps (BaseProjetos)
    sql = f"""
        SELECT
            id,
            nome,
            cod_projeto,
            contrato,
            client_id,
            id_responsavel,
            categoria,
            subcategoria,
            dataInicio AS data_inicio,
            data_conclusao,
            status,
            progress_pct,
            atraso_responsabilidade
        FROM projects
        {where}
        ORDER BY id DESC
    """
    return pd.read_sql(text(sql), conn)


def upsert_project(conn, data: Dict[str, Any]) -> int:
    """Cria/atualiza projeto. Retorna project_id."""
    pid = data.get("id")
    fields = {
        "nome": data.get("nome"),
        "cod_projeto": data.get("cod_projeto"),
        "contrato": data.get("contrato"),
        "client_id": data.get("client_id"),
        "id_responsavel": data.get("id_responsavel"),
        "categoria": data.get("categoria"),
        "subcategoria": data.get("subcategoria"),
        "dataInicio": data.get("data_inicio"),
        "data_conclusao": data.get("data_conclusao"),
        "status": data.get("status"),
        "progress_pct": data.get("progress_pct", 0),
        "atraso_responsabilidade": data.get("atraso_responsabilidade"),
    }

    if pid:
        sets = ", ".join([f"{k} = :{k}" for k in fields.keys()])
        conn.execute(
            text(f"UPDATE projects SET {sets} WHERE id = :id"),
            {"id": pid, **fields},
        )
        return int(pid)

    cols = ", ".join(fields.keys())
    params = ", ".join([f":{k}" for k in fields.keys()])
    res = conn.execute(
        text(f"INSERT INTO projects ({cols}) VALUES ({params}) RETURNING id"),
        fields,
    )
    new_id = res.scalar()
    return int(new_id)


def list_tasks(conn, project_id: Optional[int] = None) -> pd.DataFrame:
    where = ""
    params: Dict[str, Any] = {}
    if project_id:
        where = "WHERE project_id = :pid"
        params["pid"] = project_id

    sql = f"""
        SELECT
            id,
            project_id,
            title,
            descricao,
            numero_documento,
            num_sequencial,
            disciplina,
            data_inicio,
            data_conclusao,
            status_tarefa,
            id_responsavel,
            rev_atual,
            tensao,
            atraso_responsabilidade,
            created_at,
            updated_at
        FROM project_tasks
        {where}
        ORDER BY COALESCE(data_conclusao, '') DESC, id DESC
    """
    return pd.read_sql(text(sql), conn, params=params)


def upsert_task(conn, data: Dict[str, Any]) -> int:
    tid = data.get("id")
    fields = {
        "project_id": data.get("project_id"),
        "title": data.get("title"),
        "descricao": data.get("descricao"),
        "numero_documento": data.get("numero_documento"),
        "num_sequencial": data.get("num_sequencial"),
        "disciplina": data.get("disciplina"),
        "data_inicio": data.get("data_inicio"),
        "data_conclusao": data.get("data_conclusao"),
        "status_tarefa": data.get("status_tarefa", "aberta"),
        "id_responsavel": data.get("id_responsavel"),
        "rev_atual": data.get("rev_atual"),
        "tensao": data.get("tensao"),
        "atraso_responsabilidade": data.get("atraso_responsabilidade"),
        "updated_at": datetime.utcnow(),
    }

    if tid:
        sets = ", ".join([f"{k} = :{k}" for k in fields.keys()])
        conn.execute(
            text(f"UPDATE project_tasks SET {sets} WHERE id = :id"),
            {"id": tid, **fields},
        )
        return int(tid)

    cols = ", ".join([k for k in fields.keys() if k != "updated_at"])
    params = ", ".join([f":{k}" for k in fields.keys() if k != "updated_at"])
    res = conn.execute(
        text(f"INSERT INTO project_tasks ({cols}) VALUES ({params}) RETURNING id"),
        {k: v for k, v in fields.items() if k != "updated_at"},
    )
    return int(res.scalar())


def delete_task(conn, task_id: int) -> None:
    conn.execute(text("DELETE FROM project_tasks WHERE id = :id"), {"id": task_id})


def list_team(conn, project_id: int) -> pd.DataFrame:
    sql = """
        SELECT
            m.id,
            m.project_id,
            m.user_id,
            m.funcao,
            u.name AS usuario,
            u.email AS email
        FROM project_team_members m
        LEFT JOIN users u ON u.id = m.user_id
        WHERE m.project_id = :pid
        ORDER BY m.id DESC
    """
    return pd.read_sql(text(sql), conn, params={"pid": project_id})


def upsert_team_member(conn, data: Dict[str, Any]) -> int:
    mid = data.get("id")
    fields = {"project_id": data["project_id"], "user_id": data["user_id"], "funcao": data.get("funcao")}
    if mid:
        conn.execute(
            text("UPDATE project_team_members SET project_id=:project_id, user_id=:user_id, funcao=:funcao WHERE id=:id"),
            {"id": mid, **fields},
        )
        return int(mid)

    res = conn.execute(
        text("INSERT INTO project_team_members (project_id, user_id, funcao) VALUES (:project_id, :user_id, :funcao) RETURNING id"),
        fields,
    )
    return int(res.scalar())


def delete_team_member(conn, member_id: int) -> None:
    conn.execute(text("DELETE FROM project_team_members WHERE id=:id"), {"id": member_id})


def kpis(projects: pd.DataFrame, tasks: pd.DataFrame) -> Dict[str, Any]:
    today = date.today()
    def to_date(x):
        try:
            return datetime.fromisoformat(str(x)).date()
        except Exception:
            return None

    tasks_due = tasks.copy()
    tasks_due["due"] = tasks_due["data_conclusao"].apply(to_date)
    overdue = tasks_due[(tasks_due["due"].notna()) & (tasks_due["due"] < today) & (tasks_due["status_tarefa"] != "concluida")]
    due_7 = tasks_due[(tasks_due["due"].notna()) & (tasks_due["due"] >= today) & (tasks_due["due"] <= (today.replace(day=today.day) + pd.Timedelta(days=7)).date())]

    proj_due = projects.copy()
    proj_due["due"] = proj_due["data_conclusao"].apply(to_date)
    proj_overdue = proj_due[(proj_due["due"].notna()) & (proj_due["due"] < today) & (~proj_due["status"].str.lower().isin(["concluido","encerrado","finalizado"]))]

    return {
        "projetos_total": int(len(projects)),
        "projetos_atrasados": int(len(proj_overdue)),
        "tarefas_total": int(len(tasks)),
        "tarefas_atrasadas": int(len(overdue)),
    }


def delays_table(projects: pd.DataFrame) -> pd.DataFrame:
    today = date.today()

    def delay_days(due, status):
        try:
            dd = datetime.fromisoformat(str(due)).date()
        except Exception:
            return 0
        if dd and dd < today and str(status).lower() not in ["concluido","encerrado","finalizado"]:
            return (today - dd).days
        return 0

    out = projects.copy()
    out["dias_atraso"] = out.apply(lambda r: delay_days(r.get("data_conclusao"), r.get("status")), axis=1)
    return out.sort_values(["dias_atraso", "id"], ascending=[False, False])