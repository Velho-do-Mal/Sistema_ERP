import streamlit as st
import streamlit.components.v1 as components  # IMPORT CORRETO PARA HTML
import psycopg2
import json
from datetime import datetime, date, timedelta
import pandas as pd
import plotly.express as px
from bk_erp_shared.theme import apply_theme
from bk_erp_shared.erp_db import ensure_erp_tables
import bk_finance
from controle_projetos_embutido import render_controle_projetos

# --------------------------------------------------------
# CONFIGURAÇÃO BÁSICA / CSS
# --------------------------------------------------------

st.set_page_config(page_title="BK_ERP - Projetos", layout="wide")


apply_theme()
ensure_erp_tables()
# --------------------------------------------------------
# FUNÇÕES GERAIS
# --------------------------------------------------------

def format_currency_br(val):
    return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def default_state():
    """
    Estado inicial padrão de um projeto.
    Corrige o NameError quando create_project(default_state(), ...) é chamado.
    """
    return {
        "tap": {
            "nome": "",
            "status": "rascunho",
            "dataInicio": date.today().strftime("%Y-%m-%d"),
            "gerente": "",
            "patrocinador": "",
            "objetivo": "",
            "escopo": "",
            "premissas": "",
            "requisitos": "",
            "alteracoesEscopo": [],
        },
        "eapTasks": [],
        "finances": [],
        "kpis": [],
        "risks": [],
        "lessons": [],
        "close": {},
        "actionPlan": [],
    }


# --------------------------------------------------------
# BANCO DE DADOS - NEON POSTGRESQL
# --------------------------------------------------------

def get_conn():
    """
    Abre conexão com o banco Neon usando a URL do secrets.toml.
    """
    import os
    db_url = (os.getenv("DATABASE_URL") or "").strip()
    if not db_url:
        # fallback opcional para streamlit secrets (quando existir)
        try:
            db_url = st.secrets["general"]["database_url"]
        except Exception:
            raise RuntimeError("Defina DATABASE_URL (Neon/Postgres) para usar o módulo de Projetos.")
    conn = psycopg2.connect(db_url)
    return conn


def init_db():
    """
    Cria a tabela de projetos no PostgreSQL, caso ainda não exista.
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id SERIAL PRIMARY KEY,
            data TEXT,
            nome TEXT,
            status TEXT,
            dataInicio TEXT,
            gerente TEXT,
            patrocinador TEXT,
            encerrado BOOLEAN DEFAULT FALSE
        );
        """
    )

    conn.commit()
    cur.close()
    conn.close()


def list_projects():
    """
    Retorna a lista de projetos.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, nome, status, dataInicio, gerente, patrocinador, encerrado
        FROM projects
        ORDER BY id DESC;
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    projetos = []
    for r in rows:
        projetos.append(
            {
                "id": r[0],
                "nome": r[1] or "",
                "status": r[2] or "",
                "dataInicio": r[3] or "",
                "gerente": r[4] or "",
                "patrocinador": r[5] or "",
                "encerrado": bool(r[6]),
            }
        )
    return projetos


def load_project_state(project_id: int):
    """
    Carrega o JSON do campo 'data' para o estado do projeto.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT data FROM projects WHERE id = %s;", (project_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row and row[0]:
        try:
            data = json.loads(row[0])
            if "actionPlan" not in data:
                data["actionPlan"] = []
            return data
        except Exception:
            return default_state()
    return default_state()


def save_project_state(project_id: int, data: dict):
    """
    Atualiza o registro do projeto com o JSON completo e
    campos principais desnormalizados.
    """
    tap = data.get("tap", {}) if isinstance(data, dict) else {}
    nome = tap.get("nome", "") or ""
    status = tap.get("status", "rascunho") or "rascunho"
    dataInicio = tap.get("dataInicio", "") or ""
    gerente = tap.get("gerente", "") or ""
    patrocinador = tap.get("patrocinador", "") or ""

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE projects
        SET data = %s,
            nome = %s,
            status = %s,
            dataInicio = %s,
            gerente = %s,
            patrocinador = %s
        WHERE id = %s;
        """,
        (json.dumps(data), nome, status, dataInicio, gerente, patrocinador, project_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def create_project(initial_data=None, meta=None) -> int:
    """
    Cria um novo projeto e retorna o ID.
    """
    if initial_data is None:
        initial_data = default_state()
    if meta is None:
        meta = {}

    tap = initial_data.get("tap", {})

    nome = meta.get("nome") or tap.get("nome") or "Novo projeto"
    status = meta.get("status") or tap.get("status") or "rascunho"
    dataInicio = meta.get("dataInicio") or tap.get("dataInicio") or ""
    gerente = meta.get("gerente") or tap.get("gerente") or ""
    patrocinador = meta.get("patrocinador") or tap.get("patrocinador") or ""

    tap["nome"] = nome
    tap["status"] = status
    tap["dataInicio"] = dataInicio
    tap["gerente"] = gerente
    tap["patrocinador"] = patrocinador
    initial_data["tap"] = tap

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO projects (data, nome, status, dataInicio, gerente, patrocinador, encerrado)
        VALUES (%s, %s, %s, %s, %s, %s, FALSE)
        RETURNING id;
        """,
        (
            json.dumps(initial_data),
            nome,
            status,
            dataInicio,
            gerente,
            patrocinador,
        ),
    )
    project_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return project_id


def close_project(project_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE projects SET encerrado = TRUE, status = %s WHERE id = %s;",
        ("encerrado", project_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def reopen_project(project_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE projects SET encerrado = FALSE WHERE id = %s;",
        (project_id,),
    )
    conn.commit()
    cur.close()
    conn.close()


def delete_project(project_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM projects WHERE id = %s;", (project_id,))
    conn.commit()
    cur.close()
    conn.close()


# --------------------------------------------------------
# CPM / GANTT / CURVA S TRABALHO
# --------------------------------------------------------



def parse_predecessores(raw) -> list[str]:
    """Parse campo de predecessoras.

    Aceita string '1, 2.1' ou lista já pronta.
    Retorna lista de códigos não vazios.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    # separa por vírgula e ponto-e-vírgula
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    return [p for p in parts if p]


def compute_is_summary(tasks: list[dict]) -> dict[str, bool]:
    """Define se uma tarefa é resumo (possui filhos) baseado no código (1.2.3)."""
    codes = [str(t.get("codigo", "")).strip() for t in tasks if str(t.get("codigo", "")).strip()]
    is_summary: dict[str, bool] = {c: False for c in codes}
    code_set = set(codes)
    for c in codes:
        prefix = c + "."
        # se existir qualquer outro código com este prefixo => é resumo
        if any((other != c and other.startswith(prefix)) for other in code_set):
            is_summary[c] = True
    return is_summary


def calcular_cpm(tasks):
    """
    Calcula ES/EF/LS/LF (em dias corridos) com base em predecessoras (padrão FS)
    e respeitando tarefas-resumo (atividades com subtarefas), similar ao MS Project:

    - Tarefas folha (sem filhos) são as que realmente "consomem" duração.
    - Tarefas-resumo têm ES=min(ES dos filhos) e EF=max(EF dos filhos).
    - Quando uma tarefa aponta como predecessora uma tarefa-resumo, considera-se o EF da resumo
      (ou seja, o último término das subtarefas).
    - Relaxação iterativa: recalcula até estabilizar (evita depender da ordem de inserção).
    """
    if not tasks:
        return tasks, 0

    tasks = [dict(t) for t in tasks]
    mapa = {t.get("codigo"): t for t in tasks if t.get("codigo")}

    # Normaliza predecessoras para lista
    for t in tasks:
        preds = t.get("predecessoras") or []
        if isinstance(preds, str):
            preds = [x.strip() for x in preds.split(",") if x.strip()]
        t["predecessoras"] = preds

    # Mapa de filhos via código WBS (1.2.3)
    children_map = {t["codigo"]: [] for t in tasks if t.get("codigo")}
    for t in tasks:
        cod = t.get("codigo")
        if not cod:
            continue
        parts = cod.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent in children_map:
                children_map[parent].append(cod)

    for t in tasks:
        cod = t.get("codigo")
        t["is_summary"] = bool(cod and children_map.get(cod))
        t["es"] = int(t.get("es") or 0)
        t["ef"] = int(t.get("ef") or 0)
        t["ls"] = 0
        t["lf"] = 0
        t["slack"] = 0

    # Inicialização: folhas começam em 0
    for t in tasks:
        dur = int(t.get("duracao") or 0)
        if t.get("is_summary"):
            t["es"], t["ef"] = 0, 0
        else:
            t["es"], t["ef"] = 0, max(0, dur)

    def _rollup_summaries():
        changed = False
        for t in tasks:
            if not t.get("is_summary"):
                continue
            cod = t.get("codigo")
            filhos = children_map.get(cod) or []
            es_vals, ef_vals = [], []
            for c in filhos:
                child = mapa.get(c)
                if not child:
                    continue
                es_vals.append(int(child.get("es") or 0))
                ef_vals.append(int(child.get("ef") or 0))
            if es_vals and ef_vals:
                new_es = min(es_vals)
                new_ef = max(ef_vals)
                if new_es != t.get("es") or new_ef != t.get("ef"):
                    t["es"], t["ef"] = new_es, new_ef
                    changed = True
        return changed

    max_iter = 2000
    for _ in range(max_iter):
        updated = False

        # 1) primeiro, atualiza resumos com base no que já existe
        if _rollup_summaries():
            updated = True

        # 2) relaxa folhas conforme predecessoras
        for t in tasks:
            if t.get("is_summary"):
                # resumo é derivada dos filhos; não recalcula por predecessoras aqui
                continue

            dur = int(t.get("duracao") or 0)
            rel = (t.get("relacao") or "FS").strip().upper()
            preds = t.get("predecessoras") or []

            new_es = 0
            if preds:
                max_start = 0
                ok = True
                for cod_p in preds:
                    p = mapa.get(cod_p)
                    if not p:
                        ok = False
                        break
                    # se predecessora é resumo, usa EF rolado
                    cand = int(p.get("ef") or 0) if rel in ("FS", "FF") else int(p.get("es") or 0)
                    if cand > max_start:
                        max_start = cand
                if ok:
                    new_es = max_start

            new_ef = new_es + max(0, dur)

            if new_es != int(t.get("es") or 0) or new_ef != int(t.get("ef") or 0):
                t["es"], t["ef"] = new_es, new_ef
                updated = True

        # 3) atualiza resumos novamente após mover folhas
        if _rollup_summaries():
            updated = True

        if not updated:
            break

    # Projeto termina no maior EF das folhas
    projeto_fim = max((int(t.get("ef") or 0) for t in tasks if not t.get("is_summary")), default=0)

    # ---------- Backward pass (folhas) ----------
    # Sucessores (considera dependências diretas)
    succ_map = {t["codigo"]: [] for t in tasks if t.get("codigo")}
    for t in tasks:
        cod = t.get("codigo")
        if not cod:
            continue
        for cod_p in (t.get("predecessoras") or []):
            if cod_p in succ_map:
                succ_map[cod_p].append(cod)

    # Ordena por ES decrescente (aproximação)
    ordem = sorted([t for t in tasks if not t.get("is_summary")], key=lambda x: int(x.get("es") or 0), reverse=True)
    for t in ordem:
        dur = int(t.get("duracao") or 0)
        succs = succ_map.get(t.get("codigo") or "") or []
        if not succs:
            t["lf"] = projeto_fim
            t["ls"] = t["lf"] - dur
        else:
            min_ls = None
            for s_cod in succs:
                s = mapa.get(s_cod)
                if not s or s.get("is_summary"):
                    continue
                if int(s.get("lf") or 0) == 0 and int(s.get("ef") or 0) > 0:
                    # ainda não calculado: inicializa
                    s_dur = int(s.get("duracao") or 0)
                    s["lf"] = int(s.get("ef") or 0)
                    s["ls"] = int(s.get("lf") or 0) - s_dur
                if min_ls is None or int(s.get("ls") or 0) < min_ls:
                    min_ls = int(s.get("ls") or 0)
            if min_ls is None:
                min_ls = projeto_fim
            t["lf"] = min_ls
            t["ls"] = t["lf"] - dur

        t["slack"] = int(t.get("ls") or 0) - int(t.get("es") or 0)

    # Resumos: slack/ls/lf derivados (simples)
    for t in tasks:
        if t.get("is_summary"):
            t["ls"] = int(t.get("es") or 0)
            t["lf"] = int(t.get("ef") or 0)
            t["slack"] = 0

    return tasks, projeto_fim


def gerar_curva_s_trabalho(tasks, data_inicio_str):
    """
    Curva S dupla: Planejado (CPM) × Realizado (ponderado por % de avanço).
    Estilo BK Planejamento Estratégico.
    """
    import plotly.graph_objects as go
    if not tasks or not data_inicio_str:
        return None

    tasks_cpm, total_dias = calcular_cpm(tasks)
    tasks_leaf = [t for t in tasks_cpm if not t.get("is_summary")]
    if total_dias <= 0:
        return None

    soma_duracoes = sum(int(t.get("duracao") or 0) for t in tasks_leaf)
    if soma_duracoes <= 0:
        return None

    # ── Planejado (CPM) ──
    dias = list(range(0, total_dias + 1))
    planejado = []
    for d in dias:
        acum = 0.0
        for t in tasks_leaf:
            dur = int(t.get("duracao") or 0)
            es  = t.get("es", 0)
            peso = dur / soma_duracoes if soma_duracoes > 0 else 0
            if d <= es:        frac = 0.0
            elif d >= es + dur: frac = 1.0
            else:               frac = (d - es) / dur
            acum += peso * frac
        planejado.append(round(acum * 100.0, 2))

    # ── Realizado (% de avanço ponderado por duração) ──
    # Para cada tarefa-folha: contribuição = peso × (% avanço / 100)
    # Distribuída linearmente ao longo do período planejado (start→finish)
    realizado = [0.0] * (total_dias + 1)
    for t in tasks_leaf:
        dur  = int(t.get("duracao") or 0)
        es   = t.get("es", 0)
        peso = dur / soma_duracoes if soma_duracoes > 0 else 0
        perc = float(t.get("percentual_avanco") or 0) / 100.0
        perc = max(0.0, min(1.0, perc))
        # distribui proporcionalmente ao longo dos dias planejados
        for d in range(total_dias + 1):
            if d <= es:         frac = 0.0
            elif d >= es + dur: frac = perc           # para no % real
            else:               frac = perc * (d - es) / dur if dur > 0 else 0.0
            realizado[d] += peso * frac * 100.0

    realizado = [round(v, 2) for v in realizado]

    # ── Figura ──
    fig = go.Figure()

    # Área planejada
    fig.add_trace(go.Scatter(
        x=dias, y=planejado,
        mode="lines",
        name="Planejado",
        line=dict(color="#3b82f6", width=2.5, dash="dash"),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.10)",
        hovertemplate="Dia %{x}<br>Planejado: %{y:.1f}%<extra></extra>",
    ))

    # Linha realizado
    fig.add_trace(go.Scatter(
        x=dias, y=realizado,
        mode="lines+markers",
        name="Realizado",
        line=dict(color="#22c55e", width=3),
        marker=dict(size=5, color="#22c55e"),
        fill="tozeroy",
        fillcolor="rgba(34,197,94,0.12)",
        hovertemplate="Dia %{x}<br>Realizado: %{y:.1f}%<extra></extra>",
    ))

    # Linha hoje — add_shape é compatível com todas as versões do plotly
    try:
        data_ini = datetime.strptime(data_inicio_str, "%Y-%m-%d").date()
        dia_hoje = (date.today() - data_ini).days
        if 0 <= dia_hoje <= total_dias:
            fig.add_shape(
                type="line",
                x0=dia_hoje, x1=dia_hoje, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="#f59e0b", width=2, dash="dot"),
            )
            fig.add_annotation(
                x=dia_hoje, y=1.02, xref="x", yref="paper",
                text="Hoje", showarrow=False,
                font=dict(color="#f59e0b", size=11),
                xanchor="left",
            )
    except Exception:
        pass

    fig.update_layout(
        title=dict(text="📈 Curva S de Trabalho — Planejado × Realizado", font=dict(size=15, color="#e2e8f0")),
        xaxis=dict(title="Dia do Projeto", gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title="Progresso (%)", range=[0, 105], gridcolor="rgba(255,255,255,0.07)"),
        template="plotly_dark",
        height=380,
        margin=dict(l=30, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        paper_bgcolor="rgba(13,27,42,0.0)",
        plot_bgcolor="rgba(13,27,42,0.0)",
    )
    return fig



def calcular_datas_eap(tasks, data_inicio_projeto=None):
    """Calcula Data de Início/Conclusão por tarefa (MS Project simplificado)."""
    if not tasks:
        return []

    from datetime import date, datetime, timedelta

    if data_inicio_projeto is None:
        starts = []
        for t in tasks:
            try:
                if t.get("data_inicio"):
                    starts.append(datetime.strptime(str(t["data_inicio"]), "%Y-%m-%d").date())
            except Exception:
                pass
        data_inicio_projeto = min(starts) if starts else date.today()

    tmap = {str(t.get("codigo")): dict(t) for t in tasks if t.get("codigo")}
    # garante flag de tarefa-resumo (quando não existe na base)
    _is_summary = compute_is_summary(list(tmap.values()))
    for _c, _t in tmap.items():
        if "is_summary" not in _t or _t.get("is_summary") is None:
            _t["is_summary"] = bool(_is_summary.get(_c, False))

    preds_map = {}
    for cod, t in tmap.items():
        preds = parse_predecessores(t.get("predecessoras", ""))
        preds_map[cod] = [p for p in preds if p in tmap]

    scheduled = {}

    def manual_start(task):
        try:
            if task.get("data_inicio"):
                return datetime.strptime(str(task["data_inicio"]), "%Y-%m-%d").date()
        except Exception:
            pass
        return data_inicio_projeto

    pending = {c for c, t in tmap.items() if not t.get("is_summary")}
    guard = 0
    while pending and guard < 10000:
        guard += 1
        progressed = False
        for cod in list(pending):
            preds = preds_map.get(cod, [])
            if any(p not in scheduled for p in preds):
                continue

            task = tmap[cod]
            rel = (task.get("relacao") or "FS").strip().upper()
            dur = int(task.get("duracao") or 0)

            min_start = data_inicio_projeto
            if preds:
                if rel == "SS":
                    min_start = max(scheduled[p]["start"] for p in preds)
                elif rel == "FF":
                    min_finish = max(scheduled[p]["finish"] for p in preds)
                    min_start = min_finish - timedelta(days=dur)
                elif rel == "SF":
                    min_finish = max(scheduled[p]["start"] for p in preds)
                    min_start = min_finish - timedelta(days=dur)
                else:  # FS
                    min_start = max(scheduled[p]["finish"] for p in preds)

            start = max(min_start, manual_start(task))
            finish = start + timedelta(days=dur)

            scheduled[cod] = {"start": start, "finish": finish}
            pending.remove(cod)
            progressed = True

        if not progressed:
            for cod in list(pending):
                task = tmap[cod]
                dur = int(task.get("duracao") or 0)
                start = manual_start(task)
                scheduled[cod] = {"start": start, "finish": start + timedelta(days=dur)}
                pending.remove(cod)

    # resumo por hierarquia de código (prefixo)
    for cod, task in tmap.items():
        if not task.get("is_summary"):
            continue
        prefix = cod + "."
        childs = [c for c in scheduled.keys() if str(c).startswith(prefix)]
        if childs:
            start = min(scheduled[c]["start"] for c in childs)
            finish = max(scheduled[c]["finish"] for c in childs)
        else:
            dur = int(task.get("duracao") or 0)
            start = manual_start(task)
            finish = start + timedelta(days=dur)
        scheduled[cod] = {"start": start, "finish": finish}

    out = []
    for t in tasks:
        cod = str(t.get("codigo"))
        rec = dict(t)
        if cod in scheduled:
            rec["data_inicio_calc"] = scheduled[cod]["start"].isoformat()
            rec["data_conclusao_calc"] = scheduled[cod]["finish"].isoformat()
        out.append(rec)
    return out


def gerar_organograma_eap(eap_tasks, nome_projeto="Projeto"):
    """
    Organograma EAP estilo PMBOK — árvore top-down com caixas e linhas.
    - Raiz do projeto no topo (nível 1)
    - Entregas principais (nível 2) na segunda fileira
    - Sub-atividades (nível 3: 2.1, 2.2...) abaixo de cada entrega
    - Nível 4 (2.1.1, 2.1.2...) abaixo das sub-atividades
    Cores por nível conforme PMBOK.
    """
    import plotly.graph_objects as go
    if not eap_tasks:
        return None

    # ── Paleta por nível (PMBOK) ────────────────────────────
    COR_CAIXA  = {1: "#1e3a5f", 2: "#00a8e8", 3: "#ffd700", 4: "#ff6b6b"}
    COR_BORDA  = {1: "#0d2137", 2: "#0077b6", 3: "#e6ac00", 4: "#cc0000"}
    COR_TEXTO  = {1: "#ffffff", 2: "#000000", 3: "#000000", 4: "#ffffff"}

    # ── Montar dicionário pai→filhos ─────────────────────────
    tasks_sorted = sorted(eap_tasks, key=lambda t: str(t.get("codigo", "")))
    cod_set = {str(t.get("codigo","")) for t in tasks_sorted}
    task_map = {str(t.get("codigo","")): t for t in tasks_sorted}

    def get_pai(cod):
        partes = cod.split(".")
        if len(partes) <= 1:
            return "__root__"
        pai = ".".join(partes[:-1])
        while pai not in cod_set and "." in pai:
            pai = ".".join(pai.split(".")[:-1])
        return pai if pai in cod_set else "__root__"

    filhos = {"__root__": []}
    for t in tasks_sorted:
        cod = str(t.get("codigo",""))
        pai = get_pai(cod)
        filhos.setdefault(pai, [])
        filhos.setdefault(cod, [])
        filhos[pai].append(cod)

    # ── Calcular posições (x,y) por nível ───────────────────
    # BFS do topo para baixo; x distribuído uniformemente por nó-folha
    # y = -nível (cresce para baixo)

    def contar_folhas(no):
        """Conta quantas folhas (sem filhos) existem abaixo de 'no'."""
        if not filhos.get(no):
            return 1
        return sum(contar_folhas(f) for f in filhos[no])

    pos = {}  # cod → (x, y)
    nivel_map = {"__root__": 0}
    for t in tasks_sorted:
        cod = str(t.get("codigo",""))
        nivel_map[cod] = int(t.get("nivel", len(cod.split("."))))

    def atribuir_posicoes(no, x_start, x_end, depth):
        cx = (x_start + x_end) / 2.0
        cy = -depth
        pos[no] = (cx, cy)
        children = filhos.get(no, [])
        if not children:
            return
        total_folhas = sum(contar_folhas(c) for c in children)
        cur = x_start
        for c in children:
            w = (contar_folhas(c) / total_folhas) * (x_end - x_start)
            atribuir_posicoes(c, cur, cur + w, depth + 1)
            cur += w

    atribuir_posicoes("__root__", 0, 1, 0)

    # ── Construir figura ─────────────────────────────────────
    fig = go.Figure()

    # Linhas de conexão pai → filho
    shapes = []
    for pai_cod, children in filhos.items():
        if pai_cod not in pos or not children:
            continue
        px, py = pos[pai_cod]
        for c in children:
            if c not in pos:
                continue
            cx, cy = pos[c]
            # Linha vertical do pai para baixo e horizontal para o filho
            # (cotovelo estilo PMBOK)
            mid_y = (py + cy) / 2.0
            shapes.append(dict(
                type="line", xref="x", yref="y",
                x0=px, y0=py - 0.06,   # sai de baixo da caixa pai
                x1=px, y1=mid_y,
                line=dict(color="#94a3b8", width=1.5),
            ))
            shapes.append(dict(
                type="line", xref="x", yref="y",
                x0=px, y0=mid_y,
                x1=cx, y1=mid_y,
                line=dict(color="#94a3b8", width=1.5),
            ))
            shapes.append(dict(
                type="line", xref="x", yref="y",
                x0=cx, y0=mid_y,
                x1=cx, y1=cy + 0.06,   # chega no topo da caixa filho
                line=dict(color="#94a3b8", width=1.5),
            ))

    # Nós (caixas)
    node_x, node_y, node_text, node_hover = [], [], [], []
    node_colors, node_borders, node_font_colors = [], [], []
    box_w, box_h = 0.06, 0.09  # largura/altura relativa das caixas

    for cod, (nx, ny) in pos.items():
        # Nó raiz: exibir com nome do projeto (nível 0)
        if cod == "__root__":
            nome_curto = (nome_projeto or "Projeto")[:28]
            node_x.append(nx)
            node_y.append(ny)
            node_text.append(f"<b>{nome_curto}</b>")
            node_hover.append(f"<b>Projeto: {nome_projeto}</b><br>Nível 0 — Raiz")
            node_colors.append("#0f2942")   # azul muito escuro
            node_borders.append("#1e3a5f")
            node_font_colors.append("#ffffff")
            continue

        t = task_map.get(cod, {})
        desc  = str(t.get("descricao",""))
        resp  = str(t.get("responsavel","") or "—")
        nivel = int(t.get("nivel", len(cod.split("."))))
        perc  = float(t.get("percentual_avanco") or 0)
        status = str(t.get("status","") or "")

        # Texto da caixa (quebra em 2 linhas)
        label_top  = cod
        label_desc = desc[:22] + ("…" if len(desc) > 22 else "")
        label_resp = f"👤 {resp[:18]}"

        node_x.append(nx)
        node_y.append(ny)
        node_text.append(f"<b>{label_top}</b><br>{label_desc}<br>{label_resp}")
        node_hover.append(
            f"<b>{cod} — {desc}</b><br>"
            f"Responsável: {resp}<br>Nível: {nivel}<br>"
            f"Status: {status}<br>% Avanço: {perc:.0f}%"
        )
        node_colors.append(COR_CAIXA.get(nivel, "#4a5568"))
        node_borders.append(COR_BORDA.get(nivel, "#2d3748"))
        node_font_colors.append(COR_TEXTO.get(nivel, "#ffffff"))

    # Scatter para os nós (marcadores grandes que simulam caixas)
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=52,
            symbol="square",
            color=node_colors,
            line=dict(color=node_borders, width=2),
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(size=8, color=node_font_colors),
        hovertext=node_hover,
        hoverinfo="text",
        showlegend=False,
    ))

    # Legenda de níveis
    for niv, cor in COR_CAIXA.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, symbol="square", color=cor),
            name=f"Nível {niv}",
            showlegend=True,
        ))

    # Calcular altura baseada no número de níveis
    max_nivel = max((int(t.get("nivel", 1)) for t in tasks_sorted), default=2)
    n_nos = len(tasks_sorted)
    altura = max(480, 160 * (max_nivel + 1))

    fig.update_layout(
        title=dict(
            text="🏗️ EAP — Estrutura Analítica do Projeto (PMBOK)",
            font=dict(size=15, color="#e2e8f0"),
        ),
        shapes=shapes,
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-(max_nivel + 0.5), 0.5]),
        template="plotly_dark",
        height=altura,
        margin=dict(l=10, r=10, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=11)),
        paper_bgcolor="rgba(13,27,42,0.0)",
        plot_bgcolor="rgba(13,27,42,0.0)",
        hovermode="closest",
    )
    return fig


def gerar_organograma_eap_html(eap_tasks, nome_projeto="Projeto"):
    """Retorna o organograma como HTML embutível no relatório completo."""
    import plotly.io as _pio
    fig = gerar_organograma_eap(eap_tasks, nome_projeto=nome_projeto)
    if fig is None:
        return "<p>EAP sem atividades suficientes para gerar organograma.</p>"
    return _pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


def gerar_gantt(tasks, data_inicio_str):
    """
    Gráfico de Gantt estilo MS Project usando px.timeline (eixo X = datas reais).
    - Tarefas ordenadas por WBS (pai acima dos filhos)
    - Barra cinza = planejado (duração total)
    - Barra colorida sobreposta = % avanço real
    - Tarefas-resumo: borda mais grossa, ícone ▶
    - Tarefas-folha: ícone ·
    - Indentação no label por profundidade WBS
    - Linha amarela = hoje
    """
    import plotly.graph_objects as go
    if not tasks or not data_inicio_str:
        return None

    tasks_sched = calcular_datas_eap(tasks, None)
    try:
        data_inicio_dt = datetime.strptime(data_inicio_str, "%Y-%m-%d").date()
    except Exception:
        return None

    # ── Cores ───────────────────────────────────────────────
    CL_PLAN_LEAF = "rgba(148,163,184,0.35)"
    CL_PLAN_SUMM = "rgba(30,58,95,0.75)"
    CL_DONE      = "rgba(34,197,94,0.88)"
    CL_PROG      = "rgba(37,99,235,0.82)"

    # ── Determina summary vs folha ───────────────────────────
    _is_summary = compute_is_summary(tasks_sched)

    # ── Ordenar por WBS (pai antes do filho) ─────────────────
    def wbs_key(t):
        return [int(p) if p.isdigit() else p
                for p in str(t.get("codigo","")).split(".")]

    tasks_ord = sorted(tasks_sched, key=wbs_key)

    # ── Montar linhas ────────────────────────────────────────
    rows_plan = []   # barra cinza planejado
    rows_real = []   # barra colorida avanço

    for t in tasks_ord:
        cod  = str(t.get("codigo",""))
        desc = str(t.get("descricao",""))
        resp = str(t.get("responsavel","") or "")
        perc = max(0.0, min(100.0, float(t.get("percentual_avanco") or 0)))
        is_summ = bool(_is_summary.get(cod, False))

        try:
            start = datetime.strptime(
                str(t.get("data_inicio_calc") or t.get("data_inicio")),
                "%Y-%m-%d").date()
        except Exception:
            start = data_inicio_dt

        try:
            finish = datetime.strptime(
                str(t.get("data_conclusao_calc") or t.get("data_conclusao")),
                "%Y-%m-%d").date()
        except Exception:
            finish = start + timedelta(days=max(int(t.get("duracao") or 1), 1))

        if finish <= start:
            finish = start + timedelta(days=1)

        # Label com indentação WBS
        depth  = len(cod.split("."))
        indent = "    " * (depth - 1)   # NBSP para indentação
        icone  = "▶ " if is_summ else "· "
        label  = f"{indent}{icone}{cod}  {desc[:40]}"

        # Barra planejada
        rows_plan.append({
            "Task":  label,
            "Start": datetime.combine(start,  datetime.min.time()),
            "Finish": datetime.combine(finish, datetime.min.time()),
            "Tipo":  "Resumo" if is_summ else "Planejado",
            "Cor":   CL_PLAN_SUMM if is_summ else CL_PLAN_LEAF,
            "Hover": (f"<b>{cod} — {desc}</b><br>"
                      f"Início: {start.strftime('%d/%m/%Y')}<br>"
                      f"Término: {finish.strftime('%d/%m/%Y')}<br>"
                      f"Duração: {(finish-start).days} dias<br>"
                      f"Resp.: {resp}<br>"
                      + ("📁 Tarefa-resumo" if is_summ else f"% Avanço: {perc:.0f}%")),
        })

        # Barra de avanço (só folhas com % > 0)
        if not is_summ and perc > 0:
            dur_dias  = max((finish - start).days, 1)
            dias_real = max(int(dur_dias * perc / 100), 1)
            finish_r  = start + timedelta(days=dias_real)
            cor_real  = CL_DONE if perc >= 100 else CL_PROG
            rows_real.append({
                "Task":  label,
                "Start": datetime.combine(start,    datetime.min.time()),
                "Finish": datetime.combine(finish_r, datetime.min.time()),
                "Tipo":  "Concluído" if perc >= 100 else "Em andamento",
                "Cor":   cor_real,
                "Perc":  perc,
                "Hover": (f"<b>{cod} — Realizado: {perc:.0f}%</b><br>"
                          f"Início: {start.strftime('%d/%m/%Y')}<br>"
                          f"Término estimado: {finish_r.strftime('%d/%m/%Y')}"),
            })

    if not rows_plan:
        return None

    # ── Ordem explícita das tarefas no eixo Y ────────────────
    ordem_y = [r["Task"] for r in rows_plan]   # mesma ordem do WBS

    # ── Criar figura com px.timeline ─────────────────────────
    import pandas as pd
    df_plan = pd.DataFrame(rows_plan)
    df_real = pd.DataFrame(rows_real) if rows_real else None

    # Figura base com barras planejadas
    fig = go.Figure()

    # Camada 1 — barras planejadas (cinza/azul resumo)
    for r in rows_plan:
        fig.add_trace(go.Bar(
            x=[(r["Finish"] - r["Start"]).total_seconds() * 1000],
            y=[r["Task"]],
            base=[r["Start"].isoformat()],
            orientation="h",
            marker_color=r["Cor"],
            marker_line_width=1.5 if "Resumo" in r["Tipo"] else 0.8,
            marker_line_color="rgba(30,58,95,0.8)" if "Resumo" in r["Tipo"] else "rgba(148,163,184,0.4)",
            hovertext=r["Hover"],
            hoverinfo="text",
            showlegend=False,
        ))

    # Camada 2 — barras de avanço (coloridas, sobrepostas)
    for r in rows_real:
        fig.add_trace(go.Bar(
            x=[(r["Finish"] - r["Start"]).total_seconds() * 1000],
            y=[r["Task"]],
            base=[r["Start"].isoformat()],
            orientation="h",
            marker_color=r["Cor"],
            marker_line_width=0,
            text=f" {r['Perc']:.0f}%" if r["Perc"] > 8 else "",
            textposition="inside",
            textfont=dict(color="#ffffff", size=10),
            hovertext=r["Hover"],
            hoverinfo="text",
            showlegend=False,
        ))

    # ── Linha Hoje ───────────────────────────────────────────
    hoje_str = datetime.combine(date.today(), datetime.min.time()).isoformat()
    fig.add_shape(
        type="line",
        x0=hoje_str, x1=hoje_str, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#f59e0b", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=hoje_str, y=1.01, xref="x", yref="paper",
        text="<b>Hoje</b>", showarrow=False,
        font=dict(color="#f59e0b", size=11), xanchor="left",
        bgcolor="rgba(13,27,42,0.75)",
        bordercolor="#f59e0b", borderwidth=1,
    )

    # ── Legenda manual ───────────────────────────────────────
    for nome, cor in [
        ("Planejado",      CL_PLAN_LEAF),
        ("Resumo/Entrega", CL_PLAN_SUMM),
        ("Em andamento",   CL_PROG),
        ("Concluído",      CL_DONE),
    ]:
        fig.add_trace(go.Bar(
            x=[None], y=[None], orientation="h",
            marker_color=cor, name=nome, showlegend=True,
        ))

    # ── Layout ───────────────────────────────────────────────
    n_rows = len(ordem_y)
    altura = max(420, 36 * n_rows + 110)

    all_starts  = [r["Start"]  for r in rows_plan]
    all_finishs = [r["Finish"] for r in rows_plan]
    x_min = (min(all_starts)  - timedelta(days=14)).isoformat()
    x_max = (max(all_finishs) + timedelta(days=14)).isoformat()

    fig.update_layout(
        title=dict(
            text="📊 Gantt — Cronograma Planejado × Realizado",
            font=dict(size=15, color="#e2e8f0"),
        ),
        barmode="overlay",
        xaxis=dict(
            type="date",
            range=[x_min, x_max],
            tickformat="%d/%b/%y",
            dtick="M1",
            tickangle=-30,
            gridcolor="rgba(255,255,255,0.07)",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(reversed(ordem_y)),   # WBS do topo para baixo
            tickfont=dict(size=10, family="monospace"),
            gridcolor="rgba(255,255,255,0.03)",
        ),
        template="plotly_dark",
        height=altura,
        margin=dict(l=10, r=20, t=55, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=10),
        ),
        paper_bgcolor="rgba(13,27,42,0.0)",
        plot_bgcolor="rgba(13,27,42,0.0)",
        bargap=0.3,
    )
    return fig
# --------------------------------------------------------
# FINANCEIRO / CURVA S FINANCEIRA
# --------------------------------------------------------

def adicionar_dias(dt: date, qtd: int) -> date:
    return dt + timedelta(days=qtd)


def expandir_recorrencia(lanc, inicio: date, fim: date):
    ocorrencias = []
    base = datetime.strptime(lanc["dataPrevista"], "%Y-%m-%d").date()
    rec = lanc.get("recorrencia", "Nenhuma")
    qtd = lanc.get("qtdRecorrencias") or lanc.get("quantidadeRecorrencias") or 0
    try:
        qtd = int(qtd)
    except Exception:
        qtd = 0

    if rec == "Diária":
        inc = 1
    elif rec == "Semanal":
        inc = 7
    elif rec == "Quinzenal":
        inc = 14
    elif rec == "Mensal":
        inc = 30
    else:
        inc = None

    if inc is None:
        if inicio <= base <= fim:
            ocorrencias.append(base)
        return ocorrencias

    d = base
    count = 0
    while d <= fim:
        if d >= inicio:
            ocorrencias.append(d)
            count += 1
            if qtd and count >= qtd:
                break
        d = adicionar_dias(d, inc)
    return ocorrencias


def gerar_curva_s_financeira(finances, inicio_str, meses):
    if not finances or not inicio_str:
        return None, None

    ano, mes = map(int, inicio_str.split("-"))
    inicio = date(ano, mes, 1)
    fim = date(
        ano if mes + meses <= 12 else ano + (mes + meses - 1) // 12,
        (mes + meses - 1) % 12 + 1,
        1,
    ) - timedelta(days=1)

    def key_mes(d: date):
        return f"{d.year}-{str(d.month).zfill(2)}"

    mapa_prev = {}
    mapa_real = {}

    cursor = inicio
    while cursor <= fim:
        k = key_mes(cursor)
        mapa_prev[k] = 0.0
        mapa_real[k] = 0.0
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)

    for l in finances:
        tipo = l["tipo"]
        try:
            valor = float(l["valor"])
        except Exception:
            valor = 0.0
        fator = 1 if tipo == "Entrada" else -1

        ocorrencias = expandir_recorrencia(l, inicio, fim)
        for d in ocorrencias:
            k = key_mes(d)
            mapa_prev[k] += fator * valor

        if l.get("realizado") and l.get("dataRealizada"):
            try:
                dr = datetime.strptime(l["dataRealizada"], "%Y-%m-%d").date()
                if inicio <= dr <= fim:
                    k = key_mes(dr)
                    mapa_real[k] += fator * valor
            except Exception:
                pass

    labels = sorted(mapa_prev.keys())
    prev_vals = [mapa_prev[k] for k in labels]
    real_vals = [mapa_real[k] for k in labels]

    prev_acum = []
    real_acum = []
    ap = 0
    ar = 0
    for p, r in zip(prev_vals, real_vals):
        ap += p
        ar += r
        prev_acum.append(ap)
        real_acum.append(ar)

    df = pd.DataFrame(
        {
            "Mês": labels,
            "Previsto (acumulado)": prev_acum,
            "Realizado (acumulado)": real_acum,
        }
    )
    fig = px.line(
        df,
        x="Mês",
        y=["Previsto (acumulado)", "Realizado (acumulado)"],
        title="Curva S Financeira - Previsto x Realizado (Acumulado)",
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=30, r=20, t=35, b=30),
    )

    return df, fig


# --------------------------------------------------------
# INICIALIZAÇÃO
# --------------------------------------------------------

init_db()

# Login compartilhado (mesmas credenciais do Financeiro)
engine, SessionLocal = bk_finance.get_db()
bk_finance.login_ui(SessionLocal)
bk_finance.require_login()


if "current_project_id" not in st.session_state:
    projetos_ini = list_projects()
    if not projetos_ini:
        pid = create_project(default_state(), {"nome": "Projeto 1"})
        st.session_state.current_project_id = pid
        st.session_state.state = load_project_state(pid)
    else:
        pid = projetos_ini[0]["id"]
        st.session_state.current_project_id = pid
        st.session_state.state = load_project_state(pid)

if "state" not in st.session_state:
    st.session_state.state = default_state()


# --------------------------------------------------------
# HEADER GLOBAL
# --------------------------------------------------------

col_title, col_info = st.columns([4, 3])
with col_title:
    st.markdown(
        "<div class='bk-card'>"
        "<div class='main-title'>Gestão de Projetos PMBOK</div>"
        "<div class='main-subtitle'>BK Engenharia e Tecnologia &mdash; TAP, EAP, Gantt, Curva S, Finanças, Qualidade, Riscos, Lições e Encerramento.</div>"
        "</div>",
        unsafe_allow_html=True
    )

with col_info:
    st.markdown(
        f"<div style='text-align:right; font-size:12px; color:#9ca3af; padding-top:6px;'>"
        f"Usuário: <strong>BK Engenharia</strong><br>Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>",
        unsafe_allow_html=True
    )

st.markdown("---")


# --------------------------------------------------------
# SIDEBAR - PROJETOS
# --------------------------------------------------------

st.sidebar.markdown("### 🔁 Projetos")

projetos = list_projects()
if not projetos:
    pid = create_project(default_state(), {"nome": "Projeto 1"})
    st.session_state.current_project_id = pid
    st.session_state.state = load_project_state(pid)
    st.rerun()

proj_labels = []
id_to_label = {}
label_to_id = {}
for p in projetos:
    status = p["status"] or "rascunho"
    status_tag = f" ({status})"
    extra = " [ENCERRADO]" if p["encerrado"] else ""
    label = f"#{p['id']} - {p['nome']}{status_tag}{extra}"
    proj_labels.append(label)
    id_to_label[p["id"]] = label
    label_to_id[label] = p["id"]

current_id = st.session_state.current_project_id
current_label = id_to_label.get(current_id, proj_labels[0])

selected_label = st.sidebar.selectbox(
    "Selecione o projeto",
    proj_labels,
    index=proj_labels.index(current_label),
)

selected_id = label_to_id[selected_label]

if selected_id != st.session_state.current_project_id:
    st.session_state.current_project_id = selected_id
    st.session_state.state = load_project_state(selected_id)
    st.rerun()

projetos = list_projects()
current_id = st.session_state.current_project_id
current_proj = next((p for p in projetos if p["id"] == current_id), projetos[0])

with st.sidebar.expander("Ações do projeto atual", expanded=True):
    st.write(f"ID: `{current_proj['id']}`")
    st.write(f"Status: `{current_proj['status'] or 'rascunho'}`")

    c1, c2 = st.columns(2)
    with c1:
        novo_nome = st.text_input("Novo nome do projeto", value=current_proj["nome"], key="rename_proj")
        if st.button("💾 Renomear"):
            st.session_state.state["tap"]["nome"] = novo_nome
            save_project_state(st.session_state.current_project_id, st.session_state.state)
            st.success("Projeto renomeado.")
            st.rerun()
    with c2:
        if current_proj["encerrado"]:
            if st.button("🔓 Reabrir"):
                reopen_project(st.session_state.current_project_id)
                st.success("Projeto reaberto.")
                st.rerun()
        else:
            if st.button("📦 Encerrar"):
                close_project(st.session_state.current_project_id)
                st.success("Projeto encerrado (arquivado).")
                st.rerun()

    st.markdown("---")

    if st.button("➕ Criar novo projeto"):
        meta = {
            "nome": f"Projeto {len(projetos) + 1}",
            "status": "rascunho",
        }
        pid = create_project(default_state(), meta)
        st.session_state.current_project_id = pid
        st.session_state.state = load_project_state(pid)
        st.success("Novo projeto criado.")
        st.rerun()

    if st.button("🗑️ Excluir este projeto"):
        proj_id = st.session_state.current_project_id
        delete_project(proj_id)
        st.session_state.pop("current_project_id", None)
        st.session_state.pop("state", None)
        st.success("Projeto excluído.")
        st.rerun()


# --------------------------------------------------------
# CARREGA ESTADO ATUAL
# --------------------------------------------------------

state = st.session_state.state

tap = state.get("tap", {})
eapTasks = state.get("eapTasks", [])
finances = state.get("finances", [])
kpis = state.get("kpis", [])
risks = state.get("risks", [])
lessons = state.get("lessons", [])
close_data = state.get("close", {})
action_plan = state.get("actionPlan", [])

for idx, t in enumerate(eapTasks):
    if "id" not in t:
        t["id"] = int(datetime.now().timestamp() * 1000) + idx


# --------------------------------------------------------
# FUNÇÃO SALVAR
# --------------------------------------------------------

def salvar_estado():
    st.session_state.state = {
        "tap": tap,
        "eapTasks": eapTasks,
        "finances": finances,
        "kpis": kpis,
        "risks": risks,
        "lessons": lessons,
        "close": close_data,
        "actionPlan": action_plan,
    }
    save_project_state(st.session_state.current_project_id, st.session_state.state)


# --------------------------------------------------------
# TABS
# --------------------------------------------------------

tabs = st.tabs(
    [
        "🏠 Home / Resumo",
        "📜 TAP & Requisitos",
        "📦 EAP / Curva S Trabalho",
        "💰 Financeiro / Curva S",
        "📊 Qualidade (KPIs)",
        "⚠️ Riscos",
        "🧠 Lições Aprendidas",
        "✅ Encerramento",
        "📑 Relatórios HTML",
        "📌 Plano de Ação",
        "🗂️ Controle de Projetos",
    ]
)

# --------------------------------------------------------
# TAB 0 - HOME
# --------------------------------------------------------

with tabs[0]:
    st.markdown("### 🏠 Visão geral do projeto")

    nome_header = tap.get("nome") or current_proj.get("nome") or "Projeto sem nome"

    st.markdown(
        f"**Projeto atual:** `#{current_proj['id']} - {nome_header}`"
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("ID do projeto", current_proj["id"])
    with col_b:
        status_home = tap.get("status") or current_proj.get("status") or "rascunho"
        st.metric("Status TAP", status_home)
    with col_c:
        st.metric("Qtde de atividades (EAP)", len(eapTasks))
    with col_d:
        st.metric("Lançamentos financeiros", len(finances))

    st.markdown("#### Dados principais")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Nome**")
        st.info(tap.get("nome") or current_proj.get("nome") or "Não definido", icon="📌")
    with c2:
        st.write("**Gerente**")
        st.info(tap.get("gerente") or current_proj.get("gerente") or "Não informado", icon="👤")
    with c3:
        st.write("**Patrocinador**")
        st.info(tap.get("patrocinador") or current_proj.get("patrocinador") or "Não informado", icon="💼")

    atrasadas = 0
    a_fazer = 0
    if eapTasks:
        a_fazer = sum(1 for t in eapTasks if t.get("status") != "concluido")
        if tap.get("dataInicio"):
            try:
                tasks_cpm, _ = calcular_cpm(eapTasks)
                data_inicio_dt = datetime.strptime(tap["dataInicio"], "%Y-%m-%d").date()
                hoje = date.today()
                for t in tasks_sched:
                    status_t = t.get("status", "nao-iniciado")
                    if status_t != "concluido":
                        ef_dia = t.get("ef", 0)
                        fim_prev = data_inicio_dt + timedelta(days=ef_dia)
                        if fim_prev < hoje:
                            atrasadas += 1
            except Exception:
                pass

    saldo_real = 0.0
    if finances:
        df_fin_home = pd.DataFrame(finances)
        if "realizado" in df_fin_home.columns:
            entradas_real = df_fin_home[
                (df_fin_home["tipo"] == "Entrada") & (df_fin_home["realizado"])
            ]["valor"].sum()
            saidas_real = df_fin_home[
                (df_fin_home["tipo"] == "Saída") & (df_fin_home["realizado"])
            ]["valor"].sum()
            try:
                saldo_real = float(entradas_real) - float(saidas_real)
            except Exception:
                saldo_real = 0.0

    st.markdown("#### Situação operacional e financeira")
    c_sit1, c_sit2, c_sit3 = st.columns(3)
    with c_sit1:
        st.metric("Atividades em atraso", atrasadas)
    with c_sit2:
        st.metric("Atividades a fazer", a_fazer)
    with c_sit3:
        st.metric("Saldo financeiro real", format_currency_br(saldo_real))

    st.markdown("#### Últimos registros")
    col_l, col_r = st.columns(2)
    with col_l:
        st.write("**Últimas alterações de escopo**")
        alt = tap.get("alteracoesEscopo") or []
        if alt:
            df_alt = pd.DataFrame(alt)
            st.dataframe(df_alt.tail(5), use_container_width=True, height=160)
        else:
            st.caption("Nenhuma alteração registrada.")
    with col_r:
        st.write("**Últimos riscos**")
        if risks:
            df_r = pd.DataFrame(risks)
            st.dataframe(
                df_r[["descricao", "impacto", "prob", "indice"]].tail(5),
                use_container_width=True,
                height=160,
            )
        else:
            st.caption("Nenhum risco registrado.")


# --------------------------------------------------------
# TAB 1 - TAP
# --------------------------------------------------------

with tabs[1]:
    st.markdown("### 📜 Termo de Abertura do Projeto (TAP)")

    c1, c2 = st.columns(2)
    with c1:
        tap["nome"] = st.text_input("Nome do projeto", value=tap.get("nome", ""))
        data_inicio = tap.get("dataInicio") or ""
        tap["dataInicio"] = st.date_input(
            "Data de início",
            value=datetime.strptime(data_inicio, "%Y-%m-%d").date()
            if data_inicio
            else date.today(),
        ).strftime("%Y-%m-%d")
        tap["gerente"] = st.text_input("Gerente do projeto", value=tap.get("gerente", ""))
        tap["patrocinador"] = st.text_input("Patrocinador", value=tap.get("patrocinador", ""))

    with c2:
        status_opcoes = ["rascunho", "em_aprovacao", "aprovado", "encerrado"]
        status_atual = tap.get("status", "rascunho")
        if status_atual not in status_opcoes:
            status_atual = "rascunho"
        tap["status"] = st.selectbox(
            "Status do TAP",
            status_opcoes,
            index=status_opcoes.index(status_atual),
        )
        tap["objetivo"] = st.text_area(
            "Objetivo do projeto",
            value=tap.get("objetivo", ""),
            height=90,
        )
        tap["escopo"] = st.text_area(
            "Escopo inicial",
            value=tap.get("escopo", ""),
            height=90,
        )
        tap["premissas"] = st.text_area(
            "Premissas e restrições",
            value=tap.get("premissas", ""),
            height=90,
        )

    st.markdown("#### Requisitos e alterações de escopo")

    col_req, col_alt = st.columns([1, 1.2])
    with col_req:
        tap["requisitos"] = st.text_area(
            "Requisitos principais",
            value=tap.get("requisitos", ""),
            height=150,
        )

    with col_alt:
        nova_alt = st.text_area("Nova alteração de escopo", "", height=100)
        c_al1, c_al2 = st.columns(2)
        with c_al1:
            if st.button("Registrar alteração"):
                if not nova_alt.strip():
                    st.warning("Descreva a alteração antes de registrar.")
                else:
                    item = {
                        "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
                        "descricao": nova_alt.strip(),
                    }
                    tap.setdefault("alteracoesEscopo", []).append(item)
                    salvar_estado()
                    st.success("Alteração registrada.")
                    st.rerun()
        with c_al2:
            if st.button("Aprovar alteração de escopo"):
                if not tap.get("alteracoesEscopo"):
                    st.warning("Não há alterações registradas.")
                else:
                    st.info(
                        "Lembre-se de atualizar EAP, cronograma, financeiro e riscos."
                    )

        st.write("**Histórico de alterações**")
        alt = tap.get("alteracoesEscopo") or []
        if alt:
            df_alt = pd.DataFrame(alt)
            st.dataframe(df_alt, use_container_width=True, height=180)

            idx_alt = st.selectbox(
                "Selecione uma alteração para editar / excluir",
                options=list(range(len(alt))),
                format_func=lambda i: f"{df_alt.iloc[i]['data']} - {df_alt.iloc[i]['descricao'][:60]}",
                key="tap_del_alt_idx"
            )
            # --------- EDIÇÃO DE ALTERAÇÃO DE ESCOPO ---------
            alt_sel = tap["alteracoesEscopo"][idx_alt]
            nova_desc_alt_edit = st.text_area(
                "Editar descrição da alteração selecionada",
                value=alt_sel.get("descricao", ""),
                height=100,
                key="tap_alt_edit_desc"
            )
            if st.button("Salvar alteração de escopo editada"):
                tap["alteracoesEscopo"][idx_alt]["descricao"] = nova_desc_alt_edit.strip()
                salvar_estado()
                st.success("Alteração de escopo atualizada.")
                st.rerun()
            # --------- EXCLUSÃO ---------
            if st.button("Excluir alteração selecionada"):
                tap["alteracoesEscopo"].pop(idx_alt)
                salvar_estado()
                st.success("Alteração excluída.")
                st.rerun()
        else:
            st.caption("Nenhuma alteração registrada.")

    if st.button("💾 Salvar TAP", type="primary"):
        salvar_estado()
        st.success("TAP salvo e persistido no banco.")


# --------------------------------------------------------
# TAB 2 - EAP / CURVA S TRABALHO
# --------------------------------------------------------

with tabs[2]:
    st.markdown("### 📦 Estrutura Analítica do Projeto (EAP)")

    # ── Campo: Data término real do projeto ──────────────────
    _dt_termino_real_str = tap.get("dataTerminoReal", "")
    _dt_termino_real_val = None
    if _dt_termino_real_str:
        try:
            _dt_termino_real_val = datetime.strptime(_dt_termino_real_str, "%Y-%m-%d").date()
        except Exception:
            _dt_termino_real_val = None

    _col_term1, _col_term2, _col_term3 = st.columns([1, 1, 3])
    with _col_term1:
        _dt_termino_real_input = st.date_input(
            "📅 Data de Término Real do Projeto",
            value=_dt_termino_real_val,
            key="eap_termino_real",
            help="Preencha quando o projeto for encerrado de fato. Afeta indicadores de prazo.",
        )
    with _col_term2:
        if st.button("💾 Salvar data de término real", key="btn_salvar_termino"):
            tap["dataTerminoReal"] = _dt_termino_real_input.isoformat() if _dt_termino_real_input else ""
            salvar_estado()
            st.success("Data de término real salva.")

    # Indicador de desvio de prazo
    if _dt_termino_real_val and tap.get("dataInicio"):
        try:
            _di = datetime.strptime(tap["dataInicio"], "%Y-%m-%d").date()
            _duracao_real = (_dt_termino_real_val - _di).days
            _duracao_plan = sum(int(t.get("duracao") or 0) for t in eapTasks if not t.get("is_summary"))
            _desvio = _duracao_real - _duracao_plan
            _cor = "normal" if _desvio <= 0 else "inverse"
            st.metric("⏱️ Desvio de prazo (real vs. planejado)", f"{_desvio:+d} dias", delta_color=_cor)
        except Exception:
            pass

    st.divider()

    # ── Cadastrar nova atividade ─────────────────────────────
    with st.expander("➕ Cadastrar nova atividade na EAP", expanded=False):
        c1, c2, c3, c4 = st.columns([1, 2, 1, 1])
        with c1:
            codigo = st.text_input("Código (1.2.3)", key="eap_codigo")
            nivel = st.selectbox("Nível", [1, 2, 3, 4], index=0, key="eap_nivel")
        with c2:
            descricao = st.text_input("Descrição da atividade", key="eap_descricao")
        with c3:
            duracao = st.number_input("Duração (dias)", min_value=1, value=1, key="eap_dur")
        with c4:
            responsavel = st.text_input("Responsável", key="eap_resp")

        col_dt1, col_dt2 = st.columns([1.6, 1.6])
        data_inicio_new = col_dt1.date_input("Data de Início", value=date.today(), key="eap_data_inicio")
        try:
            _dur = int(duracao or 0)
        except Exception:
            _dur = 0
        data_conclusao_new = data_inicio_new + timedelta(days=_dur)
        col_dt2.text_input(
            "Data de Conclusão (automática)",
            value=data_conclusao_new.strftime("%Y-%m-%d"),
            disabled=True, key="eap_data_conclusao",
            help="Calculada como Data de Início + Duração (dias).",
        )

        col_pp, col_rel, col_stat = st.columns([2, 1, 1])
        with col_pp:
            predecessoras_str = st.text_input("Predecessoras (ex: 1.1, 1.2)", key="eap_pred")
        with col_rel:
            relacao = st.selectbox("Relação", ["FS", "FF", "SS", "SF"], index=0, key="eap_rel")
        with col_stat:
            status = st.selectbox(
                "Status",
                ["nao-iniciado", "em-andamento", "em-analise", "em-revisao", "concluido"],
                index=0, key="eap_status",
            )

        if st.button("➕ Incluir atividade EAP", type="primary", key="eap_add_btn"):
            if not codigo.strip() or not descricao.strip():
                st.warning("Informe código e descrição.")
            else:
                preds = [x.strip() for x in predecessoras_str.split(",") if x.strip()]
                eapTasks.append({
                    "id": int(datetime.now().timestamp() * 1000),
                    "codigo": codigo.strip(),
                    "descricao": descricao.strip(),
                    "nivel": int(nivel),
                    "predecessoras": preds,
                    "responsavel": responsavel.strip(),
                    "duracao": int(duracao),
                    "data_inicio": data_inicio_new.isoformat(),
                    "data_conclusao": data_conclusao_new.isoformat(),
                    "relacao": relacao,
                    "status": status,
                    "percentual_avanco": 0,
                })
                salvar_estado()
                st.success("Atividade adicionada.")
                st.rerun()

    # ── Tabela editável (inline — sem formulário de edição) ──
    if eapTasks:
        st.markdown("#### 📋 Tabela de Atividades — edite diretamente e clique em Salvar")

        # Garante campo percentual_avanco em tarefas antigas
        for _t in eapTasks:
            if "percentual_avanco" not in _t:
                _t["percentual_avanco"] = 0
            # auto-calcula % quando status = concluido
            if str(_t.get("status","")) == "concluido" and float(_t.get("percentual_avanco",0)) < 100:
                _t["percentual_avanco"] = 100

        tasks_for_table = calcular_datas_eap(eapTasks, None)
        df_eap = pd.DataFrame(tasks_for_table)
        if "data_inicio_calc" in df_eap.columns:
            df_eap["data_inicio"]    = df_eap["data_inicio_calc"]
            df_eap["data_conclusao"] = df_eap["data_conclusao_calc"]
            df_eap.drop(columns=[c for c in ["data_inicio_calc","data_conclusao_calc"] if c in df_eap.columns], inplace=True)

        df_eap_sorted = df_eap.sort_values(by="codigo").reset_index(drop=True)

        # Selecionar colunas para exibição na tabela editável
        _cols_edit = ["id","codigo","descricao","nivel","duracao","data_inicio","data_conclusao",
                      "responsavel","predecessoras","relacao","status","percentual_avanco"]
        _cols_exist = [c for c in _cols_edit if c in df_eap_sorted.columns]
        df_edit = df_eap_sorted[_cols_exist].copy()

        # predecessoras como string para edição
        if "predecessoras" in df_edit.columns:
            df_edit["predecessoras"] = df_edit["predecessoras"].apply(
                lambda v: ", ".join(v) if isinstance(v, list) else str(v or "")
            )

        # Converter datas para tipo date (data_editor precisa)
        for _dc in ["data_inicio", "data_conclusao"]:
            if _dc in df_edit.columns:
                df_edit[_dc] = pd.to_datetime(df_edit[_dc], errors="coerce").dt.date

        _status_opts = ["nao-iniciado", "em-andamento", "em-analise", "em-revisao", "concluido"]
        _relacao_opts = ["FS", "FF", "SS", "SF"]

        edited_df = st.data_editor(
            df_edit,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="eap_editor_inline",
            column_config={
                "id":          st.column_config.NumberColumn("ID", disabled=True, width="small"),
                "codigo":      st.column_config.TextColumn("Código", width="small"),
                "descricao":   st.column_config.TextColumn("Descrição", width="large"),
                "nivel":       st.column_config.SelectboxColumn("Nível", options=[1,2,3,4], width="small"),
                "duracao":     st.column_config.NumberColumn("Duração (dias)", min_value=1, step=1, width="small"),
                "data_inicio": st.column_config.DateColumn("Início", format="DD/MM/YYYY", width="small"),
                "data_conclusao": st.column_config.DateColumn("Conclusão", format="DD/MM/YYYY", width="small"),
                "responsavel": st.column_config.TextColumn("Responsável", width="medium"),
                "predecessoras": st.column_config.TextColumn("Predecessoras", width="small",
                                                              help="Códigos separados por vírgula. Ex: 1.1, 1.2"),
                "relacao":     st.column_config.SelectboxColumn("Relação", options=_relacao_opts, width="small"),
                "status":      st.column_config.SelectboxColumn("Status", options=_status_opts, width="medium"),
                "percentual_avanco": st.column_config.NumberColumn(
                    "% Avanço", min_value=0, max_value=100, step=5,
                    format="%d%%", width="small",
                    help="0 a 100. Atualiza a Curva S e o Gantt automaticamente.",
                ),
            },
        )

        _sb1, _sb2, _sb3 = st.columns([1, 1, 4])
        with _sb1:
            if st.button("💾 Salvar alterações da tabela", type="primary", key="eap_save_table"):
                # Aplica edições de volta para eapTasks
                _id_map = {int(t["id"]): t for t in eapTasks}
                for _, row in edited_df.iterrows():
                    _tid = int(row["id"])
                    if _tid in _id_map:
                        _t = _id_map[_tid]
                        _t["codigo"]    = str(row.get("codigo") or _t["codigo"]).strip()
                        _t["descricao"] = str(row.get("descricao") or _t["descricao"]).strip()
                        _t["nivel"]     = int(row.get("nivel") or _t.get("nivel", 1))
                        _t["duracao"]   = int(row.get("duracao") or _t.get("duracao", 1))
                        _t["responsavel"] = str(row.get("responsavel") or "").strip()
                        _t["relacao"]   = str(row.get("relacao") or "FS")
                        _t["status"]    = str(row.get("status") or "nao-iniciado")
                        _t["percentual_avanco"] = max(0, min(100, int(row.get("percentual_avanco") or 0)))
                        # auto-100% ao marcar concluido
                        if _t["status"] == "concluido":
                            _t["percentual_avanco"] = 100
                        # data início
                        _di = row.get("data_inicio")
                        if _di is not None:
                            try:
                                _t["data_inicio"] = _di.isoformat() if hasattr(_di, "isoformat") else str(_di)
                            except Exception:
                                pass
                        # predecessoras
                        _pred_raw = str(row.get("predecessoras") or "")
                        _t["predecessoras"] = [x.strip() for x in _pred_raw.split(",") if x.strip()]
                salvar_estado()
                st.success("✅ Tabela salva com sucesso.")
                st.rerun()

        with _sb2:
            # Excluir — via selectbox separado abaixo da tabela
            pass

        # Exclusão fora da tabela (botão vermelho)
        with st.expander("🗑️ Excluir atividade", expanded=False):
            _del_opts = {f"{t.get('codigo')} — {str(t.get('descricao',''))[:50]}": t["id"] for t in eapTasks}
            if _del_opts:
                _del_sel = st.selectbox("Selecione para excluir", list(_del_opts.keys()), key="eap_del_sel2")
                if st.button("Excluir atividade selecionada", type="secondary", key="eap_del_btn2"):
                    _del_id = _del_opts[_del_sel]
                    eapTasks[:] = [t for t in eapTasks if t.get("id") != _del_id]
                    salvar_estado()
                    st.success("Atividade excluída.")
                    st.rerun()
    else:
        st.info("Nenhuma atividade cadastrada na EAP ainda. Use o expander acima para adicionar.")

    # ── Gráficos ─────────────────────────────────────────────
    if eapTasks and tap.get("dataInicio"):
        st.divider()

        # Métricas de avanço geral
        _tasks_leaf = [t for t in eapTasks if not t.get("is_summary")]
        if _tasks_leaf:
            _soma_dur = sum(int(t.get("duracao") or 0) for t in _tasks_leaf) or 1
            _avanço_ponderado = sum(
                (int(t.get("duracao") or 0) / _soma_dur) * float(t.get("percentual_avanco") or 0)
                for t in _tasks_leaf
            )
            _concluidas = sum(1 for t in _tasks_leaf if float(t.get("percentual_avanco") or 0) >= 100)
            _em_andamento = sum(1 for t in _tasks_leaf if 0 < float(t.get("percentual_avanco") or 0) < 100)
            _nao_iniciadas = len(_tasks_leaf) - _concluidas - _em_andamento

            _m1, _m2, _m3, _m4 = st.columns(4)
            _m1.metric("📊 Avanço Geral", f"{_avanço_ponderado:.1f}%")
            _m2.metric("✅ Concluídas", _concluidas)
            _m3.metric("🔄 Em andamento", _em_andamento)
            _m4.metric("⏳ Não iniciadas", _nao_iniciadas)

        # ── Organograma EAP (sempre visível se há tarefas) ──────────
        st.markdown("#### 🏗️ Organograma da EAP — Estrutura Hierárquica")
        st.caption("Visualização em árvore top-down conforme PMBOK. Nó raiz = nome do projeto.")
        _nome_proj = tap.get("nome") or "Projeto"
        _fig_org = gerar_organograma_eap(eapTasks, nome_projeto=_nome_proj)
        if _fig_org:
            st.plotly_chart(_fig_org, use_container_width=True, key="eap_organograma_main")
        else:
            st.info("Adicione pelo menos 2 atividades para gerar o organograma.")

        st.markdown("#### 📈 Curva S de Trabalho — Planejado × Realizado")
        fig_s = gerar_curva_s_trabalho(eapTasks, tap["dataInicio"])
        if fig_s:
            st.plotly_chart(fig_s, use_container_width=True, key="curva_s_trabalho_main")
        else:
            st.warning("Não foi possível gerar a Curva S.")

        st.markdown("#### 📊 Gráfico de Gantt — Planejado × Realizado")
        fig_gantt = gerar_gantt(eapTasks, tap["dataInicio"])
        if fig_gantt:
            st.plotly_chart(fig_gantt, use_container_width=True, key="gantt_main")
        else:
            st.caption("Gantt indisponível — verifique dados da EAP e data de início no TAP.")
    elif eapTasks:
        # Mesmo sem data de início, mostra o organograma
        st.divider()
        st.markdown("#### 🏗️ Organograma da EAP — Estrutura Hierárquica")
        _fig_org2 = gerar_organograma_eap(eapTasks, nome_projeto=tap.get("nome") or "Projeto")
        if _fig_org2:
            st.plotly_chart(_fig_org2, use_container_width=True, key="eap_organograma_main2")
        st.warning("Defina a data de início no TAP para gerar Curva S e Gantt.")


# --------------------------------------------------------
# TAB 3 - FINANCEIRO / CURVA S
# --------------------------------------------------------

with tabs[3]:
    st.markdown("### 💰 Lançamentos financeiros do projeto")

    with st.expander("Adicionar lançamento financeiro", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            tipo = st.selectbox("Tipo", ["Entrada", "Saída"], index=0, key="fin_tipo")
            categoria = st.selectbox(
                "Categoria (somente para Saída)",
                ["", "Mão de Obra", "Custos Diretos", "Impostos"],
                index=0,
                key="fin_categoria",
            )
        with c2:
            descricao = st.text_input("Descrição", key="fin_desc")
            subcategoria = st.text_input("Subcategoria", key="fin_sub")
        with c3:
            valor = st.number_input(
                "Valor (R$)", min_value=0.0, step=100.0, key="fin_val"
            )
            recorrencia = st.selectbox(
                "Recorrência",
                ["Nenhuma", "Diária", "Semanal", "Quinzenal", "Mensal"],
                index=0,
                key="fin_rec",
            )

        c4, c5, c6 = st.columns(3)
        with c4:
            data_prevista = st.date_input("Data prevista", key="fin_data_prev")
        with c5:
            realizado = st.checkbox("Realizado?", key="fin_realizado")
        with c6:
            data_realizada = st.date_input(
                "Data realizada", key="fin_data_real", value=date.today()
            )

        c7, _, _ = st.columns(3)
        with c7:
            qtd_recorrencias = st.number_input(
                "Quantidade de recorrências",
                min_value=1,
                value=1,
                key="fin_qtd_rec",
            )

        if st.button("Adicionar lançamento", type="primary"):
            if not descricao.strip() or valor <= 0:
                st.warning("Informe descrição e valor maior que zero.")
            else:
                if tipo == "Saída" and not categoria:
                    st.warning("Selecione a categoria para Saída.")
                else:
                    lanc = {
                        "id": int(datetime.now().timestamp() * 1000),
                        "tipo": tipo,
                        "descricao": descricao.strip(),
                        "categoria": categoria if tipo == "Saída" else "",
                        "subcategoria": subcategoria.strip(),
                        "valor": float(valor),
                        "recorrencia": recorrencia,
                        "qtdRecorrencias": int(qtd_recorrencias) if recorrencia != "Nenhuma" else 1,
                        "dataPrevista": data_prevista.strftime("%Y-%m-%d"),
                        "realizado": bool(realizado),
                        "dataRealizada": data_realizada.strftime("%Y-%m-%d")
                        if realizado
                        else "",
                    }
                    finances.append(lanc)
                    salvar_estado()
                    st.success("Lançamento adicionado.")
                    st.rerun()

    if finances:
        st.markdown("#### Extrato financeiro detalhado")

        df_fin_base = pd.DataFrame(finances)

        if "qtdRecorrencias" not in df_fin_base.columns:
            df_fin_base["qtdRecorrencias"] = 1

        df_fin_base["qtdRecorrencias"] = df_fin_base["qtdRecorrencias"].fillna(1)

        if "recorrencia" not in df_fin_base.columns:
            df_fin_base["recorrencia"] = "Nenhuma"

        linhas = []
        for _, row in df_fin_base.iterrows():
            rec = row.get("recorrencia", "Nenhuma")

            qtd_raw = row.get("qtdRecorrencias", 1)
            if pd.isna(qtd_raw):
                qtd = 1
            else:
                try:
                    qtd = int(qtd_raw)
                except Exception:
                    qtd = 1

            data_base = datetime.strptime(row["dataPrevista"], "%Y-%m-%d").date()

            if rec == "Diária":
                inc = 1
            elif rec == "Semanal":
                inc = 7
            elif rec == "Quinzenal":
                inc = 14
            elif rec == "Mensal":
                inc = 30
            else:
                inc = 0

            if rec == "Nenhuma" or qtd <= 1 or inc == 0:
                new_row = row.copy()
                new_row["Prevista"] = row["dataPrevista"]
                new_row["Parcela"] = ""
                linhas.append(new_row)
            else:
                for i in range(qtd):
                    new_row = row.copy()
                    data_parcela = adicionar_dias(data_base, inc * i)
                    new_row["Prevista"] = data_parcela.strftime("%Y-%m-%d")
                    new_row["Parcela"] = f"{i+1}/{qtd}"
                    linhas.append(new_row)

        df_fin_display = pd.DataFrame(linhas)

        # Garantir que há uma coluna numérica para somas (evita problemas de tipo)
        df_fin_display["valor_num"] = pd.to_numeric(df_fin_display["valor"], errors="coerce").fillna(0.0)

        df_fin_display["Valor (R$)"] = df_fin_display["valor_num"].map(
            lambda x: format_currency_br(x)
        )
        df_fin_display["Realizada"] = df_fin_display["dataRealizada"].replace("", "-")
        df_fin_display["Status"] = df_fin_display["realizado"].map(
            lambda x: "Realizado" if x else "Pendente"
        )
        df_fin_display["Recorrência"] = df_fin_display["recorrencia"]
        df_fin_display["Qtd. rec."] = df_fin_display["qtdRecorrencias"].fillna(1).astype(int)

        cols_show = [
            "tipo",
            "descricao",
            "categoria",
            "subcategoria",
            "Valor (R$)",
            "Prevista",
            "Realizada",
            "Status",
            "Recorrência",
            "Qtd. rec.",
            "Parcela",
        ]
        st.dataframe(
            df_fin_display[cols_show], use_container_width=True, height=260
        )

        idx_fin = st.selectbox(
            "Selecione o lançamento para editar / excluir",
            options=list(range(len(df_fin_display))),
            format_func=lambda i: f"{df_fin_display.iloc[i]['tipo']} - {df_fin_display.iloc[i]['descricao'][:50]} - {df_fin_display.iloc[i]['Valor (R$)']} - Prevista {df_fin_display.iloc[i]['Prevista']}",
            key="fin_del_idx"
        )

        # --------- EDIÇÃO DE LANÇAMENTO FINANCEIRO ---------
        sel_id = df_fin_display.iloc[idx_fin]["id"]
        lanc_sel = next((l for l in finances if l["id"] == sel_id), None)

        if lanc_sel:
            # --- SINCRONIZA O FORMULÁRIO QUANDO MUDA O LANÇAMENTO SELECIONADO ---
            if st.session_state.get("fin_last_sel_id") != sel_id:
                st.session_state["fin_last_sel_id"] = sel_id

                # Campos simples
                st.session_state[f"fin_tipo_edit_{sel_id}"] = lanc_sel.get("tipo", "Entrada")
                st.session_state[f"fin_categoria_edit_{sel_id}"] = lanc_sel.get("categoria", "")
                st.session_state[f"fin_desc_edit_{sel_id}"] = lanc_sel.get("descricao", "")
                st.session_state[f"fin_sub_edit_{sel_id}"] = lanc_sel.get("subcategoria", "")
                st.session_state[f"fin_val_edit_{sel_id}"] = float(lanc_sel.get("valor", 0.0))
                st.session_state[f"fin_rec_edit_{sel_id}"] = lanc_sel.get("recorrencia", "Nenhuma")

                # Datas
                dp_str = lanc_sel.get("dataPrevista") or date.today().strftime("%Y-%m-%d")
                try:
                    dp_dt = datetime.strptime(dp_str, "%Y-%m-%d").date()
                except Exception:
                    dp_dt = date.today()
                st.session_state[f"fin_data_prev_edit_{sel_id}"] = dp_dt

                dr_str = lanc_sel.get("dataRealizada") or date.today().strftime("%Y-%m-%d")
                try:
                    dr_dt = datetime.strptime(dr_str, "%Y-%m-%d").date()
                except Exception:
                    dr_dt = date.today()
                st.session_state[f"fin_data_real_edit_{sel_id}"] = dr_dt

                # Realizado / recorrências
                st.session_state[f"fin_realizado_edit_{sel_id}"] = bool(lanc_sel.get("realizado"))
                try:
                    qtd_base_int = int(lanc_sel.get("qtdRecorrencias", 1))
                except Exception:
                    qtd_base_int = 1
                st.session_state[f"fin_qtd_rec_edit_{sel_id}"] = qtd_base_int

            # ----------------- FORMULÁRIO DE EDIÇÃO -----------------
            st.markdown("#### Editar lançamento selecionado")
            fe1, fe2, fe3 = st.columns(3)
            with fe1:
                tipo_opts = ["Entrada", "Saída"]
                tipo_edit = st.selectbox(
                    "Tipo (edição)",
                    tipo_opts,
                    key=f"fin_tipo_edit_{sel_id}",  # valor vem de session_state
                )

                cat_opts = ["", "Mão de Obra", "Custos Diretos", "Impostos"]
                categoria_edit = st.selectbox(
                    "Categoria (edição - somente Saída)",
                    cat_opts,
                    key=f"fin_categoria_edit_{sel_id}",
                )

            with fe2:
                desc_edit = st.text_input(
                    "Descrição (edição)",
                    key=f"fin_desc_edit_{sel_id}",
                )
                sub_edit = st.text_input(
                    "Subcategoria (edição)",
                    key=f"fin_sub_edit_{sel_id}",
                )

            with fe3:
                valor_edit = st.number_input(
                    "Valor (R$) - edição",
                    min_value=0.0,
                    step=100.0,
                    key=f"fin_val_edit_{sel_id}",
                )
                rec_opts = ["Nenhuma", "Diária", "Semanal", "Quinzenal", "Mensal"]
                recorrencia_edit = st.selectbox(
                    "Recorrência (edição)",
                    rec_opts,
                    key=f"fin_rec_edit_{sel_id}",
                )

            fe4, fe5, fe6 = st.columns(3)
            with fe4:
                data_prevista_edit = st.date_input(
                    "Data prevista (edição)",
                    key=f"fin_data_prev_edit_{sel_id}",
                )

            with fe5:
                realizado_edit = st.checkbox(
                    "Realizado? (edição)",
                    key=f"fin_realizado_edit_{sel_id}",
                )

            with fe6:
                data_realizada_edit = st.date_input(
                    "Data realizada (edição)",
                    key=f"fin_data_real_edit_{sel_id}",
                )

            fe7, _, _ = st.columns(3)
            with fe7:
                qtd_rec_edit = st.number_input(
                    "Quantidade de recorrências (edição)",
                    min_value=1,
                    key=f"fin_qtd_rec_edit_{sel_id}",
                )

            # BOTÃO DE SALVAR
            if st.button("Salvar alterações do lançamento selecionado", key=f"fin_edit_save_{sel_id}"):
                for l in finances:
                    if l["id"] == sel_id:
                        l["tipo"] = tipo_edit
                        l["descricao"] = desc_edit.strip()
                        l["categoria"] = categoria_edit if tipo_edit == "Saída" else ""
                        l["subcategoria"] = sub_edit.strip()
                        l["valor"] = float(valor_edit)
                        l["recorrencia"] = recorrencia_edit
                        l["qtdRecorrencias"] = int(qtd_rec_edit) if recorrencia_edit != "Nenhuma" else 1
                        l["dataPrevista"] = data_prevista_edit.strftime("%Y-%m-%d")
                        l["realizado"] = bool(realizado_edit)
                        l["dataRealizada"] = (
                            data_realizada_edit.strftime("%Y-%m-%d")
                            if realizado_edit
                            else ""
                        )
                        break
                salvar_estado()
                st.success("Lançamento atualizado.")
                st.rerun()

        # --------- EXCLUSÃO ---------
        if st.button("Excluir lançamento selecionado", key="fin_del_btn"):
            finances[:] = [l for l in finances if l["id"] != sel_id]
            salvar_estado()
            st.success("Lançamento excluído.")
            st.rerun()

        # <-- Verificação/Soma de entradas e saídas (garantida pela coluna valor_num)
        total_entradas = float(df_fin_display[df_fin_display["tipo"] == "Entrada"]["valor_num"].sum())
        total_saidas = float(df_fin_display[df_fin_display["tipo"] == "Saída"]["valor_num"].sum())
        saldo = total_entradas - total_saidas
        st.markdown(
            f"**Total de Entradas:** {format_currency_br(total_entradas)} &nbsp;&nbsp; "
            f"**Total de Saídas:** {format_currency_br(total_saidas)} &nbsp;&nbsp; "
            f"**Saldo:** {format_currency_br(saldo)}"
        )

        st.markdown("#### Curva S Financeira (Previsto x Realizado)")
        c1, c2 = st.columns(2)
        with c1:
            inicio_mes = st.text_input(
                "Início do período (AAAA-MM)",
                value=f"{datetime.now().year}-{str(datetime.now().month).zfill(2)}",
                key="fluxo_inicio",
            )
        with c2:
            meses = st.number_input(
                "Número de meses", min_value=1, max_value=36, value=6, key="fluxo_meses"
            )

        if st.button("Gerar Curva S Financeira", type="primary"):
            df_fluxo, fig_fluxo = gerar_curva_s_financeira(
                finances, inicio_mes, int(meses)
            )
            if fig_fluxo:
                # <-- chave única adicionada para evitar StreamlitDuplicateElementId
                st.plotly_chart(fig_fluxo, use_container_width=True, key="curva_s_financeira_tab")
            else:
                st.warning(
                    "Não foi possível gerar a Curva S financeira. Verifique os lançamentos."
                )
    else:
        st.info("Nenhum lançamento financeiro cadastrado até o momento.")


# --------------------------------------------------------
# TAB 4 - KPIs
# --------------------------------------------------------

with tabs[4]:
    st.markdown("### 📊 KPIs de Qualidade")

    with st.expander("Registrar ponto de KPI", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            nome_kpi = st.text_input("Nome do KPI", key="kpi_nome")
        with c2:
            unidade = st.text_input(
                "Unidade (% , horas, nº itens, etc.)", key="kpi_unid"
            )
        with c3:
            meses_proj = st.number_input(
                "Duração do projeto (meses)",
                min_value=1,
                max_value=60,
                value=12,
                key="kpi_meses",
            )
        with c4:
            mes_ref = st.number_input(
                "Mês de referência",
                min_value=1,
                max_value=60,
                value=1,
                key="kpi_mes_ref",
            )

        c5, c6 = st.columns(2)
        with c5:
            prev = st.number_input("Valor previsto", value=0.0, key="kpi_prev")
        with c6:
            real = st.number_input("Valor realizado", value=0.0, key="kpi_real")

        if st.button("Adicionar ponto KPI", type="primary"):
            if not nome_kpi.strip() or not unidade.strip():
                st.warning("Informe nome e unidade do KPI.")
            else:
                kpis.append(
                    {
                        "nome": nome_kpi.strip(),
                        "unidade": unidade.strip(),
                        "mesesProjeto": int(meses_proj),
                        "mes": int(mes_ref),
                        "previsto": float(prev),
                        "realizado": float(real),
                    }
                )
                salvar_estado()
                st.success("Ponto de KPI registrado.")
                st.rerun()

    if kpis:
        st.markdown("#### Tabela de KPIs")
        df_k = pd.DataFrame(kpis)
        st.dataframe(df_k, use_container_width=True, height=260)

        idx_kpi = st.selectbox(
            "Selecione o ponto de KPI para editar / excluir",
            options=list(range(len(kpis))),
            format_func=lambda i: f"{kpis[i]['nome']} - Mês {kpis[i]['mes']} (Previsto: {kpis[i]['previsto']}, Realizado: {kpis[i]['realizado']})",
            key="kpi_del_idx"
        )

        # --------- EDIÇÃO DE KPI ---------
        k_sel = kpis[idx_kpi]
        ek1, ek2, ek3, ek4 = st.columns(4)
        with ek1:
            nome_kpi_edit = st.text_input(
                "Nome do KPI (edição)",
                value=k_sel.get("nome", ""),
                key="kpi_nome_edit"
            )
        with ek2:
            unidade_edit = st.text_input(
                "Unidade (edição)",
                value=k_sel.get("unidade", ""),
                key="kpi_unid_edit"
            )
        with ek3:
            meses_proj_edit = st.number_input(
                "Duração do projeto (meses) - edição",
                min_value=1,
                max_value=60,
                value=int(k_sel.get("mesesProjeto", 12)),
                key="kpi_meses_edit",
            )
        with ek4:
            mes_ref_edit = st.number_input(
                "Mês de referência - edição",
                min_value=1,
                max_value=60,
                value=int(k_sel.get("mes", 1)),
                key="kpi_mes_ref_edit",
            )

        ek5, ek6 = st.columns(2)
        with ek5:
            prev_edit = st.number_input(
                "Valor previsto (edição)",
                value=float(k_sel.get("previsto", 0.0)),
                key="kpi_prev_edit"
            )
        with ek6:
            real_edit = st.number_input(
                "Valor realizado (edição)",
                value=float(k_sel.get("realizado", 0.0)),
                key="kpi_real_edit"
            )

        if st.button("Salvar alterações do KPI selecionado", key="kpi_edit_btn"):
            k_sel["nome"] = nome_kpi_edit.strip()
            k_sel["unidade"] = unidade_edit.strip()
            k_sel["mesesProjeto"] = int(meses_proj_edit)
            k_sel["mes"] = int(mes_ref_edit)
            k_sel["previsto"] = float(prev_edit)
            k_sel["realizado"] = float(real_edit)
            salvar_estado()
            st.success("KPI atualizado.")
            st.rerun()

        if st.button("Excluir ponto de KPI selecionado", key="kpi_del_btn"):
            kpis.pop(idx_kpi)
            salvar_estado()
            st.success("Ponto de KPI excluído.")
            st.rerun()

        st.markdown("#### Gráfico do KPI")
        kpi_names = list({k["nome"] for k in kpis})
        kpi_sel = st.selectbox("Selecione o KPI para plotar", kpi_names, key="kpi_sel")
        serie = [k for k in kpis if k["nome"] == kpi_sel]
        serie = sorted(serie, key=lambda x: x["mes"])
        df_plot = pd.DataFrame(
            {
                "Mês": [f"M{p['mes']}" for p in serie],
                "Previsto": [p["previsto"] for p in serie],
                "Realizado": [p["realizado"] for p in serie],
            }
        )
        fig_kpi = px.line(
            df_plot,
            x="Mês",
            y=["Previsto", "Realizado"],
            title=f"Evolução do KPI: {kpi_sel}",
        )
        fig_kpi.update_traces(mode="lines+markers")
        fig_kpi.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=30, r=20, t=35, b=30),
        )
        # <-- chave única adicionada para evitar duplicação de elemento
        st.plotly_chart(fig_kpi, use_container_width=True, key="kpi_chart_tab")
    else:
        st.info("Nenhum KPI registrado até o momento.")


# --------------------------------------------------------
# TAB 5 - RISCOS
# --------------------------------------------------------

with tabs[5]:
    st.markdown("### ⚠️ Registro de riscos")

    def peso_impacto(impacto):
        if impacto == "alto":
            return 3
        if impacto == "medio":
            return 2
        return 1

    def peso_prob(prob):
        if prob == "alta":
            return 3
        if prob == "media":
            return 2
        return 1

    with st.expander("Adicionar risco", expanded=True):
        desc_risk = st.text_input("Descrição do risco", key="risk_desc")
        c1_, c2_, c3_ = st.columns(3)
        with c1_:
            impacto = st.selectbox(
                "Impacto", ["baixo", "medio", "alto"], index=0, key="risk_imp"
            )
        with c2_:
            prob = st.selectbox(
                "Probabilidade", ["baixa", "media", "alta"], index=0, key="risk_prob"
            )
        with c3_:
            resposta = st.selectbox(
                "Resposta",
                ["mitigar", "eliminar", "aceitar", "transferir"],
                index=0,
                key="risk_resp",
            )
        plano = st.text_area("Plano de tratamento", key="risk_plano")

        if st.button("Adicionar risco", type="primary"):
            if not desc_risk.strip():
                st.warning("Descreva o risco.")
            else:
                indice = peso_impacto(impacto) * peso_prob(prob)
                risks.append(
                    {
                        "descricao": desc_risk.strip(),
                        "impacto": impacto,
                        "prob": prob,
                        "resposta": resposta,
                        "plano": plano.strip(),
                        "indice": indice,
                    }
                )
                salvar_estado()
                st.success("Risco adicionado.")
                st.rerun()

    if risks:
        df_r = pd.DataFrame(risks).sort_values(by="indice", ascending=False)
        st.markdown("#### Matriz de riscos (ordenada por criticidade)")
        st.dataframe(
            df_r[["descricao", "impacto", "prob", "indice", "resposta"]],
            use_container_width=True,
            height=260,
        )

        idx_risk = st.selectbox(
            "Selecione o risco para editar / excluir",
            options=list(range(len(risks))),
            format_func=lambda i: f"{risks[i]['descricao'][:60]} (Índice {risks[i]['indice']})",
            key="risk_del_idx"
        )

        # --------- EDIÇÃO DE RISCO ---------
        r_sel = risks[idx_risk]
        er1, er2, er3 = st.columns(3)
        with er1:
            desc_risk_edit = st.text_input(
                "Descrição do risco (edição)",
                value=r_sel.get("descricao", ""),
                key="risk_desc_edit"
            )
        with er2:
            imp_opts = ["baixo", "medio", "alto"]
            imp_val = r_sel.get("impacto", "baixo")
            if imp_val not in imp_opts:
                imp_val = "baixo"
            impacto_edit = st.selectbox(
                "Impacto (edição)",
                imp_opts,
                index=imp_opts.index(imp_val),
                key="risk_imp_edit"
            )
        with er3:
            prob_opts = ["baixa", "media", "alta"]
            prob_val = r_sel.get("prob", "baixa")
            if prob_val not in prob_opts:
                prob_val = "baixa"
            prob_edit = st.selectbox(
                "Probabilidade (edição)",
                prob_opts,
                index=prob_opts.index(prob_val),
                key="risk_prob_edit"
            )

        er4, = st.columns(1)
        with er4:
            resp_opts = ["mitigar", "eliminar", "aceitar", "transferir"]
            resp_val = r_sel.get("resposta", "mitigar")
            if resp_val not in resp_opts:
                resp_val = "mitigar"
            resposta_edit = st.selectbox(
                "Resposta (edição)",
                resp_opts,
                index=resp_opts.index(resp_val),
                key="risk_resp_edit"
            )

        plano_edit = st.text_area(
            "Plano de tratamento (edição)",
            value=r_sel.get("plano", ""),
            key="risk_plano_edit"
        )

        if st.button("Salvar alterações do risco selecionado", key="risk_edit_btn"):
            r_sel["descricao"] = desc_risk_edit.strip()
            r_sel["impacto"] = impacto_edit
            r_sel["prob"] = prob_edit
            r_sel["resposta"] = resposta_edit
            r_sel["plano"] = plano_edit.strip()
            r_sel["indice"] = peso_impacto(impacto_edit) * peso_prob(prob_edit)
            salvar_estado()
            st.success("Risco atualizado.")
            st.rerun()

        if st.button("Excluir risco selecionado", key="risk_del_btn"):
            risks.pop(idx_risk)
            salvar_estado()
            st.success("Risco excluído.")
            st.rerun()
    else:
        st.info("Nenhum risco registrado.")


# --------------------------------------------------------
# TAB 6 - LIÇÕES
# --------------------------------------------------------

with tabs[6]:
    st.markdown("### 🧠 Lições aprendidas")

    with st.expander("Registrar lição", expanded=True):
        col1_, col2_ = st.columns(2)
        with col1_:
            titulo_l = st.text_input("Título da lição", key="lesson_tit")
            fase_l = st.selectbox(
                "Fase",
                ["inicio", "planejamento", "execucao", "monitoramento", "encerramento"],
                key="lesson_fase",
            )
        with col2_:
            categoria_l = st.selectbox(
                "Categoria",
                ["processo", "tecnico", "pessoas", "cliente", "negocio"],
                key="lesson_cat",
            )
        desc_l = st.text_area("Descrição da lição", key="lesson_desc")
        rec_l = st.text_area(
            "Recomendação para futuros projetos", key="lesson_rec"
        )

        if st.button("Adicionar lição", type="primary"):
            if not titulo_l.strip() or not desc_l.strip():
                st.warning("Título e descrição são obrigatórios.")
            else:
                lessons.append(
                    {
                        "titulo": titulo_l.strip(),
                        "fase": fase_l,
                        "categoria": categoria_l,
                        "descricao": desc_l.strip(),
                        "recomendacao": rec_l.strip(),
                    }
                )
                salvar_estado()
                st.success("Lição adicionada.")
                st.rerun()

    if lessons:
        df_l = pd.DataFrame(lessons)
        st.dataframe(df_l, use_container_width=True, height=260)

        idx_lesson = st.selectbox(
            "Selecione a lição para excluir",
            options=list(range(len(lessons))),
            format_func=lambda i: f"{lessons[i]['titulo']} - {lessons[i]['fase']} - {lessons[i]['categoria']}",
            key="lesson_del_idx"
        )
        if st.button("Excluir lição selecionada", key="lesson_del_btn"):
            lessons.pop(idx_lesson)
            salvar_estado()
            st.success("Lição excluída.")
            st.rerun()
    else:
        st.info("Nenhuma lição registrada.")


# --------------------------------------------------------
# TAB 7 - ENCERRAMENTO
# --------------------------------------------------------

with tabs[7]:
    st.markdown("### ✅ Encerramento do projeto")

    col1__, col2__ = st.columns(2)
    with col1__:
        close_data["resumo"] = st.text_area(
            "Resumo executivo",
            value=close_data.get("resumo", ""),
            height=120,
        )
        close_data["resultados"] = st.text_area(
            "Resultados alcançados",
            value=close_data.get("resultados", ""),
            height=120,
        )
        close_data["escopo"] = st.text_area(
            "Atendimento aos requisitos / escopo",
            value=close_data.get("escopo", ""),
            height=120,
        )

    with col2__:
        close_data["aceite"] = st.text_area(
            "Aceite formal do cliente",
            value=close_data.get("aceite", ""),
            height=120,
        )
        close_data["recomendacoes"] = st.text_area(
            "Recomendações para projetos futuros",
            value=close_data.get("recomendacoes", ""),
            height=120,
        )
        close_data["obs"] = st.text_area(
            "Observações finais da gerência",
            value=close_data.get("obs", ""),
            height=120,
        )

    if st.button("💾 Salvar encerramento", type="primary"):
        salvar_estado()
        st.success("Dados de encerramento salvos.")


# -------------------------
# TAB 8 - RELATÓRIOS HTML (cole este bloco no lugar do bloco atual de relatórios)
# -------------------------
import plotly.io as pio
import plotly.graph_objects as go

# CSS claro usado nos relatórios (tema para o HTML exportado)
REPORT_CSS = """
<style>
body { font-family: "Segoe UI", Arial, sans-serif; margin:0; padding:0; background:#f4f6fb; color:#222; }
.header { background: linear-gradient(120deg,#0d47a1,#00bcd4); color:#fff; padding:18px 30px; display:flex; justify-content:space-between; align-items:center; }
.header .title { font-size:22px; font-weight:700; margin:0; }
.header .subtitle { font-size:14px; margin:2px 0 0 0; opacity:0.95; }
.container { max-width:1100px; margin:20px auto; background:#fff; border-radius:10px; padding:20px 24px; box-shadow:0 8px 28px rgba(2,6,23,0.08); color:#222; }
.section-title { font-size:16px; color:#0d47a1; margin:12px 0 8px 0; }
.table-report { width:100%; border-collapse:collapse; font-size:13px; }
.table-report th, .table-report td { border:1px solid #e6eef6; padding:8px 10px; text-align:left; vertical-align:top; }
.table-report th { background:#eaf6ff; color:#0d47a1; font-weight:700; }
.small-note { color:#555; font-size:13px; }
.badge { display:inline-block; padding:6px 12px; border-radius:20px; background:#eaf6ff; color:#0d47a1; font-weight:700; font-size:12px; }
.report-grid { display:flex; gap:16px; flex-wrap:wrap; margin-top:12px; }
.report-card { flex:1 1 320px; background:#fff; border:1px solid #eef6ff; padding:12px; border-radius:8px; }
.report-chart { text-align:center; margin-top:8px; }
.footer { text-align:center; color:#666; margin-top:18px; font-size:13px; }
.badge-status { display:inline-block; padding:6px 10px; border-radius:6px; font-weight:700; font-size:12px; }
.badge-completo { background:#e6f6ea; color:#147a30; border:1px solid #c8efd0; }
.badge-atraso { background:#fdecea; color:#c2271d; border:1px solid #f6c8c6; }
.badge-pendente { background:#e9f2fb; color:#0d47a1; border:1px solid #dbeeff; }
</style>
"""

def montar_html_completo(html_corpo: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="utf-8">
        <title>Relatório do Projeto</title>
        {REPORT_CSS}
    </head>
    <body>
        {html_corpo}
    </body>
    </html>
    """

def build_eap_html_table(eap_tasks):
    if not eap_tasks:
        return "<p>Não há atividades cadastradas na EAP.</p>"
    try:
        df = pd.DataFrame(eap_tasks).sort_values(by="codigo")
    except Exception:
        df = pd.DataFrame(eap_tasks)
    html = "<table class='table-report'><thead><tr>"
    headers = ["Código", "Descrição", "Nível", "Duração (dias)", "Responsável", "Status", "Predecessoras"]
    for h in headers:
        html += f"<th>{h}</th>"
    html += "</tr></thead><tbody>"
    for _, row in df.iterrows():
        codigo = row.get("codigo", "")
        descricao = row.get("descricao", "")
        nivel = int(row.get("nivel", 1)) if row.get("nivel") else 1
        dur = row.get("duracao", "")
        resp = row.get("responsavel", "")
        status = row.get("status", "")
        preds = row.get("predecessoras", [])
        preds_text = ", ".join(preds) if isinstance(preds, list) else str(preds)
        indent_px = 12 * max(nivel - 1, 0)
        html += "<tr>"
        html += f"<td>{codigo}</td>"
        html += f"<td style='padding-left:{indent_px}px'>{descricao}</td>"
        html += f"<td>{nivel}</td>"
        html += f"<td>{dur}</td>"
        html += f"<td>{resp}</td>"
        html += f"<td>{status}</td>"
        html += f"<td>{preds_text}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html

def months_between(start_date, end_date):
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

# helper para último dia do mês
def end_of_month(dt: date):
    if dt.month == 12:
        return date(dt.year, 12, 31)
    else:
        return date(dt.year, dt.month + 1, 1) - timedelta(days=1)

with tabs[8]:
    st.markdown("### 📑 Relatórios em HTML / CSS")
    tipo_rel = st.selectbox("Selecione o relatório",
                            ["Extrato financeiro", "Resumo TAP", "Riscos e Lições", "Relatório completo"],
                            index=0)

    df_fin = pd.DataFrame(finances) if finances else pd.DataFrame()
    df_r = pd.DataFrame(risks) if risks else pd.DataFrame()
    df_l = pd.DataFrame(lessons) if lessons else pd.DataFrame()
    df_eap_rel = pd.DataFrame(eapTasks) if eapTasks else pd.DataFrame()

    # --------------------- Extrato Financeiro ---------------------
    if tipo_rel == "Extrato financeiro":
        if df_fin.empty:
            st.info("Não há lançamentos financeiros para gerar o extrato.")
        else:
            if "qtdRecorrencias" not in df_fin.columns:
                df_fin["qtdRecorrencias"] = 1
            for col in ["categoria", "subcategoria", "dataPrevista", "dataRealizada", "realizado", "tipo"]:
                if col not in df_fin.columns:
                    df_fin[col] = (False if col == "realizado" else "")
            df_fin["Valor"] = (df_fin["valor"] * df_fin["qtdRecorrencias"]).map(format_currency_br)
            df_fin["Prevista"] = df_fin["dataPrevista"]
            df_fin["Realizada"] = df_fin["dataRealizada"].replace("", "-")
            df_fin["Status"] = df_fin["realizado"].map(lambda x: "Realizado" if x else "Pendente")
            df_fin["Tipo"] = df_fin["tipo"]
            df_fin["Categoria"] = df_fin["categoria"]
            df_fin["Subcategoria"] = df_fin["subcategoria"]

            df_show = df_fin[["Tipo", "descricao", "Categoria", "Subcategoria", "Valor", "Prevista", "Realizada", "Status"]].copy()
            df_show.columns = ["Tipo", "Descrição", "Categoria", "Subcategoria", "Valor", "Prevista", "Realizada", "Status"]
            html_tabela = df_show.to_html(index=False, classes="table-report", border=0, justify="left")

            total_entradas = (df_fin[df_fin["Tipo"] == "Entrada"]["valor"] * df_fin[df_fin["Tipo"] == "Entrada"]["qtdRecorrencias"]).sum()
            total_saidas = (df_fin[df_fin["Tipo"] == "Saída"]["valor"] * df_fin[df_fin["Tipo"] == "Saída"]["qtdRecorrencias"]).sum()
            saldo = total_entradas - total_saidas

            # gráfico entradas x saídas (totais)
            fig_totals = go.Figure()
            fig_totals.add_trace(go.Bar(
                x=["Entradas", "Saídas"],
                y=[total_entradas, total_saidas],
                marker_color=['#2ecc71', '#e74c3c'],
            ))
            fig_totals.update_layout(template='plotly_white', height=320, margin=dict(t=30, b=20),
                                     yaxis_tickformat=",.2f")
            fig_totals.update_yaxes(title_text='Valor (R$)')
            totals_html = pio.to_html(fig_totals, include_plotlyjs='cdn', full_html=False)

            html_corpo = f"""
            <div class="container">
              <div class="header">
                <div>
                    <div class="title">Extrato Financeiro do Projeto</div>
                    <div class="subtitle">Projeto: {tap.get('nome','')} — Gerente: {tap.get('gerente','')}</div>
                </div>
                <div class="badge">Relatório Financeiro</div>
              </div>
              <div style="padding:18px;">
                <h3 class="section-title">Resumo financeiro</h3>
                <div class="report-grid">
                    <div class="report-card"><strong>Total Entradas</strong><div style="margin-top:10px">{format_currency_br(total_entradas)}</div></div>
                    <div class="report-card"><strong>Total Saídas</strong><div style="margin-top:10px">{format_currency_br(total_saidas)}</div></div>
                    <div class="report-card"><strong>Saldo</strong><div style="margin-top:10px">{format_currency_br(saldo)}</div></div>
                </div>
                <h3 class="section-title" style="margin-top:16px;">Gráfico Entradas x Saídas</h3>
                <div>{totals_html}</div>
                <h3 class="section-title" style="margin-top:16px;">Lançamentos detalhados</h3>
                {html_tabela}
              </div>
              <div class="footer">Relatório gerado em {datetime.now().strftime("%d/%m/%Y %H:%M")}</div>
            </div>
            """
            components.html(REPORT_CSS + html_corpo, height=820, scrolling=True)
            html_completo = montar_html_completo(html_corpo)
            st.download_button("⬇️ Baixar relatório em HTML", data=html_completo.encode("utf-8"),
                                file_name="relatorio_extrato_financeiro.html", mime="text/html")

    # --------------------- Resumo TAP ---------------------
    elif tipo_rel == "Resumo TAP":
        html_corpo = f"""
        <div class="container">
          <div class="header">
            <div>
                <div class="title">Resumo do Termo de Abertura do Projeto (TAP)</div>
                <div class="subtitle">Projeto ID: {st.session_state.current_project_id}</div>
            </div>
            <div class="badge">Resumo TAP</div>
          </div>
          <div style="padding:18px;">
            <h3 class="section-title">Identificação</h3>
            <p><strong>Nome:</strong> {tap.get('nome','')}</p>
            <p><strong>Gerente:</strong> {tap.get('gerente','')}</p>
            <p><strong>Patrocinador:</strong> {tap.get('patrocinador','')}</p>
            <p><strong>Data de início:</strong> {tap.get('dataInicio','')}</p>
            <p><strong>Status:</strong> {tap.get('status','rascunho')}</p>
            <h3 class="section-title">Objetivo</h3>
            <p>{tap.get('objetivo','').replace(chr(10),'<br>')}</p>
            <h3 class="section-title">Escopo inicial</h3>
            <p>{tap.get('escopo','').replace(chr(10),'<br>')}</p>
          </div>
          <div class="footer">Relatório gerado em {datetime.now().strftime("%d/%m/%Y %H:%M")}</div>
        </div>
        """
        components.html(REPORT_CSS + html_corpo, height=700, scrolling=True)
        html_completo = montar_html_completo(html_corpo)
        st.download_button("⬇️ Baixar relatório em HTML", data=html_completo.encode("utf-8"),
                            file_name="relatorio_resumo_tap.html", mime="text/html")

    # --------------------- Riscos e Lições ---------------------
    elif tipo_rel == "Riscos e Lições":
        if not df_r.empty:
            df_r_show = df_r[["descricao","impacto","prob","indice","resposta"]].copy()
            df_r_show.columns = ["Risco","Impacto","Probabilidade","Índice","Resposta"]
            html_riscos = df_r_show.to_html(index=False, classes="table-report", border=0, justify="left")
        else:
            html_riscos = "<p>Não há riscos cadastrados.</p>"

        if not df_l.empty:
            df_l_show = df_l[["titulo","fase","categoria","descricao","recomendacao"]].copy()
            df_l_show.columns = ["Título","Fase","Categoria","Lição","Recomendação"]
            html_licoes = df_l_show.to_html(index=False, classes="table-report", border=0, justify="left")
        else:
            html_licoes = "<p>Não há lições registradas.</p>"

        html_corpo = f"""
        <div class="container">
          <div class="header">
            <div>
                <div class="title">Riscos e Lições Aprendidas</div>
                <div class="subtitle">Projeto: {tap.get('nome','')}</div>
            </div>
            <div class="badge">Riscos & Lições</div>
          </div>
          <div style="padding:18px;">
            <h3 class="section-title">Riscos mapeados</h3>
            {html_riscos}
            <h3 class="section-title">Lições aprendidas</h3>
            {html_licoes}
          </div>
          <div class="footer">Relatório gerado em {datetime.now().strftime("%d/%m/%Y %H:%M")}</div>
        </div>
        """
        components.html(REPORT_CSS + html_corpo, height=700, scrolling=True)
        html_completo = montar_html_completo(html_corpo)
        st.download_button("⬇️ Baixar relatório em HTML", data=html_completo.encode("utf-8"),
                            file_name="relatorio_riscos_licoes.html", mime="text/html")

    # --------------------- Relatório completo ---------------------
    else:
        qtd_eap = len(eapTasks)
        qtd_fin = len(finances)
        qtd_kpi = len(kpis)
        qtd_risk = len(risks)
        qtd_les = len(lessons)

        html_eap = build_eap_html_table(eapTasks)

        resumo_fin_html = "<p>Não há lançamentos financeiros cadastrados.</p>"
        html_fluxo_table = "<p>Gráfico completo exibido abaixo no aplicativo (interativo).</p>"
        df_fluxo_rel = None
        fig_fluxo_rel = None
        total_previsto_final = 0.0
        total_realizado_final = 0.0

        # gerar curva S financeira (quando possível)
        if finances:
            try:
                if eapTasks and tap.get("dataInicio"):
                    tasks_cpm, total_dias = calcular_cpm(eapTasks)
                    data_inicio_dt = datetime.strptime(tap["dataInicio"], "%Y-%m-%d").date()
                    projeto_fim_dt = data_inicio_dt + timedelta(days=total_dias)
                    meses = months_between(data_inicio_dt.replace(day=1), projeto_fim_dt.replace(day=1))
                    inicio_mes_str = f"{data_inicio_dt.year}-{str(data_inicio_dt.month).zfill(2)}"
                    df_fluxo_rel, _ = gerar_curva_s_financeira(finances, inicio_mes_str, meses)
                else:
                    inicio_mes_str = f"{datetime.now().year}-{str(datetime.now().month).zfill(2)}"
                    df_fluxo_rel, _ = gerar_curva_s_financeira(finances, inicio_mes_str, 6)

                # Construir gráfico com cores definidas para Previsto/Realizado
                if df_fluxo_rel is not None and len(df_fluxo_rel):
                    total_previsto_final = float(df_fluxo_rel["Previsto (acumulado)"].iloc[-1])
                    total_realizado_final = float(df_fluxo_rel["Realizado (acumulado)"].iloc[-1])
                    html_fluxo_table = df_fluxo_rel.to_html(index=False, classes="table-report", border=0)

                    # construir figura custom para prev/real com cores claras
                    fig_flux = go.Figure()
                    fig_flux.add_trace(go.Scatter(
                        x=df_fluxo_rel["Mês"],
                        y=df_fluxo_rel["Previsto (acumulado)"],
                        mode='lines+markers',
                        name='Previsto (acum.)',
                        line=dict(color='#0d47a1', width=2),
                        marker=dict(size=6)
                    ))
                    fig_flux.add_trace(go.Scatter(
                        x=df_fluxo_rel["Mês"],
                        y=df_fluxo_rel["Realizado (acumulado)"],
                        mode='lines+markers',
                        name='Realizado (acum.)',
                        line=dict(color='#2ecc71', width=2),
                        marker=dict(size=6)
                    ))
                    fig_flux.update_layout(template='plotly_white', height=360, margin=dict(t=30, b=40),
                                           xaxis_title='Mês', yaxis_title='Valor (R$)')
                    fig_flux_rel_html = pio.to_html(fig_flux, include_plotlyjs='cdn', full_html=False)

                    # Diferença prev - real (barras)
                    prev_vals = df_fluxo_rel["Previsto (acumulado)"].tolist()
                    real_vals = df_fluxo_rel["Realizado (acumulado)"].tolist()
                    diff = [p - r for p, r in zip(prev_vals, real_vals)]
                    color_cat = ['Positivo' if d >= 0 else 'Negativo' for d in diff]
                    diff_fig = px.bar(x=df_fluxo_rel["Mês"], y=diff, color=color_cat,
                                      color_discrete_map={'Positivo':'#2ecc71','Negativo':'#e74c3c'},
                                      labels={'x':'Mês','y':'Dif. Prev - Real'})
                    diff_fig.update_layout(showlegend=False, template='plotly_white', height=300)
                    diff_html = pio.to_html(diff_fig, include_plotlyjs='cdn', full_html=False)

                    ratio = (total_realizado_final / total_previsto_final) if total_previsto_final else 0.0
                    if ratio >= 0.95:
                        sugestao_fluxo = "Fluxo de caixa saudável: realização próxima ao previsto."
                    elif ratio >= 0.8:
                        sugestao_fluxo = "Atenção: realização moderadamente abaixo do previsto. Verificar pagamentos/consumo."
                    else:
                        sugestao_fluxo = "Risco financeiro: realização muito abaixo do previsto. Revisar custos/cronograma."
                else:
                    fig_flux_rel_html = ""
                    diff_html = ""
                    sugestao_fluxo = "Não foi possível gerar o fluxo financeiro automaticamente."
            except Exception:
                fig_flux_rel_html = ""
                diff_html = ""
                sugestao_fluxo = "Não foi possível gerar o fluxo financeiro automaticamente. Verifique os dados."
        else:
            fig_flux_rel_html = ""
            diff_html = ""
            sugestao_fluxo = "Não há lançamentos financeiros."

        # Criar gráfico mensal por tipo (Entrada vs Saída) com as mesmas cores
        fluxo_por_mes_html = ""
        if df_fluxo_rel is not None and len(df_fluxo_rel) and finances:
            try:
                # Definir periodo
                start_label = df_fluxo_rel["Mês"].iloc[0]
                end_label = df_fluxo_rel["Mês"].iloc[-1]
                sy, sm = map(int, start_label.split("-"))
                ey, em = map(int, end_label.split("-"))
                inicio = date(sy, sm, 1)
                fim = end_of_month(date(ey, em, 1))

                mapa_entr_prev = {k: 0.0 for k in df_fluxo_rel["Mês"].tolist()}
                mapa_sai_prev = {k: 0.0 for k in df_fluxo_rel["Mês"].tolist()}
                mapa_entr_real = {k: 0.0 for k in df_fluxo_rel["Mês"].tolist()}
                mapa_sai_real = {k: 0.0 for k in df_fluxo_rel["Mês"].tolist()}

                def key_mes(d: date):
                    return f"{d.year}-{str(d.month).zfill(2)}"

                for l in finances:
                    tipo = l.get("tipo", "Entrada")
                    try:
                        valor = float(l.get("valor", 0.0))
                    except Exception:
                        valor = 0.0
                    ocorr = expandir_recorrencia(l, inicio, fim)
                    for d in ocorr:
                        k = key_mes(d)
                        if tipo == "Entrada":
                            mapa_entr_prev[k] += valor
                        else:
                            mapa_sai_prev[k] += valor
                    if l.get("realizado") and l.get("dataRealizada"):
                        try:
                            dr = datetime.strptime(l["dataRealizada"], "%Y-%m-%d").date()
                            if inicio <= dr <= fim:
                                k = key_mes(dr)
                                if tipo == "Entrada":
                                    mapa_entr_real[k] += valor
                                else:
                                    mapa_sai_real[k] += valor
                        except Exception:
                            pass

                # montar df
                months = df_fluxo_rel["Mês"].tolist()
                df_mt = pd.DataFrame({
                    "Mês": months,
                    "Entrada Previsto": [mapa_entr_prev[k] for k in months],
                    "Saída Previsto": [mapa_sai_prev[k] for k in months],
                    "Entrada Realizado": [mapa_entr_real[k] for k in months],
                    "Saída Realizado": [mapa_sai_real[k] for k in months],
                })
                # gráfico agrupado por tipo e status (prev/real)
                fig_type = go.Figure()
                fig_type.add_trace(go.Bar(x=df_mt["Mês"], y=df_mt["Entrada Previsto"], name='Entrada Previsto', marker_color='#2ecc71', opacity=0.6))
                fig_type.add_trace(go.Bar(x=df_mt["Mês"], y=df_mt["Entrada Realizado"], name='Entrada Realizado', marker_color='#27ae60'))
                fig_type.add_trace(go.Bar(x=df_mt["Mês"], y=df_mt["Saída Previsto"], name='Saída Previsto', marker_color='#f39c12', opacity=0.6))
                fig_type.add_trace(go.Bar(x=df_mt["Mês"], y=df_mt["Saída Realizado"], name='Saída Realizado', marker_color='#e74c3c'))
                fig_type.update_layout(barmode='group', template='plotly_white', height=360, legend_title_text='Séries')
                fluxo_por_mes_html = pio.to_html(fig_type, include_plotlyjs='cdn', full_html=False)
            except Exception:
                fluxo_por_mes_html = ""
        else:
            fluxo_por_mes_html = ""

        # KPIs: tabela com diferença e gráfico (cores Previsto azul / Realizado verde)
        kpi_table_html = "<p>Não há KPIs cadastrados.</p>"
        kpi_plot_html = ""
        sugestao_kpi = ""
        if kpis:
            try:
                df_k_all = pd.DataFrame(kpis).copy()
                df_k_all["Diferença"] = df_k_all["realizado"] - df_k_all["previsto"]
                # tabela
                df_k_show = df_k_all[["nome", "unidade", "mes", "previsto", "realizado", "Diferença"]].copy()
                df_k_show.columns = ["Nome", "Unidade", "Mês", "Previsto", "Realizado", "Diferença"]
                # formatar valores numéricos
                df_k_show["Previsto"] = df_k_show["Previsto"].map(lambda x: f"{x:.2f}")
                df_k_show["Realizado"] = df_k_show["Realizado"].map(lambda x: f"{x:.2f}")
                df_k_show["Diferença"] = df_k_show["Diferença"].map(lambda x: f"{x:.2f}")
                kpi_table_html = df_k_show.to_html(index=False, classes="table-report", border=0)

                # escolher KPI principal (o primeiro)
                kpi_names = list({k["nome"] for k in kpis})
                kpi_sel_auto = kpi_names[0]
                serie = [k for k in kpis if k["nome"] == kpi_sel_auto]
                serie = sorted(serie, key=lambda x: x["mes"])
                meses_k = [f"M{p['mes']}" for p in serie]
                previstos_k = [p["previsto"] for p in serie]
                realizados_k = [p["realizado"] for p in serie]

                figk = go.Figure()
                figk.add_trace(go.Scatter(x=meses_k, y=previstos_k, mode='lines+markers', name='Previsto', line=dict(color='#0d47a1')))
                figk.add_trace(go.Scatter(x=meses_k, y=realizados_k, mode='lines+markers', name='Realizado', line=dict(color='#2ecc71')))
                figk.update_layout(template='plotly_white', height=340, margin=dict(t=30), yaxis_title='Valor')
                kpi_plot_html = pio.to_html(figk, include_plotlyjs='cdn', full_html=False)

                # sugestão KPI
                ratios = []
                for pv, rl in zip(previstos_k, realizados_k):
                    try:
                        if pv and pv != 0:
                            ratios.append(rl / pv)
                    except Exception:
                        continue
                avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
                if avg_ratio >= 0.95:
                    sugestao_kpi = "Desempenho do KPI muito bom — metas sendo atingidas."
                elif avg_ratio >= 0.8:
                    sugestao_kpi = "KPI aceitável, mas atenção às variações mensais."
                else:
                    sugestao_kpi = "KPI abaixo do esperado — investigar causas (recursos/qualidade)."
            except Exception:
                kpi_table_html = "<p>Não foi possível gerar tabela/Gráfico de KPIs.</p>"
                kpi_plot_html = ""
                sugestao_kpi = "Erro ao gerar análise de KPI."

        # Riscos e Plano de Ação
        risks_html = "<p>Não há riscos cadastrados.</p>"
        if risks:
            df_r_show = pd.DataFrame(risks)[["descricao","impacto","prob","indice","resposta"]].copy()
            df_r_show.columns = ["Risco","Impacto","Probabilidade","Índice","Resposta"]
            risks_html = df_r_show.to_html(index=False, classes="table-report", border=0)
        action_html = "<p>Não há ações no plano.</p>"
        if action_plan:
            df_ap = pd.DataFrame(action_plan)[["descricao","responsavel","status","prazo","risco_relacionado"]].copy()
            df_ap.columns = ["Ação","Responsável","Status","Prazo","Risco relacionado"]
            action_html = df_ap.to_html(index=False, classes="table-report", border=0)

        # Gantt colorido: concluído verde, atraso vermelho, pendente azul
        gantt_html = ""
        try:
            if eapTasks and tap.get("dataInicio"):
                tasks_cpm, projeto_fim = calcular_cpm(eapTasks)
                data_inicio_dt = datetime.strptime(tap["dataInicio"], "%Y-%m-%d").date()
                rows = []
                hoje = date.today()
                for t in tasks_sched:
                    es = int(t.get("es", 0))
                    ef = int(t.get("ef", 0))
                    start = data_inicio_dt + timedelta(days=es)
                    # terminar no último dia (ef-1) ou ef? para plot, usar ef-1 para terminar no dia anterior? manter ef
                    finish = data_inicio_dt + timedelta(days=max(ef, es+1))
                    status_t = t.get("status", "")
                    # avaliar atraso
                    if status_t == "concluido":
                        estado = "concluido"
                    else:
                        fim_prev = data_inicio_dt + timedelta(days=ef)
                        estado = "atrasado" if fim_prev < hoje else "pendente"
                    rows.append({
                        "Task": f"{t.get('codigo')} - {t.get('descricao')}",
                        "Start": start,
                        "Finish": finish,
                        "Responsável": t.get("responsavel",""),
                        "Estado": estado
                    })
                if rows:
                    dfg = pd.DataFrame(rows)
                    color_map = {"concluido": "#2ecc71", "atrasado": "#e74c3c", "pendente": "#3498db"}
                    fig_gantt = px.timeline(dfg, x_start="Start", x_end="Finish", y="Task", color="Estado",
                                            color_discrete_map=color_map, hover_data=["Responsável"])
                    fig_gantt.update_yaxes(autorange="reversed")
                    fig_gantt.update_layout(template='plotly_white', height=520, margin=dict(l=20, r=20, t=50, b=40))
                    gantt_html = pio.to_html(fig_gantt, include_plotlyjs='cdn', full_html=False)
            else:
                gantt_html = "<p>Gantt indisponível — defina EAP e data de início.</p>"
        except Exception:
            gantt_html = "<p>Erro ao gerar Gantt.</p>"

        lessons_html = (pd.DataFrame(lessons)[['titulo','fase','categoria','descricao','recomendacao']].to_html(index=False, classes='table-report') if lessons else '<p>Não há lições registradas.</p>')

        # ── Organograma EAP (HTML para relatório) ──────────────────
        org_html = gerar_organograma_eap_html(eapTasks, nome_projeto=tap.get("nome") or "Projeto")

        # ── Curva S trabalho planejado×realizado (HTML) ─────────────
        curva_s_html = "<p>Curva S indisponível — defina data de início e EAP.</p>"
        try:
            if eapTasks and tap.get("dataInicio"):
                fig_cs = gerar_curva_s_trabalho(eapTasks, tap["dataInicio"])
                if fig_cs:
                    curva_s_html = pio.to_html(fig_cs, include_plotlyjs="cdn", full_html=False)
        except Exception:
            pass

        # ── Gantt planejado×realizado % avanço (HTML) ─────────────
        gantt_pr_html = "<p>Gantt indisponível — defina EAP e data de início.</p>"
        try:
            if eapTasks and tap.get("dataInicio"):
                fig_gantt_pr = gerar_gantt(eapTasks, tap["dataInicio"])
                if fig_gantt_pr:
                    gantt_pr_html = pio.to_html(fig_gantt_pr, include_plotlyjs="cdn", full_html=False)
        except Exception:
            pass

        # Montar HTML completo
        html_corpo = f"""
        <div class="container">
          <div class="header">
            <div>
                <div class="title">Relatório Completo do Projeto</div>
                <div class="subtitle">Projeto: {tap.get('nome','')} — ID {st.session_state.current_project_id}</div>
            </div>
            <div class="badge">Relatório Completo</div>
          </div>

          <div style="padding:18px;">
            <h3 class="section-title">1. Identificação e TAP</h3>
            <p><strong>Gerente:</strong> {tap.get('gerente','')} &nbsp;&nbsp; <strong>Patrocinador:</strong> {tap.get('patrocinador','')}</p>
            <p><strong>Data de início:</strong> {tap.get('dataInicio','')} &nbsp;&nbsp; <strong>Status:</strong> {tap.get('status','rascunho')}</p>

            <h3 class="section-title">2. Objetivo e Escopo</h3>
            <p><strong>Objetivo:</strong><br>{tap.get('objetivo','').replace(chr(10),'<br>')}</p>
            <p><strong>Escopo inicial:</strong><br>{tap.get('escopo','').replace(chr(10),'<br>')}</p>

            <h3 class="section-title">3. Resumo de números</h3>
            <div class="report-grid">
                <div class="report-card"><strong>Atividades na EAP</strong><div style="margin-top:8px">{qtd_eap}</div></div>
                <div class="report-card"><strong>Lançamentos financeiros</strong><div style="margin-top:8px">{qtd_fin}</div></div>
                <div class="report-card"><strong>Pontos de KPI</strong><div style="margin-top:8px">{qtd_kpi}</div></div>
                <div class="report-card"><strong>Riscos</strong><div style="margin-top:8px">{qtd_risk}</div></div>
                <div class="report-card"><strong>Lições</strong><div style="margin-top:8px">{qtd_les}</div></div>
            </div>

            <h3 class="section-title">4. Estrutura Analítica do Projeto (EAP)</h3>
            <h4 class="section-title">4.1 Organograma Hierárquico (PMBOK)</h4>
            <div>{org_html}</div>
            <h4 class="section-title" style="margin-top:14px;">4.2 Tabela da EAP</h4>
            {html_eap}

            <h3 class="section-title">5. Cronograma e Avanço Físico</h3>
            <h4 class="section-title">5.1 Curva S de Trabalho — Planejado × Realizado</h4>
            <div>{curva_s_html}</div>
            <h4 class="section-title" style="margin-top:14px;">5.2 Gráfico de Gantt — % Avanço por Atividade</h4>
            <div>{gantt_pr_html}</div>
            <h4 class="section-title" style="margin-top:14px;">5.3 Gantt — Status (Concluído / Atrasado / Pendente)</h4>
            <div>{gantt_html}</div>

            <h3 class="section-title">6. Resultados Financeiros (Previsto x Realizado)</h3>
            <div class="report-grid">
                <div class="report-card">
                    <strong>Resumo financeiro</strong>
                    <div style="margin-top:8px;">Total Previsto (acum): <strong>{format_currency_br(total_previsto_final)}</strong><br>
                    Total Realizado (acum): <strong>{format_currency_br(total_realizado_final)}</strong><br>
                    Saldo: <strong>{format_currency_br(total_previsto_final - total_realizado_final)}</strong></div>
                </div>
                <div class="report-card">
                    <strong>Análise rápida do fluxo</strong>
                    <p class="small-note">{sugestao_fluxo}</p>
                </div>
            </div>

            <div style="margin-top:12px;">
              <h4 class="section-title">Fluxo de Caixa (interativo)</h4>
              <div class="report-grid">
                <div class="report-card">{fig_flux_rel_html}</div>
                <div class="report-card">{diff_html}</div>
              </div>
              <h4 class="section-title" style="margin-top:12px;">Fluxo por mês - Entrada x Saída</h4>
              <div>{fluxo_por_mes_html}</div>
            </div>

            <h3 class="section-title" style="margin-top:10px;">7. KPIs (Previstos x Realizados)</h3>
            <div class="report-grid">
                <div class="report-card">
                    <strong>Tabela de KPIs</strong>
                    <div style="margin-top:8px;">{kpi_table_html}</div>
                </div>
                <div class="report-card">
                    <strong>Gráfico KPI principal</strong>
                    <div style="margin-top:8px;">{kpi_plot_html}</div>
                    <p class="small-note">{sugestao_kpi}</p>
                </div>
            </div>

            <h3 class="section-title">8. Riscos</h3>
            {risks_html}

            <h3 class="section-title">9. Plano de Ação</h3>
            {action_html}

            
            <div>{gantt_html}</div>

            <h3 class="section-title">10. Lições Aprendidas</h3>
            {lessons_html}

            <h3 class="section-title">11. Encerramento</h3>
            <p><strong>Resumo executivo:</strong><br>{close_data.get('resumo','').replace(chr(10),'<br>')}</p>
            <p><strong>Resultados alcançados:</strong><br>{close_data.get('resultados','').replace(chr(10),'<br>')}</p>

          </div>
          <div class="footer">Relatório gerado em {datetime.now().strftime("%d/%m/%Y %H:%M")} — BK Engenharia</div>
        </div>
        """

        # Exibe no app
        components.html(REPORT_CSS + html_corpo, height=1100, scrolling=True)
        # Prepara download (HTML completo)
        html_completo = montar_html_completo(html_corpo)
        st.download_button("⬇️ Baixar relatório em HTML", data=html_completo.encode("utf-8"),
                           file_name="relatorio_completo_projeto.html", mime="text/html")

        # Gráficos interativos adicionais abaixo (mantidos)
        st.markdown("#### 📈 Curva S de trabalho")
        if eapTasks and tap.get("dataInicio"):
            fig_s = gerar_curva_s_trabalho(eapTasks, tap["dataInicio"])
            if fig_s:
                st.plotly_chart(fig_s, width='stretch', key="curva_s_trabalho_relatorio")
        else:
            st.caption("Curva S de trabalho indisponível - verifique EAP e data de início.")

        st.markdown("#### 💹 Curva S Financeira (Previsto x Realizado)")
        if df_fluxo_rel is not None and 'Previsto (acumulado)' in df_fluxo_rel.columns:
            # montar fig_flux novamente para app (cores claros)
            fig_flux_app = go.Figure()
            fig_flux_app.add_trace(go.Scatter(x=df_fluxo_rel["Mês"], y=df_fluxo_rel["Previsto (acumulado)"], mode='lines+markers', name='Previsto', line=dict(color='#0d47a1')))
            fig_flux_app.add_trace(go.Scatter(x=df_fluxo_rel["Mês"], y=df_fluxo_rel["Realizado (acumulado)"], mode='lines+markers', name='Realizado', line=dict(color='#2ecc71')))
            fig_flux_app.update_layout(template='plotly_dark', height=350, margin=dict(l=30, r=20, t=35, b=30))
            st.plotly_chart(fig_flux_app, width='stretch', key="curva_s_financeira_report")
        else:
            if finances:
                st.caption("Curva S financeira indisponível para o período calculado (verifique data de início ou EAP).")
            else:
                st.caption("Curva S financeira indisponível - não há lançamentos.")

        st.markdown("#### 📊 KPI principal")
        if kpis:
            kpi_names = list({k["nome"] for k in kpis})
            kpi_sel_auto = kpi_names[0]
            serie = [k for k in kpis if k["nome"] == kpi_sel_auto]
            serie = sorted(serie, key=lambda x: x["mes"])
            df_plot = pd.DataFrame({
                "Mês": [f"M{p['mes']}" for p in serie],
                "Previsto": [p["previsto"] for p in serie],
                "Realizado": [p["realizado"] for p in serie],
            })
            fig_kpi = go.Figure()
            fig_kpi.add_trace(go.Scatter(x=df_plot["Mês"], y=df_plot["Previsto"], mode='lines+markers', name='Previsto', line=dict(color='#0d47a1')))
            fig_kpi.add_trace(go.Scatter(x=df_plot["Mês"], y=df_plot["Realizado"], mode='lines+markers', name='Realizado', line=dict(color='#2ecc71')))
            fig_kpi.update_layout(template='plotly_dark', height=350, margin=dict(l=30, r=20, t=35, b=30))
            st.plotly_chart(fig_kpi, width='stretch', key="kpi_chart_report")
        else:
            st.caption("Não há KPIs para exibir no relatório completo.")
# --------------------------------------------------------
# TAB 9 - PLANO DE AÇÃO
# --------------------------------------------------------

with tabs[9]:
    st.markdown("### 📌 Plano de Ação")

    with st.expander("Registrar item do plano de ação", expanded=True):
        pa1, pa2, pa3 = st.columns(3)
        with pa1:
            acao_desc = st.text_input("Ação / atividade", key="ap_desc")
        with pa2:
            acao_resp = st.text_input("Responsável", key="ap_resp")
        with pa3:
            acao_status = st.selectbox("Status", ["pendente", "em_andamento", "concluido"], key="ap_status")
        pa4, pa5 = st.columns(2)
        with pa4:
            acao_prazo = st.date_input("Prazo", key="ap_prazo", value=date.today())
        with pa5:
            if risks:
                riscos_fmt = [f"{i+1} - {r['descricao'][:50]}" for i, r in enumerate(risks)]
                idx_risk_ref = st.selectbox("Risco associado (opcional)", options=range(len(risks) + 1),
                                           format_func=lambda i: "Nenhum" if i == 0 else riscos_fmt[i-1], key="ap_risk_ref")
            else:
                idx_risk_ref = 0
                st.caption("Nenhum risco cadastrado para associar.")
        if st.button("Adicionar ação", type="primary", key="ap_add_btn"):
            if not acao_desc.strip():
                st.warning("Descreva a ação.")
            else:
                risk_ref = None
                if idx_risk_ref > 0:
                    risk_ref = risks[idx_risk_ref - 1]["descricao"]
                action_plan.append({
                    "descricao": acao_desc.strip(),
                    "responsavel": acao_resp.strip(),
                    "status": acao_status,
                    "prazo": acao_prazo.strftime("%Y-%m-%d"),
                    "risco_relacionado": risk_ref,
                })
                salvar_estado()
                st.success("Ação adicionada ao plano.")
                st.rerun()

    if action_plan:
        df_ap = pd.DataFrame(action_plan)
        st.markdown("#### Ações cadastradas")
        st.dataframe(df_ap, use_container_width=True, height=260)

        idx_ap = st.selectbox("Selecione a ação para excluir", options=list(range(len(action_plan))),
                              format_func=lambda i: f"{action_plan[i]['descricao'][:60]} - {action_plan[i]['status']}", key="ap_del_idx")
        if st.button("Excluir ação selecionada", key="ap_del_btn"):
            action_plan.pop(idx_ap)
            salvar_estado()
            st.success("Ação excluída.")
            st.rerun()
    else:
        st.info("Nenhuma ação registrada no plano de ação.")


# -------------------------------------------------
# ABA: CONTROLE DE PROJETOS (embutido)
# -------------------------------------------------
with tabs[10]:
    st.markdown("### 🗂️ Controle de Projetos (BK x Cliente)")
    pid = int(st.session_state.get("current_project_id") or 0)
    if not pid:
        st.info("Selecione um projeto para visualizar o Controle de Projetos.")
    else:
        # Renderiza o Controle de Projetos usando o mesmo banco do ERP
        render_controle_projetos(engine, pid)
