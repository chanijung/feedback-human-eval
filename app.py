import html
import json
import logging
import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import pandas as pd
import gspread
from gspread.exceptions import WorksheetNotFound
from google.oauth2 import service_account

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Feedback Human Evaluation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #f5f6f8;
    --surface: #ffffff;
    --surface2: #eef0f4;
    --border: #d1d5dc;
    --accent: #2563eb;
    --accent-soft: rgba(37,99,235,0.08);
    --text: #1e293b;
    --text-dim: #64748b;
    --radius: 10px;
    --green: #16a34a;
    --amber: #d97706;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif;
}

#MainMenu, footer, header { display: none !important; }
.block-container { padding: 1.2rem 1.5rem !important; max-width: 100% !important; }

/* Top bar */
.top-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.6rem 1rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); margin-bottom: 1rem;
}
.top-bar h1 {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem;
    font-weight: 600; color: var(--accent); margin: 0;
}
.progress-info { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: var(--text-dim); }

/* Feedback card (for sets display) */
.fb-card {
    background: var(--surface); border: 1.5px solid var(--border);
    border-radius: var(--radius); padding: 1rem;
}
.fb-card-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; font-weight: 600;
    color: var(--accent); text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

/* Unit card */
.unit-card {
    background: var(--surface); border: 2px solid var(--accent);
    border-radius: var(--radius); padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
}
.unit-source-badge {
    display: inline-block; background: var(--accent-soft);
    color: var(--accent); font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; font-weight: 600; padding: 0.15rem 0.5rem;
    border-radius: 4px; text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
}
.unit-text { font-size: 1rem; line-height: 1.75; color: var(--text); }

/* Section label */
.sec-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.2rem;
}

/* Nav center info */
.nav-center {
    text-align: center; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem; padding-top: 0.55rem; line-height: 1.4;
    color: var(--text-dim);
}

/* Buttons */
.stButton > button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.85rem !important; border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface2) !important; color: var(--text) !important;
    transition: all 0.15s !important;
}
.stButton > button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

/* Primary / main action buttons — red (multiple selectors for Streamlit version compat) */
[data-testid="baseButton-primary"],
button[kind="primary"],
.stButton > button[kind="primary"] {
    background-color: #dc2626 !important;
    background: #dc2626 !important;
    color: #ffffff !important;
    border-color: #dc2626 !important;
    font-weight: 600 !important;
}
[data-testid="baseButton-primary"]:hover,
button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background-color: #b91c1c !important;
    background: #b91c1c !important;
    color: #ffffff !important;
    border-color: #b91c1c !important;
}

/* Radio */
.stRadio > div { gap: 0.4rem !important; }
.stRadio label { font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.92rem !important; }

/* Tabs — Task 1 / Task 2: larger, bolder */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    padding: 0.7rem 1.6rem !important;
}
.stTabs [data-baseweb="tab"] button,
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span,
button[data-baseweb="tab"] {
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em !important;
}

/* Selectbox label */
.stSelectbox label { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.82rem !important; }

/* Completed item indicator */
.done-chip {
    display: inline-block; background: rgba(22,163,74,0.12);
    color: var(--green); border-radius: 20px; padding: 0.1rem 0.6rem;
    font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; font-weight: 600;
}
.todo-chip {
    display: inline-block; background: rgba(217,119,6,0.1);
    color: var(--amber); border-radius: 20px; padding: 0.1rem 0.6rem;
    font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; font-weight: 600;
}

/* Hide anchor link icons on headings */
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── GOOGLE SHEETS ─────────────────────────────────────────────────────────────
_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
_RANKINGS_SHEET = "Rankings"
_UNITS_SHEET = "UnitAnnotations"
_RANKINGS_HEADER = ["annotator", "paper_id", "ranked_models", "timestamp"]
# unit_hash = MD5(paper_id + source + unit_text) — used as a stable unique key for upsert
_UNITS_HEADER = [
    "unit_hash", "annotator", "paper_id", "feedback_source", "feedback_unit",
    "validity", "action", "details", "helpfulness", "timestamp",
]


def _unit_hash(paper_id: str, source: str, unit_text: str) -> str:
    """Stable 12-char hex key for a (paper_id, source, unit_text) triple."""
    raw = f"{paper_id}||{source}||{unit_text.strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _get_gc():
    """Create gspread client from secrets. Not cached so that fixing secrets takes effect after refresh."""
    try:
        if "gcp_service_account" not in st.secrets or "SPREADSHEET_ID" not in st.secrets:
            return None
        sa = dict(st.secrets["gcp_service_account"])
        # TOML/JSON sometimes give private_key with literal \n; Google expects real newlines
        if "private_key" in sa and isinstance(sa.get("private_key"), str):
            sa["private_key"] = sa["private_key"].replace("\\n", "\n")
        creds = service_account.Credentials.from_service_account_info(sa, scopes=_SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        logging.warning("GSheets init failed: %s", e)
        return None


def _sheet_id():
    raw = st.secrets.get("SPREADSHEET_ID", "").strip()
    return raw.split("/d/")[1].split("/")[0] if "/d/" in raw else raw


def _ensure_ws(sheet_name: str, header: list):
    """Returns (worksheet or None, error_message or None). Ensures row 1 is the column header."""
    gc = _get_gc()
    if gc is None:
        return None, "Google Sheets not configured (check SPREADSHEET_ID and gcp_service_account in secrets)."
    try:
        sheet_id = _sheet_id()
        if not sheet_id:
            return None, "SPREADSHEET_ID is empty."
        ss = gc.open_by_key(sheet_id)
        try:
            ws = ss.worksheet(sheet_name)
        except WorksheetNotFound:
            ws = ss.add_worksheet(title=sheet_name, rows=2000, cols=len(header) + 2)
        # Always ensure first row is the header (for new sheets or existing empty sheets)
        row1 = ws.row_values(1) if ws.row_count >= 1 else []
        if row1 != header:
            ws.update([header], "A1")
        return ws, None
    except Exception as e:
        logging.warning("Worksheet error: %s", e)
        return None, str(e)


def save_ranking(annotator: str, paper_id: str, ranked_models: list) -> tuple[bool, str | None]:
    """Returns (success, error_message)."""
    ws, err = _ensure_ws(_RANKINGS_SHEET, _RANKINGS_HEADER)
    if ws is None:
        return False, err
    try:
        ts = datetime.now(timezone.utc).isoformat()
        row = [annotator, paper_id, json.dumps(ranked_models), ts]
        existing = ws.get_all_values()
        for i, r in enumerate(existing[1:], start=2):
            if len(r) >= 2 and r[0].strip() == annotator and r[1].strip() == paper_id:
                ws.update([row], f"A{i}")
                return True, None
        ws.append_row(row)
        return True, None
    except Exception as e:
        logging.error("save_ranking failed: %s", e)
        return False, str(e)


def save_unit_annotation(
    annotator: str, paper_id: str, source: str, unit_text: str,
    validity: str, action: str, details: str, helpfulness: int,
) -> tuple[bool, str | None]:
    """Upsert a unit annotation. Matches existing row by (unit_hash, annotator).
    Returns (success, error_message).
    """
    ws, err = _ensure_ws(_UNITS_SHEET, _UNITS_HEADER)
    if ws is None:
        return False, err
    try:
        ts = datetime.now(timezone.utc).isoformat()
        uhash = _unit_hash(paper_id, source, unit_text)
        row = [uhash, annotator, paper_id, source, unit_text, validity, action, details, str(helpfulness), ts]
        existing = ws.get_all_values()
        # Find header to locate column positions
        header = existing[0] if existing else _UNITS_HEADER
        try:
            hash_col = header.index("unit_hash")
            ann_col = header.index("annotator")
        except ValueError:
            hash_col, ann_col = 0, 1
        for i, r in enumerate(existing[1:], start=2):
            if (len(r) > max(hash_col, ann_col)
                    and r[hash_col].strip() == uhash
                    and r[ann_col].strip() == annotator):
                ws.update([row], f"A{i}")
                return True, None
        ws.append_row(row)
        return True, None
    except Exception as e:
        logging.error("save_unit_annotation failed: %s", e)
        return False, str(e)


def load_rankings_from_sheets(annotator: str) -> dict:
    """Returns {paper_id: [model_rank1, model_rank2, ...]}"""
    ws, _ = _ensure_ws(_RANKINGS_SHEET, _RANKINGS_HEADER)
    if ws is None:
        return {}
    try:
        rows = ws.get_all_values()
        result = {}
        for r in rows[1:]:
            if r and r[0].strip() == annotator and len(r) >= 3:
                try:
                    result[r[1].strip()] = json.loads(r[2].strip())
                except json.JSONDecodeError:
                    pass
        return result
    except Exception:
        return {}


def load_unit_annots_from_sheets(annotator: str) -> dict:
    """Returns {(paper_id, source, unit_text): {validity, action, details, helpfulness}}"""
    ws, _ = _ensure_ws(_UNITS_SHEET, _UNITS_HEADER)
    if ws is None:
        return {}
    try:
        rows = ws.get_all_values()
        if not rows:
            return {}
        # Resolve column positions from header row
        header = rows[0]
        def ci(name: str, fallback: int) -> int:
            return header.index(name) if name in header else fallback
        i_ann  = ci("annotator", 1)
        i_pid  = ci("paper_id", 2)
        i_src  = ci("feedback_source", 3)
        i_unit = ci("feedback_unit", 4)
        i_val  = ci("validity", 5)
        i_act  = ci("action", 6)
        i_det  = ci("details", 7)
        i_help = ci("helpfulness", 8)

        result = {}
        for r in rows[1:]:
            if not r or len(r) <= max(i_ann, i_pid, i_src, i_unit, i_help):
                continue
            if r[i_ann].strip() != annotator:
                continue
            key = (r[i_pid].strip(), r[i_src].strip(), r[i_unit].strip())
            try:
                raw_h = r[i_help].strip()
                helpfulness = int(raw_h) if raw_h.isdigit() else None
            except (ValueError, IndexError):
                helpfulness = None
            result[key] = {
                "validity": r[i_val].strip() or None,
                "action": r[i_act].strip() or None,
                "details": r[i_det].strip() if len(r) > i_det else "",
                "helpfulness": helpfulness,
            }
        return result
    except Exception:
        return {}


# ── DATA ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_SETS_PATH = _ROOT / "data" / "fb_sets_comparison.csv"
_UNITS_PATH = _ROOT / "data" / "fb_units_eval.csv"


@st.cache_data(ttl=60)
def _load_sets(_mtime: float = 0) -> pd.DataFrame:
    return pd.read_csv(_SETS_PATH).reset_index(drop=True)


@st.cache_data(ttl=60)
def _load_units(_mtime: float = 0) -> pd.DataFrame:
    return pd.read_csv(_UNITS_PATH).reset_index(drop=True)


def _model_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feedback_set-")]


def _model_names(df: pd.DataFrame) -> list[str]:
    return [c.replace("feedback_set-", "") for c in _model_cols(df)]


def _get_assigned(df: pd.DataFrame, username: str) -> pd.DataFrame:
    def contains(s: str) -> bool:
        return username.lower() in [a.strip().lower() for a in str(s).split(",")]
    return df[df["annotators"].apply(contains)].reset_index(drop=True)


def _shuffled_models(annotator: str, models: list[str]) -> list[str]:
    """Deterministically shuffle model order per annotator to reduce position bias."""
    seed = int(hashlib.md5(annotator.lower().encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)
    order = list(range(len(models)))
    rng.shuffle(order)
    return [models[i] for i in order]


_SET_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


# ── SESSION INIT ──────────────────────────────────────────────────────────────
def _init():
    if "annotator" not in st.session_state:
        q = st.query_params.get("annotator", "")
        st.session_state.annotator = str(q).strip() if q else ""
    defaults = {
        "sets_nav": 0,
        "units_nav": 0,
        "rankings": {},        # {paper_id: [model_rank1, ...]}
        "unit_annots": {},     # {(paper_id, source, unit_text): {validity, action, details, helpfulness}}
        "sheets_loaded": False,
        "last_save_toast": None,  # {"ok": bool, "msg": str, "task": "ranking"|"unit"}
        "switch_to_tab2": False,  # trigger JS tab switch after rerun
        "_loaded_for_annotator": "",  # tracks which annotator the sheets data belongs to
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init()


def _clear_user_state() -> None:
    """Clear all per-annotator session state so a fresh user starts clean."""
    st.session_state.sheets_loaded = False
    st.session_state.rankings = {}
    st.session_state.unit_annots = {}
    st.session_state.sets_nav = 0
    st.session_state.units_nav = 0
    st.session_state._loaded_for_annotator = ""
    for key in list(st.session_state.keys()):
        if key.startswith("draft_") or key.startswith("rank_"):
            del st.session_state[key]


# If the annotator changed (e.g. via URL) without going through the logout flow,
# clear any stale annotation data from the previous user.
_current = st.session_state.annotator
if _current and st.session_state.get("_loaded_for_annotator", "") not in ("", _current):
    _clear_user_state()

# ── LOGIN GATE ────────────────────────────────────────────────────────────────
if not st.session_state.annotator:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("## 👤 Enter your first name to begin")
        st.caption("Your name is recorded with every annotation.")
        name = st.text_input("Name", placeholder="e.g. chani", label_visibility="collapsed")
        if st.button("Continue →", type="primary"):
            if name.strip():
                st.session_state.annotator = name.strip()
                st.query_params["annotator"] = name.strip()
                st.rerun()
            else:
                st.warning("Please enter your name.")
    st.stop()

annotator: str = st.session_state.annotator

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df_sets = _load_sets(_SETS_PATH.stat().st_mtime if _SETS_PATH.exists() else 0.0)
df_units = _load_units(_UNITS_PATH.stat().st_mtime if _UNITS_PATH.exists() else 0.0)

assigned_sets = _get_assigned(df_sets, annotator)
assigned_units = _get_assigned(df_units, annotator)

models = _model_names(df_sets)

# ── LOAD FROM GOOGLE SHEETS ONCE PER SESSION ─────────────────────────────────
if not st.session_state.sheets_loaded and _get_gc() is not None:
    loaded_r = load_rankings_from_sheets(annotator)
    st.session_state.rankings.update(loaded_r)
    loaded_u = load_unit_annots_from_sheets(annotator)
    st.session_state.unit_annots.update(loaded_u)
    # Jump to first unannotated unit
    paper_ids_done = set(st.session_state.rankings.keys())
    for i, row in assigned_sets.iterrows():
        if str(row["paper_id"]) not in paper_ids_done:
            st.session_state.sets_nav = i
            break
    for i, row in assigned_units.iterrows():
        key = (str(row["paper_id"]), str(row["feedback_source"]), str(row["feedback_unit"]).strip())
        if key not in st.session_state.unit_annots:
            st.session_state.units_nav = i
            break
    st.session_state.sheets_loaded = True
    st.session_state._loaded_for_annotator = annotator

# ── PROGRESS COUNTS ───────────────────────────────────────────────────────────
n_sets = len(assigned_sets)
n_units = len(assigned_units)
n_sets_done = sum(
    1 for _, r in assigned_sets.iterrows()
    if str(r["paper_id"]) in st.session_state.rankings
)
n_units_done = sum(
    1 for _, r in assigned_units.iterrows()
    if (str(r["paper_id"]), str(r["feedback_source"]), str(r["feedback_unit"]).strip())
       in st.session_state.unit_annots
)

# ── TOP BAR ───────────────────────────────────────────────────────────────────
col_bar, col_change = st.columns([8, 1])
with col_bar:
    st.markdown(f"""
    <div class="top-bar">
      <h1>⬡ Feedback Human Evaluation</h1>
      <span class="progress-info">
        <span style="color:var(--accent); margin-right:1rem;">👤 {html.escape(annotator)}</span>
        Task 1: {n_sets_done}/{n_sets}
        &nbsp;·&nbsp;
        Task 2: {n_units_done}/{n_units}
      </span>
    </div>
    """, unsafe_allow_html=True)
with col_change:
    if st.button("Change name"):
        _clear_user_state()
        st.session_state.annotator = ""
        st.query_params.pop("annotator", None)
        st.rerun()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "📋  Task 1 — Rank Feedback Sets",
    "🔍  Task 2 — Evaluate Feedback Units",
])

# Auto-switch to Tab 2 when triggered by "Save & Next" on the last Task 1 item
if st.session_state.get("switch_to_tab2"):
    st.session_state.switch_to_tab2 = False
    import streamlit.components.v1 as components
    components.html("""
    <script>
    setTimeout(function() {
        const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
        if (tabs.length > 1) tabs[1].click();
    }, 200);
    </script>
    """, height=0)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — RANKING
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if assigned_sets.empty:
        st.info(f"No papers are assigned to **{annotator}** for Task 1.")
        st.stop()

    nav1 = min(st.session_state.sets_nav, len(assigned_sets) - 1)
    srow = assigned_sets.iloc[nav1]
    paper_id1 = str(srow["paper_id"])
    title1 = str(srow.get("title", "") or "").strip()

    shuffled = _shuffled_models(annotator, models)
    labels = _SET_LABELS[: len(shuffled)]
    label_to_model = {labels[i]: shuffled[i] for i in range(len(shuffled))}
    model_to_label = {v: k for k, v in label_to_model.items()}

    # ── Navigation ────────────────────────────────────────────────────────────
    c_prev, c_pos, c_next = st.columns([2, 3, 2])
    with c_prev:
        if st.button("← Prev", disabled=(nav1 == 0), key="sets_prev"):
            st.session_state.sets_nav = nav1 - 1
            st.rerun()
    with c_pos:
        is_ranked = paper_id1 in st.session_state.rankings
        badge_html = '<span class="done-chip">✓ Ranked</span>' if is_ranked else '<span class="todo-chip">Not yet ranked</span>'
        st.markdown(
            f"<div class='nav-center'>Paper {nav1 + 1} / {len(assigned_sets)}<br>{badge_html}</div>",
            unsafe_allow_html=True,
        )
    with c_next:
        if st.button("Next →", disabled=(nav1 == len(assigned_sets) - 1), key="sets_next"):
            st.session_state.sets_nav = nav1 + 1
            st.rerun()

    st.markdown("---")

    # ── Paper info ────────────────────────────────────────────────────────────
    if title1:
        st.markdown(f"### {title1}")
    st.caption(f"`paper_id: {paper_id1}`")

    st.markdown("""
    **Instructions:** Read all feedback sets below, then assign a unique rank to each one.
    Evaluate based on three criteria: **validity** (is the feedback a valid issue/question?),
    **actionability** (can authors act on it?), and **helpfulness** (overall value to the authors).
    **Rank 1 = best, rank 5 = worst.**
    The sets are labeled A–E; the underlying model is hidden to avoid bias.
    """)

    st.markdown("---")

    # ── Feedback sets (nested tabs) ────────────────────────────────────────────
    st.markdown("#### 📄 Read Feedback Sets")
    inner_tabs = st.tabs([f"Set {lbl}" for lbl in labels])
    for inner_tab, lbl in zip(inner_tabs, labels):
        model = label_to_model[lbl]
        col_key = f"feedback_set-{model}"
        text = str(srow.get(col_key, "") or "").strip()
        with inner_tab:
            if text:
                st.markdown(text)
            else:
                st.caption("(no content)")

    st.markdown("---")

    # ── Ranking UI ────────────────────────────────────────────────────────────
    st.markdown("#### 🏆 Your Ranking")
    st.caption("Assign each set a unique rank. Duplicate ranks will be flagged.")

    existing_ranking = st.session_state.rankings.get(paper_id1, [])
    existing_model_ranks: dict[str, int] = {}
    for rank_i, mname in enumerate(existing_ranking, start=1):
        existing_model_ranks[mname] = rank_i

    rank_options: list = [None] + list(range(1, len(shuffled) + 1))
    rank_fmt = {None: "— select —", **{i: f"{i}{'  (best)' if i == 1 else '  (worst)' if i == len(shuffled) else ''}" for i in range(1, len(shuffled) + 1)}}

    draft_key = f"draft_{paper_id1}"
    if draft_key not in st.session_state:
        st.session_state[draft_key] = {m: existing_model_ranks.get(m) for m in shuffled}
        # Sync widget state keys so Streamlit doesn't use stale values from a previous session
        for _lbl, _model in label_to_model.items():
            st.session_state[f"rank_{paper_id1}_{_lbl}"] = existing_model_ranks.get(_model)

    rank_cols = st.columns(len(shuffled))
    assigned_ranks: dict[str, int] = {}
    for i, (lbl, model) in enumerate(label_to_model.items()):
        with rank_cols[i]:
            current_val = st.session_state[draft_key].get(model)
            idx = rank_options.index(current_val) if current_val in rank_options else 0
            chosen = st.selectbox(
                f"Set {lbl}",
                options=rank_options,
                format_func=lambda x, rf=rank_fmt: rf.get(x, "—"),
                index=idx,
                key=f"rank_{paper_id1}_{lbl}",
            )
            st.session_state[draft_key][model] = chosen
            if chosen is not None:
                assigned_ranks[model] = chosen

    all_filled = len(assigned_ranks) == len(shuffled)
    all_unique = len(set(assigned_ranks.values())) == len(assigned_ranks)

    if all_filled and not all_unique:
        st.warning("⚠️ Each rank must be unique. Adjust your ranking before submitting.")

    is_last_paper = (nav1 == len(assigned_sets) - 1)
    btn_label = "💾 Save & Go to Task 2 →" if is_last_paper else "💾 Save & Next →"

    save_c, status_c = st.columns([2, 6])
    with save_c:
        if st.button(
            btn_label,
            type="primary",
            disabled=(not all_filled or not all_unique),
            key="submit_ranking",
        ):
            sorted_models = [m for m, _ in sorted(assigned_ranks.items(), key=lambda x: x[1])]
            st.session_state.rankings[paper_id1] = sorted_models
            ok, err = save_ranking(annotator, paper_id1, sorted_models)
            if ok:
                st.session_state.last_save_toast = {"ok": True, "msg": "✅ Ranking saved to Google Sheets!", "task": "ranking"}
            elif err:
                st.session_state.last_save_toast = {"ok": False, "msg": f"❌ Save failed: {err}", "task": "ranking"}
            else:
                st.session_state.last_save_toast = {"ok": None, "msg": "💾 Ranking saved locally (Google Sheets not configured).", "task": "ranking"}
            if is_last_paper:
                st.session_state.units_nav = 0
                st.session_state.switch_to_tab2 = True
            else:
                st.session_state.sets_nav = nav1 + 1
            st.rerun()

    # ── Persistent save status banner ─────────────────────────────────────────
    toast = st.session_state.get("last_save_toast")
    if toast and toast.get("task") == "ranking":
        with status_c:
            if toast["ok"] is True:
                st.success(toast["msg"])
            elif toast["ok"] is False:
                st.error(toast["msg"])
            else:
                st.info(toast["msg"])

    # Show saved ranking summary
    if paper_id1 in st.session_state.rankings:
        saved = st.session_state.rankings[paper_id1]
        summary = " → ".join([f"**Set {model_to_label.get(m, m)}**" for m in saved])
        st.markdown(f"**Saved ranking (best → worst):** {summary}")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — UNIT ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if assigned_units.empty:
        st.info(f"No feedback units are assigned to **{annotator}** for Task 2.")
        st.stop()

    nav2 = min(st.session_state.units_nav, len(assigned_units) - 1)
    urow = assigned_units.iloc[nav2]
    paper_id2 = str(urow["paper_id"])
    title2 = str(urow.get("title", "") or "").strip()
    source2 = str(urow.get("feedback_source", "") or "").strip()
    unit_text2 = str(urow.get("feedback_unit", "") or "").strip()
    unit_key2 = (paper_id2, source2, unit_text2)
    existing2 = st.session_state.unit_annots.get(unit_key2, {})

    # ── Navigation ────────────────────────────────────────────────────────────
    c_prev2, c_pos2, c_next2 = st.columns([2, 3, 2])
    with c_prev2:
        if st.button("← Prev", disabled=(nav2 == 0), key="units_prev"):
            st.session_state.units_nav = nav2 - 1
            st.rerun()
    with c_pos2:
        is_done2 = unit_key2 in st.session_state.unit_annots
        badge2 = '<span class="done-chip">✓ Annotated</span>' if is_done2 else '<span class="todo-chip">Not yet annotated</span>'
        st.markdown(
            f"<div class='nav-center'>Unit {nav2 + 1} / {len(assigned_units)}<br>{badge2}</div>",
            unsafe_allow_html=True,
        )
    with c_next2:
        if st.button("Next →", disabled=(nav2 == len(assigned_units) - 1), key="units_next"):
            st.session_state.units_nav = nav2 + 1
            st.rerun()

    # Jump to first unannotated
    jump_c, _ = st.columns([2, 6])
    with jump_c:
        if st.button("⏭ Jump to first unannotated", type="primary", key="jump_unannotated"):
            for i, row in assigned_units.iterrows():
                k = (str(row["paper_id"]), str(row["feedback_source"]), str(row["feedback_unit"]).strip())
                if k not in st.session_state.unit_annots:
                    st.session_state.units_nav = int(i)
                    st.rerun()

    st.markdown("---")

    # ── Paper + unit ──────────────────────────────────────────────────────────
    if title2:
        st.markdown(f"### {title2}")
    st.caption(f"`paper_id: {paper_id2}`")

    st.markdown(f"""
    <div class="unit-card">
      <div class="unit-source-badge">Source: {html.escape(source2)}</div>
      <div class="unit-text">{html.escape(unit_text2)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 1. VALIDITY ───────────────────────────────────────────────────────────
    st.markdown("##### 1. Validity")
    st.caption("Do you agree that this feedback is a valid issue/question?")

    _validity_opts = ["agree", "disagree"]
    _validity_desc = {
        "agree": "You agree the point is valid.",
        "disagree": "You disagree with the premise or issue, or you think the reviewer is mistaken.",
    }
    _cur_validity = existing2.get("validity")
    _validity_idx = _validity_opts.index(_cur_validity) if _cur_validity in _validity_opts else None

    validity = st.radio(
        "Validity",
        options=_validity_opts,
        index=_validity_idx,
        format_func=lambda x: f"{x}  —  {_validity_desc[x]}",
        label_visibility="collapsed",
        key=f"validity_{nav2}",
    )

    st.markdown("---")

    # ── 2. ACTION ─────────────────────────────────────────────────────────────
    st.markdown("##### 2. Action")
    st.caption("What action are you willing to take in response to this feedback?")

    _action_opts = [
        "will_revise",
        "defer_future_work",
        "point_to_existing_content",
        "no_revision_accept",
        "no_revision_contest",
        "no_action_other",
    ]
    _action_desc = {
        "will_revise": "Make a concrete change to the manuscript (add text/experiments/citations, fix figures, restructure, etc.).",
        "defer_future_work": "Acknowledge the point is valid but defer it (future work, out of scope, resource constraints). No revision promised.",
        "point_to_existing_content": "The paper already addresses this; point to a specific section, appendix, figure, or table. No revision promised.",
        "no_revision_accept": "Acknowledge the point is valid but make no change and do not defer to future work.",
        "no_revision_contest": "Dispute or reject the feedback and make no change.",
        "no_action_other": "No action for a reason not captured above. Explain in 'Details' below.",
    }
    _cur_action = existing2.get("action")
    _action_idx = _action_opts.index(_cur_action) if _cur_action in _action_opts else None

    action = st.radio(
        "Action",
        options=_action_opts,
        index=_action_idx,
        format_func=lambda x: f"{x}  —  {_action_desc[x]}",
        label_visibility="collapsed",
        key=f"action_{nav2}",
    )

    details = st.text_area(
        "Details",
        value=existing2.get("details", ""),
        placeholder="Short description of the action or reason for no action",
        height=80,
        key=f"details_{nav2}",
        label_visibility="visible",
    )

    st.markdown("---")

    # ── 3. HELPFULNESS ────────────────────────────────────────────────────────
    st.markdown("##### 3. Helpfulness")
    st.caption("Rate the overall value of this feedback to the authors on a 5-point scale.")

    _help_opts = [5, 4, 3, 2, 1]
    _help_fmt = {
        1: "1 — Not helpful at all",
        2: "2 — Slightly helpful",
        3: "3 — Moderately helpful",
        4: "4 — Helpful",
        5: "5 — Very helpful",
    }
    _cur_help = existing2.get("helpfulness")
    _help_idx = (5 - _cur_help) if (_cur_help is not None and 1 <= _cur_help <= 5) else None

    helpfulness = st.radio(
        "Helpfulness",
        options=_help_opts,
        index=_help_idx,
        format_func=lambda x: _help_fmt[x],
        horizontal=True,
        label_visibility="collapsed",
        key=f"help_{nav2}",
    )

    st.markdown("---")

    # ── Persistent save status banner ─────────────────────────────────────────
    toast = st.session_state.get("last_save_toast")
    if toast and toast.get("task") == "unit":
        if toast["ok"] is True:
            st.success(toast["msg"])
        elif toast["ok"] is False:
            st.error(toast["msg"])
        else:
            st.info(toast["msg"])

    # ── Save & Next (primary action) ───────────────────────────────────────────
    can_save = validity is not None and action is not None and helpfulness is not None

    bc1, _ = st.columns([2, 8])
    with bc1:
        if st.button("💾 Save & Next →", type="primary", disabled=not can_save, key="save_next_unit"):
            annot = {"validity": validity, "action": action, "details": details, "helpfulness": helpfulness}
            st.session_state.unit_annots[unit_key2] = annot
            ok, err = save_unit_annotation(annotator, paper_id2, source2, unit_text2, validity, action, details, helpfulness)
            if ok:
                st.session_state.last_save_toast = {"ok": True, "msg": "✅ Saved to Google Sheets!", "task": "unit"}
            elif err:
                st.session_state.last_save_toast = {"ok": False, "msg": f"❌ Save failed: {err}", "task": "unit"}
            else:
                st.session_state.last_save_toast = {"ok": None, "msg": "💾 Saved locally (Google Sheets not configured).", "task": "unit"}
            st.session_state.units_nav = min(len(assigned_units) - 1, nav2 + 1)
            st.rerun()

    if not can_save:
        st.caption("⚠️ Complete all three sections (validity, action, helpfulness) to enable saving.")
