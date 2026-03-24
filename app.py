import html
import json
import logging
import hashlib
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import pandas as pd
import gspread
from gspread.exceptions import WorksheetNotFound
from google.oauth2 import service_account

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Human Evaluation of Paper Feedback",
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
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 0.8rem 1rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius);
    text-align: center;
    gap: 0.4rem;
}
.top-bar h1 {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.15rem;
    font-weight: 700; color: var(--accent); margin: 0;
}
.progress-info {
    font-family: 'IBM Plex Mono', monospace;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    gap: 0.45rem 1rem;
    font-size: 0.82rem;
    color: var(--text-dim);
}
/* Task 1/2 progress bars (top bar) */
.progress-bars-wrap {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    width: min(26rem, 100%);
    min-width: 12rem;
}
.task-progress-row {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}
.task-progress-label {
    flex: 0 0 4.75rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 800;
    color: #b91c1c;
    text-align: left;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    line-height: 1.2;
}
.progress-track {
    flex: 1;
    height: 12px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 7px;
    overflow: hidden;
    min-width: 0;
}
.progress-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #ef4444 0%, #b91c1c 100%);
    transition: width 0.4s ease;
    box-sizing: border-box;
}
.task-progress-count {
    flex: 0 0 auto;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    color: #b91c1c;
    min-width: 2.75rem;
    text-align: right;
    line-height: 1.2;
}

/* Feedback card (for sets display) */
.fb-card {
    background: var(--surface); border: 1.5px solid var(--border);
    border-radius: var(--radius); padding: 1rem;
    max-height: 520px; overflow-y: auto;
    font-size: 0.92rem;
    line-height: 1.5;
}
/* Custom scrollbar for feedback cards */
.fb-card::-webkit-scrollbar {
    width: 13px;
}
.fb-card::-webkit-scrollbar-track {
    background: var(--surface2);
    border-radius: 10px;
}
.fb-card::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 10px;
}
.fb-card::-webkit-scrollbar-thumb:hover {
    background: var(--text-dim);
}
.fb-line {
    margin-bottom: 0.65em;
}
.fb-line:last-child {
    margin-bottom: 0;
}
.fb-card-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; font-weight: 600;
    color: var(--accent); text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
}

/* Task 2: rank slots 1, 2, 3 — letter chosen per slot (outer border via st.container) */
.rank-slot-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    color: var(--text);
    margin-bottom: 0.15rem;
}
.rank-slot-hint {
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-bottom: 0.5rem;
    line-height: 1.3;
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

/* Centered section heading (Feedback Sets, Your Ranking) */
.sec-heading-center {
    text-align: center;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
    color: var(--text);
}

/* Nav center info */
.nav-center {
    text-align: center; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem; padding-top: 0.55rem; line-height: 1.4;
    color: var(--text-dim);
}
.nav-center strong {
    color: var(--text);
    font-weight: 700;
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

/* Paper title label — small text above the title */
.paper-title-label {
    text-align: center;
    font-size: 0.75rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
}

/* Paper title — Task 1 & 2: center, bolder */
.paper-title {
    text-align: center;
    font-weight: 700;
    font-size: 1.7rem;
    margin-bottom: 0.5rem;
    color: var(--text);
}

/* Current step banner — 2× paper-title scale; boxed for visibility */
.current-step-wrap {
    text-align: center;
    margin: 0 auto 1.1rem auto;
    max-width: 58rem;
    line-height: 1.25;
    background: var(--surface);
    border: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.35rem 1.75rem 1.45rem 1.75rem;
    box-shadow: 0 8px 28px rgba(37, 99, 235, 0.14);
}
.current-step-line {
    font-weight: 700;
    font-size: 1.7rem !important;
    color: var(--text);
    margin: 0;
    letter-spacing: -0.02em;
}
.current-step-oneway {
    font-size: 1.2rem !important;
    font-weight: 700;
    color: #dc2626;
    margin: 0.55rem 0 0 0;
    line-height: 1.35;
    text-shadow: 0 0 1px rgba(220, 38, 38, 0.15);
}

/* Instructions heading in red */
.instructions-label { color: #dc2626; font-weight: 600; }

/* Irreversible step warning (must stay red in light/dark) */
.instructions-irreversible {
    color: #dc2626;
    font-weight: 600;
    margin-top: 0.75rem;
    line-height: 1.55;
}

/* Completion / thank-you panel */
.ending-panel {
    text-align: center;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem 1.5rem 1.75rem;
    max-width: 36rem;
    margin: 0 auto 1rem;
}
.ending-panel h2 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--green);
    margin: 0 0 0.75rem 0;
}
.ending-panel p {
    color: var(--text);
    font-size: 1rem;
    line-height: 1.6;
    margin: 0 0 0.5rem 0;
}
.ending-stats {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-dim);
    margin-top: 0.75rem;
}

/* All tasks done — explicit “End annotation” CTA */
.end-annotation-wrap {
    max-width: 40rem;
    margin: 0 auto 1.1rem;
    padding: 1.1rem 1.25rem 1.25rem;
    text-align: center;
    background: linear-gradient(180deg, rgba(22,163,74,0.1) 0%, var(--surface) 55%);
    border: 2px solid var(--green);
    border-radius: var(--radius);
    box-shadow: 0 4px 18px rgba(22, 163, 74, 0.12);
}
.end-annotation-wrap .end-annotation-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--green);
    margin: 0 0 0.35rem 0;
    line-height: 1.35;
}
.end-annotation-wrap .end-annotation-sub {
    font-size: 0.95rem;
    color: var(--text-dim);
    margin: 0 0 0.85rem 0;
    line-height: 1.5;
}

/* Instructions block — larger font, visually separated from paper title */
.instructions-block {
    font-size: 1.08rem;
    line-height: 1.65;
    margin-top: 1.25rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)


# ── GOOGLE SHEETS ─────────────────────────────────────────────────────────────
_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
_RANKINGS_SHEET = "Rankings"
_UNITS_SHEET = "UnitAnnotations"
_RANKINGS_HEADER = [
    "annotator",
    "paper_id",
    "ranked_models",
    "distractor_start_num",
    "distractor_step_num",
    "distractor_user_answers",
    "distractor_is_correct",
    "distractor_completed_at",
    "timestamp",
]
# unit_hash = MD5(paper_id + source + unit_text) — used as a stable unique key for upsert
_UNITS_HEADER = [
    "unit_hash", "annotator", "paper_id", "feedback_source", "feedback_unit",
    "validity", "specificity", "action", "details", "helpfulness", "timestamp",
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


def save_ranking(
    annotator: str,
    paper_id: str,
    ranked_models: list,
    counting_payload: dict | None = None,
) -> tuple[bool, str | None]:
    """Returns (success, error_message)."""
    ws, err = _ensure_ws(_RANKINGS_SHEET, _RANKINGS_HEADER)
    if ws is None:
        return False, err
    try:
        ts = datetime.now(timezone.utc).isoformat()
        existing = ws.get_all_values()
        header = existing[0] if existing else _RANKINGS_HEADER
        col_map = {name: idx for idx, name in enumerate(header)}

        def _col(name: str) -> int:
            return col_map.get(name, -1)

        ann_idx = _col("annotator")
        pid_idx = _col("paper_id")
        if ann_idx < 0 or pid_idx < 0:
            return False, "Rankings sheet header is invalid: missing annotator/paper_id columns."

        def _build_row(prev: list[str] | None) -> list[str]:
            row = list(prev) if prev else []
            if len(row) < len(header):
                row.extend([""] * (len(header) - len(row)))

            row[ann_idx] = annotator
            row[pid_idx] = paper_id

            ranked_idx = _col("ranked_models")
            if ranked_idx >= 0:
                row[ranked_idx] = json.dumps(ranked_models)

            if counting_payload is not None:
                start_idx = _col("distractor_start_num")
                step_idx = _col("distractor_step_num")
                ans_idx = _col("distractor_user_answers")
                ok_idx = _col("distractor_is_correct")
                done_idx = _col("distractor_completed_at")
                if min(start_idx, step_idx, ans_idx, ok_idx, done_idx) < 0:
                    raise ValueError("Rankings sheet header is invalid: missing counting task columns.")
                row[start_idx] = str(counting_payload["start_num"])
                row[step_idx] = str(counting_payload["step_num"])
                row[ans_idx] = json.dumps(counting_payload["answers"])
                row[ok_idx] = "true" if counting_payload["is_correct"] else "false"
                row[done_idx] = counting_payload["completed_at"]

            ts_idx = _col("timestamp")
            if ts_idx >= 0:
                row[ts_idx] = ts
            return row

        for i, r in enumerate(existing[1:], start=2):
            if len(r) > max(ann_idx, pid_idx) and r[ann_idx].strip() == annotator and r[pid_idx].strip() == paper_id:
                ws.update([_build_row(r)], f"A{i}")
                return True, None

        ws.append_row(_build_row(None))
        return True, None
    except Exception as e:
        logging.error("save_ranking failed: %s", e)
        return False, str(e)


def save_unit_annotation(
    annotator: str, paper_id: str, source: str, unit_text: str,
    validity: str, specificity: int, action: str, details: str, helpfulness: int,
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
        row = [uhash, annotator, paper_id, source, unit_text, validity, str(specificity), action, details, str(helpfulness), ts]
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


def restore_counting_from_rankings_sheet(annotator: str) -> None:
    """Restore counting-task session fields from Rankings rows (survives page refresh)."""
    ws, _ = _ensure_ws(_RANKINGS_SHEET, _RANKINGS_HEADER)
    if ws is None:
        return
    try:
        rows = ws.get_all_values()
        if not rows or len(rows) < 2:
            return
        header = rows[0]

        def ci(name: str, fallback: int) -> int:
            return header.index(name) if name in header else fallback

        i_ann = ci("annotator", 0)
        i_start = ci("distractor_start_num", -1)
        i_step = ci("distractor_step_num", -1)
        i_ans = ci("distractor_user_answers", -1)
        i_ok = ci("distractor_is_correct", -1)
        i_done = ci("distractor_completed_at", -1)
        if i_done < 0 or i_ann < 0:
            return

        best_row: list[str] | None = None
        best_done = ""
        for r in rows[1:]:
            if not r or len(r) <= max(i_ann, i_done):
                continue
            if r[i_ann].strip() != annotator:
                continue
            done = (r[i_done] or "").strip()
            if not done:
                continue
            if done >= best_done:
                best_done = done
                best_row = r

        if best_row is None:
            return

        if i_start >= 0 and len(best_row) > i_start and (best_row[i_start] or "").strip():
            st.session_state.distractor_start_num = int(str(best_row[i_start]).strip())
        if i_step >= 0 and len(best_row) > i_step and (best_row[i_step] or "").strip():
            st.session_state.distractor_step_num = int(str(best_row[i_step]).strip())
        st.session_state.counting_completed_at = best_done

        if i_ok >= 0 and len(best_row) > i_ok:
            v = (best_row[i_ok] or "").strip().lower()
            if v in ("true", "false"):
                st.session_state.counting_is_correct = v == "true"

        if i_ans >= 0 and len(best_row) > i_ans:
            raw_a = (best_row[i_ans] or "").strip()
            if raw_a:
                try:
                    parsed = json.loads(raw_a)
                    if isinstance(parsed, list) and len(parsed) == 5:
                        st.session_state.counting_answers = [str(x) for x in parsed]
                except json.JSONDecodeError:
                    logging.warning("Could not parse distractor_user_answers for annotator=%s", annotator)
    except Exception as e:
        logging.warning("restore_counting_from_rankings_sheet failed: %s", e)


def apply_resume_navigation_after_sheet_load(
    assigned_units: pd.DataFrame,
    assigned_sets: pd.DataFrame,
    n_units: int,
    n_units_done: int,
    n_sets: int,
    n_sets_done: int,
) -> None:
    """Set units_nav / sets_nav once after loading Google Sheets (e.g. refresh)."""
    task1_done = assigned_units.empty or (n_units_done >= n_units)
    counting_done = bool(st.session_state.get("counting_completed_at"))
    task2_done = assigned_sets.empty or (n_sets_done >= n_sets)

    if not task1_done:
        pos = _first_unannotated_unit_pos(assigned_units)
        if pos is not None:
            st.session_state.units_nav = pos
        return

    if not counting_done:
        return

    if not task2_done:
        if n_sets > 0:
            for pos, (_, row) in enumerate(assigned_sets.iterrows()):
                if str(row["paper_id"]) not in st.session_state.rankings:
                    st.session_state.sets_nav = pos
                    break
        return

    # All annotation work done; keep user on Task 2 last paper for End annotation CTA
    if n_sets > 0:
        st.session_state.sets_nav = n_sets - 1


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
        i_spec = ci("specificity", 6)
        i_act  = ci("action", 7)
        i_det  = ci("details", 8)
        i_help = ci("helpfulness", 9)

        result = {}
        for r in rows[1:]:
            if not r or len(r) <= max(i_ann, i_pid, i_src, i_unit):
                continue
            if r[i_ann].strip() != annotator:
                continue
            key = (r[i_pid].strip(), r[i_src].strip(), r[i_unit].strip())
            try:
                raw_h = r[i_help].strip() if len(r) > i_help else ""
                helpfulness = int(raw_h) if raw_h.isdigit() else None
            except (ValueError, IndexError):
                helpfulness = None
            try:
                raw_s = r[i_spec].strip() if len(r) > i_spec else ""
                specificity = int(raw_s) if raw_s.isdigit() else None
            except (ValueError, IndexError):
                specificity = None
            result[key] = {
                "validity": r[i_val].strip() or None,
                "specificity": specificity,
                "action": r[i_act].strip() if len(r) > i_act else None or None,
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


def _annotator_display_name(username: str, assigned_sets: pd.DataFrame, assigned_units: pd.DataFrame) -> str:
    """Token from CSV `annotators` that matches username (case-insensitive), preserving original spelling."""
    u = username.strip().lower()
    if not u:
        return username.strip()
    for df in (assigned_sets, assigned_units):
        if df is None or df.empty or "annotators" not in df.columns:
            continue
        for _, row in df.iterrows():
            raw = str(row.get("annotators", "") or "")
            for part in raw.split(","):
                tok = part.strip()
                if tok and tok.lower() == u:
                    return tok
    return username.strip()


def _first_unannotated_unit_pos(assigned_units: pd.DataFrame) -> int | None:
    """Row index in assigned_units for the first unit without a saved annotation, or None if all done."""
    if assigned_units.empty:
        return None
    for pos, (_, row) in enumerate(assigned_units.iterrows()):
        k = (str(row["paper_id"]), str(row["feedback_source"]), str(row["feedback_unit"]).strip())
        if k not in st.session_state.unit_annots:
            return pos
    return None


def _shuffled_models(annotator: str, models: list[str]) -> list[str]:
    """Deterministically shuffle model order per annotator to reduce position bias."""
    seed = int(hashlib.md5(annotator.lower().encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)
    order = list(range(len(models)))
    rng.shuffle(order)
    return [models[i] for i in order]


def _shuffle_task1_units(df: pd.DataFrame, annotator: str) -> pd.DataFrame:
    """Shuffle Task 1 unit rows with a deterministic seed per annotator (stable across refreshes)."""
    if df.empty:
        return df.reset_index(drop=True)
    seed = int(hashlib.md5(f"task1_units::{annotator.lower()}".encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)
    order = list(range(len(df)))
    rng.shuffle(order)
    return df.iloc[order].reset_index(drop=True)


def _format_feedback_text(text: str) -> str:
    """Render plain feedback text as safe HTML with explicit line spacing."""
    safe = html.escape(text).strip()
    if not safe:
        return ""
    lines = safe.split("\n")
    return "".join(
        f"<div class='fb-line'>{line if line.strip() else '&nbsp;'}</div>"
        for line in lines
    )


def _expected_counting_answers(start_num: int, step_num: int) -> list[int]:
    return [start_num - (step_num * i) for i in range(5)]


def _sync_task_flow_phase(assigned_units: pd.DataFrame, n_units: int, n_units_done: int) -> None:
    """Linear flow: Task 1 → Counting → Task 2. No backward navigation between tasks."""
    counting_done = bool(st.session_state.get("counting_completed_at"))
    task1_done = assigned_units.empty or (n_units_done >= n_units)
    p = int(st.session_state.get("task_flow_phase", 1))
    if not task1_done:
        st.session_state.task_flow_phase = 1
        return
    if not counting_done:
        if assigned_units.empty:
            st.session_state.task_flow_phase = 2
        else:
            if p > 2:
                p = 2
            st.session_state.task_flow_phase = min(max(p, 1), 2)
        return
    st.session_state.task_flow_phase = 3


_SET_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _task1_read_unit_form_from_session(nav_idx: int) -> tuple[dict | None, bool]:
    """Read Task 1 widgets from session_state (nav row is above widgets; use keys for nav_idx)."""
    validity = st.session_state.get(f"validity_{nav_idx}")
    specificity = st.session_state.get(f"spec_{nav_idx}")
    action = st.session_state.get(f"action_{nav_idx}")
    details = st.session_state.get(f"details_{nav_idx}", "")
    if not isinstance(details, str):
        details = str(details)
    helpfulness = st.session_state.get(f"help_{nav_idx}")
    can_save_basic = (
        validity is not None and specificity is not None and action is not None and helpfulness is not None
    )
    details_needed = action == "no_action_other" and not details.strip()
    can_save = can_save_basic and not details_needed
    if not can_save:
        return None, False
    return {
        "validity": validity,
        "specificity": specificity,
        "action": action,
        "details": details,
        "helpfulness": helpfulness,
    }, True


def _task1_persist_if_complete(annotator: str, nav_idx: int, assigned_units: pd.DataFrame) -> bool:
    """If form at nav_idx is complete, upsert to Sheets. Returns True if caller may change units_nav.
    On sheet error, sets toast, calls st.rerun(), and returns False."""
    annot, can_save = _task1_read_unit_form_from_session(nav_idx)
    if not can_save or annot is None:
        return True
    urow = assigned_units.iloc[nav_idx]
    paper_id2 = str(urow["paper_id"])
    source2 = str(urow.get("feedback_source", "") or "").strip()
    unit_text2 = str(urow.get("feedback_unit", "") or "").strip()
    unit_key2 = (paper_id2, source2, unit_text2)
    st.session_state.unit_annots[unit_key2] = {
        "validity": annot["validity"],
        "specificity": annot["specificity"],
        "action": annot["action"],
        "details": annot["details"],
        "helpfulness": annot["helpfulness"],
    }
    ok, err = save_unit_annotation(
        annotator,
        paper_id2,
        source2,
        unit_text2,
        annot["validity"],
        annot["specificity"],
        annot["action"],
        annot["details"],
        annot["helpfulness"],
    )
    if ok:
        st.session_state.last_save_toast = {
            "ok": True,
            "msg": "✅ Response saved!",
            "task": "unit",
            "unit_key": unit_key2,
        }
    elif err:
        st.session_state.last_save_toast = {
            "ok": False,
            "msg": f"❌ Save failed: {err}",
            "task": "unit",
            "unit_key": unit_key2,
        }
    else:
        st.session_state.last_save_toast = {
            "ok": None,
            "msg": "💾 Saved locally (Google Sheets not configured).",
            "task": "unit",
            "unit_key": unit_key2,
        }
    if ok or err is None:
        return True
    st.rerun()
    return False


def _ranking_counting_payload_for_save() -> tuple[dict | None, str | None]:
    """Build counting_payload for save_ranking when counting finished; else None. Error string if state invalid."""
    if not st.session_state.get("counting_completed_at"):
        return None, None
    raw_answers = st.session_state.get("counting_answers", [])
    if len(raw_answers) != 5:
        return None, "❌ Counting task state is invalid: expected exactly five answers."
    if not all(re.fullmatch(r"-?\d+", str(v)) for v in raw_answers):
        return None, "❌ Counting task state is invalid: all five answers must be integers."
    counting_payload = {
        "start_num": int(st.session_state.get("distractor_start_num")),
        "step_num": int(st.session_state.get("distractor_step_num")),
        "answers": [int(v) for v in raw_answers],
        "is_correct": bool(st.session_state.get("counting_is_correct")),
        "completed_at": st.session_state.get("counting_completed_at"),
    }
    return counting_payload, None


def _task2_read_rank_labels_from_session(paper_id: str, n_ranks: int) -> list:
    return [st.session_state.get(f"rankpos_{paper_id}_{rank_num}") for rank_num in range(1, n_ranks + 1)]


def _task2_persist_ranking_if_complete(
    annotator: str,
    paper_id1: str,
    label_to_model: dict,
    labels_in_order: list,
) -> bool:
    """If ranking draft is complete and valid, save to Sheets. True = caller may change sets_nav."""
    all_filled = all(x is not None for x in labels_in_order)
    all_unique = len(set(labels_in_order)) == len(labels_in_order) if all_filled else False
    if not all_filled or not all_unique:
        return True
    sorted_models = [label_to_model[lb] for lb in labels_in_order]
    st.session_state.rankings[paper_id1] = sorted_models
    counting_payload, count_err = _ranking_counting_payload_for_save()
    if count_err:
        st.session_state.last_save_toast = {"ok": False, "msg": count_err, "task": "ranking"}
        st.rerun()
        return False
    ok, err = save_ranking(annotator, paper_id1, sorted_models, counting_payload=counting_payload)
    if ok:
        st.session_state.last_save_toast = {"ok": True, "msg": "✅ Ranking saved!", "task": "ranking"}
    elif err:
        st.session_state.last_save_toast = {"ok": False, "msg": f"❌ Save failed: {err}", "task": "ranking"}
    else:
        st.session_state.last_save_toast = {
            "ok": None,
            "msg": "💾 Ranking saved locally (Google Sheets not configured).",
            "task": "ranking",
        }
    if ok or err is None:
        return True
    st.rerun()
    return False


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
        "last_save_toast": None,  # {"ok", "msg", "task": "ranking"|"unit"; unit task also "unit_key"}
        "switch_to_tab2": False,  # legacy (tabs removed; kept for safety)
        "switch_to_tab_distractor": False,
        "switch_to_tab1": False,
        "task_flow_phase": 1,  # 1 = Task 1, 2 = Counting, 3 = Task 2 (linear, no backward)
        "_loaded_for_annotator": "",  # tracks which annotator the sheets data belongs to
        "show_completion_page": False,  # True after user clicks "End annotation" (thank-you page)
        "distractor_start_num": random.randint(100, 500),
        "distractor_step_num": random.choice([6,7]),
        "counting_answers": ["", "", "", "", ""],
        "counting_is_correct": None,
        "counting_completed_at": "",
        "task2_final_save_done": False,  # last-paper Save & Next pressed when all work complete (Task 2)
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
    st.session_state.show_completion_page = False
    st.session_state.switch_to_tab2 = False
    st.session_state.switch_to_tab_distractor = False
    st.session_state.switch_to_tab1 = False
    st.session_state.task_flow_phase = 1
    st.session_state.counting_answers = ["", "", "", "", ""]
    st.session_state.counting_is_correct = None
    st.session_state.counting_completed_at = ""
    st.session_state.task2_final_save_done = False
    for key in list(st.session_state.keys()):
        if key.startswith(("draft_", "rank_", "rankpos_")):
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
        st.markdown("## 👤 Enter your NAME that you entered in the form to begin")
        name = st.text_input("Name", placeholder="e.g. Chani Jung", label_visibility="collapsed")
        if st.button("Continue →", type="primary"):
            if name.strip():
                cand = name.strip()
                _df_s = _load_sets(_SETS_PATH.stat().st_mtime if _SETS_PATH.exists() else 0.0)
                _df_u = _load_units(_UNITS_PATH.stat().st_mtime if _UNITS_PATH.exists() else 0.0)
                _assigned_s = _get_assigned(_df_s, cand)
                _assigned_u = _get_assigned(_df_u, cand)
                if _assigned_s.empty and _assigned_u.empty:
                    st.error("Wrong annotator name. Try with another name.")
                else:
                    st.session_state.annotator = cand
                    st.query_params["annotator"] = cand
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
assigned_units = _shuffle_task1_units(assigned_units, annotator)

if annotator and assigned_sets.empty and assigned_units.empty:
    _, mid_bad, _ = st.columns([1, 2, 1])
    with mid_bad:
        st.error("Wrong annotator name. Try with another name.")
        if st.button("Try again", key="invalid_annotator_clear", type="primary"):
            st.session_state.annotator = ""
            st.query_params.pop("annotator", None)
            _clear_user_state()
            st.rerun()
    st.stop()

annotator_display = _annotator_display_name(annotator, assigned_sets, assigned_units)

models = _model_names(df_sets)

# ── LOAD FROM GOOGLE SHEETS ONCE PER SESSION ─────────────────────────────────
if not st.session_state.sheets_loaded and _get_gc() is not None:
    loaded_r = load_rankings_from_sheets(annotator)
    st.session_state.rankings.update(loaded_r)
    loaded_u = load_unit_annots_from_sheets(annotator)
    st.session_state.unit_annots.update(loaded_u)
    restore_counting_from_rankings_sheet(annotator)
    st.session_state.sheets_loaded = True
    st.session_state._loaded_for_annotator = annotator
    st.session_state._should_resume_nav = True

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

# Rankings sheet rows can carry distractor_* from a prior session. If Task 1 is not
# finished yet, that restored state must not make _sync_task_flow_phase jump to Task 2
# (skipping the counting step) right after the final unit save.
if n_units > 0 and n_units_done < n_units and bool(st.session_state.get("counting_completed_at")):
    st.session_state.counting_completed_at = ""
    st.session_state.counting_is_correct = None
    st.session_state.counting_answers = ["", "", "", "", ""]

if st.session_state.pop("_should_resume_nav", False):
    apply_resume_navigation_after_sheet_load(
        assigned_units, assigned_sets, n_units, n_units_done, n_sets, n_sets_done,
    )

counting_complete = bool(st.session_state.get("counting_completed_at"))
task1_complete = assigned_units.empty or (n_units_done >= n_units)
task2_complete = assigned_sets.empty or (n_sets_done >= n_sets)
all_annotations_complete = task1_complete and counting_complete and task2_complete
has_assigned_work = (n_units > 0) or (n_sets > 0)


def _maybe_render_end_annotation_banner_only(end_only_ui: bool = False) -> None:
    """Banner when all items are done; End annotation beside Save & Next, or alone after final save."""
    if not (
        has_assigned_work
        and all_annotations_complete
        and not st.session_state.get("show_completion_page", False)
    ):
        return
    sub = (
        "Click <strong>End annotation</strong> below to confirm you are finished."
        if end_only_ui
        else "Click <strong>End annotation</strong> to confirm you are finished."
    )
    st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
    _, cta_mid, _ = st.columns([1, 3, 1], gap="small")
    with cta_mid:
        st.markdown(
            f"""
            <div class="end-annotation-wrap">
              <p class="end-annotation-title">✓ You have completed all assigned Task 1 and Task 2 items</p>
              <p class="end-annotation-sub">{sub}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _maybe_render_end_annotation_cta() -> None:
    """Full-width End annotation when Task 2 has no papers (no Save row to pair with)."""
    if not (
        has_assigned_work
        and all_annotations_complete
        and not st.session_state.get("show_completion_page", False)
    ):
        return
    st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
    _, cta_mid, _ = st.columns([1, 3, 1], gap="small")
    with cta_mid:
        st.markdown(
            """
            <div class="end-annotation-wrap">
              <p class="end-annotation-title">✓ You have completed all assigned Task 1 and Task 2 items</p>
              <p class="end-annotation-sub">Click <strong>End annotation</strong> below to confirm you are finished.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "End annotation",
            type="primary",
            use_container_width=True,
            key="btn_end_annotation",
        ):
            st.session_state.show_completion_page = True
            st.rerun()


_sync_task_flow_phase(assigned_units, n_units, n_units_done)
task_flow_phase = int(st.session_state.get("task_flow_phase", 1))

# ── TOP BAR ───────────────────────────────────────────────────────────────────
_t1_pct = min(100.0, (100.0 * n_units_done / n_units)) if n_units > 0 else 100.0
_t2_pct = min(100.0, (100.0 * n_sets_done / n_sets)) if n_sets > 0 else 100.0
_t1_title = html.escape(f"Task 1: {n_units_done}/{n_units} units")
_t2_title = html.escape(f"Task 2: {n_sets_done}/{n_sets} papers")

col_l, col_bar, col_r = st.columns([1.2, 8, 1.2], vertical_alignment="center")
with col_bar:
    st.markdown(f"""
    <div class="top-bar">
      <h1>Human Evaluation of Paper Feedback</h1>
      <span class="progress-info">
        <span style="color:var(--accent);">👤 {html.escape(annotator_display)}</span>
        <div class="progress-bars-wrap">
          <div class="task-progress-row">
            <span class="task-progress-label">Task 1</span>
            <div class="progress-track" title="{_t1_title}" role="progressbar" aria-valuenow="{n_units_done}" aria-valuemin="0" aria-valuemax="{n_units}">
              <div class="progress-fill" style="width: {_t1_pct:.1f}%;"></div>
            </div>
            <span class="task-progress-count">{n_units_done}/{n_units}</span>
          </div>
          <div class="task-progress-row">
            <span class="task-progress-label">Task 2</span>
            <div class="progress-track" title="{_t2_title}" role="progressbar" aria-valuenow="{n_sets_done}" aria-valuemin="0" aria-valuemax="{n_sets}">
              <div class="progress-fill" style="width: {_t2_pct:.1f}%;"></div>
            </div>
            <span class="task-progress-count">{n_sets_done}/{n_sets}</span>
          </div>
        </div>
      </span>
    </div>
    """, unsafe_allow_html=True)
with col_r:
    if st.button("Change name", use_container_width=True):
        _clear_user_state()
        st.session_state.annotator = ""
        st.query_params.pop("annotator", None)
        st.rerun()

# Space after top bar
st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

# ── THANK-YOU PAGE (only after user clicks "End annotation") ────────────────
if (
    has_assigned_work
    and all_annotations_complete
    and st.session_state.get("show_completion_page", False)
):
    _, end_col, _ = st.columns([1, 2, 1])
    with end_col:
        st.markdown(
            f"""
            <div class="ending-panel">
              <h2>Thank you — you are done</h2>
              <p>All assigned work for <strong>Task 1</strong> and <strong>Task 2</strong> is complete.</p>
              <p class="ending-stats">
                Task 1: {n_units_done}/{n_units} units · Task 2: {n_sets_done}/{n_sets} papers
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("← Return to Task 2", use_container_width=True, key="btn_return_annotation"):
            st.session_state.show_completion_page = False
            # Thank-you page is only shown when all work is done; always land on Task 2 (not Task 1 / Counting).
            st.session_state.task_flow_phase = 3
            if n_sets > 0:
                st.session_state.sets_nav = n_sets - 1
            st.rerun()
    st.stop()

# ── LINEAR TASK FLOW (no tabs; cannot return to earlier tasks) ────────────────
_phase_label = {
    1: "Task 1 — Evaluate Feedback Units",
    2: "Counting Task",
    3: "Task 2 — Rank Feedback Sets",
}.get(task_flow_phase, f"Unknown ({task_flow_phase})")
_phase_safe = html.escape(str(_phase_label))
st.markdown(
    f"""
    <div class="current-step-wrap">
      <div class="current-step-line">{_phase_safe}</div>
      <div class="current-step-oneway">You cannot proceed to next task until this task is complete</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — UNIT ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════
if task_flow_phase == 1:
    if assigned_units.empty:
        st.info(f"No feedback units are assigned to **{annotator}** for Task 1.")
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
        if st.button("← Prev", disabled=(nav2 == 0), key="units_prev", use_container_width=True):
            if _task1_persist_if_complete(annotator, nav2, assigned_units):
                st.session_state.units_nav = nav2 - 1
                st.rerun()
    with c_pos2:
        is_done2 = unit_key2 in st.session_state.unit_annots
        badge2 = '<span class="done-chip">✓ Annotated</span>' if is_done2 else '<span class="todo-chip">Not yet annotated</span>'
        st.markdown(
            f"<div class='nav-center'><strong>Unit {nav2 + 1} / {len(assigned_units)}</strong><br>{badge2}</div>",
            unsafe_allow_html=True,
        )

        # Centered Go to # UI
        st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)
        gc1, gc2, gc3 = st.columns([1.5, 1, 0.9])
        with gc1:
            st.markdown("<div style='text-align:right; font-size:0.85rem; padding-top:0.45rem;'>Go to #</div>", unsafe_allow_html=True)
        with gc2:
            goto_input = st.number_input(
                "unit#", min_value=1, max_value=len(assigned_units),
                value=nav2 + 1, step=1, label_visibility="collapsed", key="goto_unit_num",
            )
        with gc3:
            if st.button("Go", key="goto_unit_btn", use_container_width=True):
                if _task1_persist_if_complete(annotator, nav2, assigned_units):
                    st.session_state.units_nav = int(goto_input) - 1
                    st.rerun()

    with c_next2:
        if st.button(
            "Next →",
            disabled=(nav2 == len(assigned_units) - 1),
            key="units_next",
            use_container_width=True,
        ):
            if _task1_persist_if_complete(annotator, nav2, assigned_units):
                st.session_state.units_nav = nav2 + 1
                st.rerun()

    if nav2 == len(assigned_units) - 1:
        st.info(
            "You are on the **last** unit. **Next** does not go to the next task. "
            "Use **Save & Next** below to save. When Task 1 is complete, use **Go to next task** to continue."
        )

    # Jump to first unannotated (left aligned or in a separate row)
    c_jump, _ = st.columns([2.5, 4.5])
    with c_jump:
        if st.button("⏭ Jump to first unannotated", type="primary", key="jump_unannotated", use_container_width=True):
            pos = _first_unannotated_unit_pos(assigned_units)
            if pos is not None and _task1_persist_if_complete(annotator, nav2, assigned_units):
                st.session_state.units_nav = pos
                st.rerun()

    st.markdown("---")

    # ── Paper + unit ──────────────────────────────────────────────────────────
    if title2:
        st.markdown(
            f"<div class='paper-title-label'>Paper Title</div><div class='paper-title'>{html.escape(title2)}</div>",
            unsafe_allow_html=True,
        )
    # st.caption(f"`paper_id: {paper_id2}`")

    st.markdown("""
    <div class="instructions-block">
    <span class="instructions-label">📌 Instructions:</span> Below is a piece of <strong>feedback on your paper</strong>. <strong>Evaluate</strong> it using the four sections that follow: <strong>validity</strong>, <strong>specificity</strong>, <strong>action</strong>, and <strong>helpfulness</strong>.
    <p class="instructions-irreversible">
    You can move between items within this task using the navigation controls. However, once you move to the next task, you cannot return to an earlier task.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(f"""
    <div class="unit-card">
      <div class="unit-text">{html.escape(unit_text2)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── QUESTIONS ─────────────────────────────────────────────────────────────
    q_col1, q_col2, q_col3, q_col4 = st.columns([1, 1, 1.2, 1], gap="medium")

    with q_col1:
        # ── 1. VALIDITY ───────────────────────────────────────────────────────────
        st.markdown("##### 1. Validity")
        st.caption("Do you agree that this feedback is a valid issue/question/suggestion?")

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

    with q_col2:
        # ── 2. SPECIFICITY ────────────────────────────────────────────────────────
        st.markdown("##### 2. Specificity")
        st.caption("Is the feedback anchored to specific parts of the paper?")

        _spec_opts = [5, 4, 3, 2, 1]
        _spec_fmt = {
            1: "1 — Very vague",
            2: "2 — Mostly vague",
            3: "3 — Moderately specific",
            4: "4 — Mostly specific",
            5: "5 — Very specific",
        }
        _cur_spec = existing2.get("specificity")
        _spec_idx = (5 - _cur_spec) if (_cur_spec is not None and 1 <= _cur_spec <= 5) else None

        specificity = st.radio(
            "Specificity",
            options=_spec_opts,
            index=_spec_idx,
            format_func=lambda x: _spec_fmt[x],
            label_visibility="collapsed",
            key=f"spec_{nav2}",
        )

    with q_col3:
        # ── 3. ACTION ─────────────────────────────────────────────────────────────
        st.markdown("##### 3. Action")
        st.caption("What action are you willing to take?")

        _action_opts = [
            "will_revise",
            "defer_future_work",
            "point_to_existing_content",
            "no_revision_accept",
            "no_revision_contest",
            "no_action_other",
        ]
        _action_desc = {
            "will_revise": "Make a concrete change to the manuscript.",
            "defer_future_work": "Acknowledge but defer (future work/out of scope).",
            "point_to_existing_content": "Already addresses this; point to section/table.",
            "no_revision_accept": "Valid but make no change/no deferral.",
            "no_revision_contest": "Dispute or reject and make no change.",
            "no_action_other": "No action for another reason (please specify in Details below).",
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
            placeholder="Short description of the action or reason for no action (required for 'another reason')",
            height=80,
            key=f"details_{nav2}",
            label_visibility="visible",
        )

    with q_col4:
        # ── 4. HELPFULNESS ────────────────────────────────────────────────────────
        st.markdown("##### 4. Helpfulness")
        st.caption("How useful is the feedback overall to the authors?")

        _help_opts = [5, 4, 3, 2, 1]
        _help_fmt = {
            1: "1 — Not helpful",
            2: "2 — Slightly",
            3: "3 — Moderately",
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
            label_visibility="collapsed",
            key=f"help_{nav2}",
        )

    st.markdown("---")

    # ── Save status banner (only for the unit it was triggered on) ─────────────
    toast = st.session_state.get("last_save_toast")
    if toast and toast.get("task") == "unit" and toast.get("unit_key") == unit_key2:
        if toast["ok"] is True:
            st.success(toast["msg"])
        elif toast["ok"] is False:
            st.error(toast["msg"])
        else:
            st.info(toast["msg"])

    # ── Save & Next (primary action) ───────────────────────────────────────────
    can_save_basic = validity is not None and specificity is not None and action is not None and helpfulness is not None
    details_needed = action == "no_action_other" and not details.strip()
    can_save = can_save_basic and not details_needed

    is_last_unit = (nav2 == len(assigned_units) - 1)
    btn_label = "💾 Save & Next →"

    def _commit_unit_save_next() -> None:
        annot = {"validity": validity, "specificity": specificity, "action": action, "details": details, "helpfulness": helpfulness}
        st.session_state.unit_annots[unit_key2] = annot
        ok, err = save_unit_annotation(annotator, paper_id2, source2, unit_text2, validity, specificity, action, details, helpfulness)
        if ok:
            st.session_state.last_save_toast = {
                "ok": True,
                "msg": "✅ Response saved!",
                "task": "unit",
                "unit_key": unit_key2,
            }
        elif err:
            st.session_state.last_save_toast = {
                "ok": False,
                "msg": f"❌ Save failed: {err}",
                "task": "unit",
                "unit_key": unit_key2,
            }
        else:
            st.session_state.last_save_toast = {
                "ok": None,
                "msg": "💾 Saved locally (Google Sheets not configured).",
                "task": "unit",
                "unit_key": unit_key2,
            }
        if ok or err is None:
            if is_last_unit:
                st.session_state.units_nav = nav2
            else:
                st.session_state.units_nav = nav2 + 1
        st.rerun()

    if task1_complete and not assigned_units.empty:
        st.markdown(
            """
            <p class="instructions-irreversible">
            Once you go to the next task, you cannot return to Task 1. Make sure you are finished before continuing.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.success("Task 1 is complete. Use **Go to next task** beside Save & Next when you are ready.")

    if task1_complete and not assigned_units.empty:
        _pair, _ = st.columns([4.2, 5.8], gap="small")
        with _pair:
            bc_save, bc_next_task = st.columns(2, gap="small")
            with bc_save:
                if st.button(btn_label, type="primary", disabled=not can_save, key="save_next_unit"):
                    _commit_unit_save_next()
            with bc_next_task:
                if st.button("Go to next task →", type="primary", key="go_to_next_task_from_task1", use_container_width=True):
                    st.session_state.sets_nav = 0
                    st.session_state.task_flow_phase = 2
                    st.rerun()
    else:
        bc_save, _ = st.columns([2, 8])
        with bc_save:
            if st.button(btn_label, type="primary", disabled=not can_save, key="save_next_unit"):
                _commit_unit_save_next()

    if not can_save:
        if details_needed:
            st.warning("⚠️ Please provide reason in 'Details' for selecting 'no_action_other'.")
        else:
            st.caption("⚠️ Complete all sections (validity, specificity, action, helpfulness) to enable saving.")


# ══════════════════════════════════════════════════════════════════════════════
# COUNTING TASK
# ══════════════════════════════════════════════════════════════════════════════
elif task_flow_phase == 2:
    start_num = int(st.session_state.get("distractor_start_num", random.randint(100, 500)))
    step_num = int(st.session_state.get("distractor_step_num", random.choice([3, 4])))
    expected_answers = _expected_counting_answers(start_num, step_num)
    st.markdown(
        f"""
        <div class="instructions-block">
        <span class="instructions-label">📌 Instructions:</span>
        Write down five numbers starting from <strong>{start_num}</strong> counting backward by <strong>{step_num}</strong>.
        <p class="instructions-irreversible">
        After you continue to Task 2, you cannot return to Task 1 or this step.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_cols = st.columns(5)
    answer_inputs: list[str] = []
    for idx, col in enumerate(input_cols):
        key = f"counting_answer_{idx}"
        if key not in st.session_state:
            st.session_state[key] = st.session_state.counting_answers[idx]
        with col:
            value = st.text_input(
                f"Number {idx + 1}",
                key=key
            ).strip()
            answer_inputs.append(value)

    st.session_state.counting_answers = answer_inputs
    all_filled = all(v != "" for v in answer_inputs)
    st.markdown(
        """
        <p class="instructions-irreversible" style="margin-bottom:0.65rem;">
        Continuing to Task 2 is one-way: you will not be able to go back to earlier tasks.
        </p>
        """,
        unsafe_allow_html=True,
    )
    c_mid_btn, c_mid_msg = st.columns([2, 6], gap="small")
    with c_mid_btn:
        if st.button(
            "Continue to Task 2 →",
            type="primary",
            key="go_task2_from_distractor",
            disabled=not all_filled,
        ):
            parsed_answers: list[int] = []
            bad_idx = -1
            for idx, raw in enumerate(answer_inputs):
                if not re.fullmatch(r"-?\d+", raw):
                    bad_idx = idx
                    break
                parsed_answers.append(int(raw))

            if bad_idx >= 0:
                st.session_state.last_save_toast = {
                    "ok": False,
                    "msg": f"❌ Number {bad_idx + 1} must be an integer.",
                    "task": "counting",
                }
                st.rerun()

            is_correct = parsed_answers == expected_answers
            completed_at = datetime.now(timezone.utc).isoformat()
            st.session_state.counting_is_correct = is_correct
            st.session_state.counting_completed_at = completed_at

            counting_paper_id = "__counting_task__"
            if not assigned_sets.empty:
                nav_counting = min(st.session_state.sets_nav, len(assigned_sets) - 1)
                counting_paper_id = str(assigned_sets.iloc[nav_counting]["paper_id"])

            existing_rank = st.session_state.rankings.get(counting_paper_id, [])
            counting_payload = {
                "start_num": start_num,
                "step_num": step_num,
                "answers": parsed_answers,
                "is_correct": is_correct,
                "completed_at": completed_at,
            }
            ok, err = save_ranking(
                annotator,
                counting_paper_id,
                existing_rank,
                counting_payload=counting_payload,
            )
            if not ok:
                st.session_state.last_save_toast = {
                    "ok": False,
                    "msg": f"❌ Counting task save failed: {err}",
                    "task": "counting",
                }
                st.rerun()

            st.session_state.last_save_toast = {
                "ok": True,
                "msg": "✅ Counting task recorded.",
                "task": "counting",
            }
            st.session_state.task_flow_phase = 3
            st.rerun()
    with c_mid_msg:
        if not all_filled:
            st.caption("⚠️ Fill in all five numbers to continue.")
        toast = st.session_state.get("last_save_toast")
        if toast and toast.get("task") == "counting":
            if toast["ok"] is True:
                st.success(toast["msg"])
            else:
                st.error(toast["msg"])


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — RANKING
# ══════════════════════════════════════════════════════════════════════════════
elif task_flow_phase == 3:
    if assigned_sets.empty:
        st.info(f"No papers are assigned to **{annotator}** for Task 2.")
        _maybe_render_end_annotation_cta()
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
        if st.button("← Prev", disabled=(nav1 == 0), key="sets_prev", use_container_width=True):
            lbls = _task2_read_rank_labels_from_session(paper_id1, len(shuffled))
            if _task2_persist_ranking_if_complete(annotator, paper_id1, label_to_model, lbls):
                st.session_state.sets_nav = nav1 - 1
                st.rerun()
    with c_pos:
        is_ranked = paper_id1 in st.session_state.rankings
        badge_html = '<span class="done-chip">✓ Ranked</span>' if is_ranked else '<span class="todo-chip">Not yet ranked</span>'
        st.markdown(
            f"<div class='nav-center'><strong>Paper {nav1 + 1} / {len(assigned_sets)}</strong><br>{badge_html}</div>",
            unsafe_allow_html=True,
        )
    with c_next:
        if st.button(
            "Next →",
            disabled=(nav1 == len(assigned_sets) - 1),
            key="sets_next",
            use_container_width=True,
        ):
            lbls = _task2_read_rank_labels_from_session(paper_id1, len(shuffled))
            if _task2_persist_ranking_if_complete(annotator, paper_id1, label_to_model, lbls):
                st.session_state.sets_nav = nav1 + 1
                st.rerun()

    if nav1 == len(assigned_sets) - 1:
        st.info(
            "You are on the **last** paper. **Next** does not go to the next task. "
            "Use **Save & Next** below to save your ranking and finish this step."
        )

    st.markdown("---")

    # ── Paper info ────────────────────────────────────────────────────────────
    if title1:
        st.markdown(
            f"<div class='paper-title-label'>Paper Title</div><div class='paper-title'>{html.escape(title1)}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div class="instructions-block">
    <span class="instructions-label">📌 Instructions:</span> Read all feedback sets below (Sets A, B, and C), then <strong>rank them according to four criteria</strong>: <strong>validity</strong> (is the feedback a valid 
issue/question/suggestion?), <strong>specificity</strong> (is it anchored to specific parts of the 
paper?), <strong>actionability</strong> (can the authors clearly act on it?), and <strong>helpfulness</strong> (how useful is it overall to the authors?). Each set letter must be used exactly once.
    <p class="instructions-irreversible">
    You can move between papers within Task 2 using the navigation controls. You cannot return to Task 1 or the counting step after you have advanced.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Feedback sets (side-by-side columns) ──────────────────────────────────
    st.markdown("<div class='sec-heading-center'>📄 Feedback Sets</div>", unsafe_allow_html=True)
    set_cols = st.columns(len(labels))
    for col, lbl in zip(set_cols, labels):
        model = label_to_model[lbl]
        col_key = f"feedback_set-{model}"
        text = str(srow.get(col_key, "") or "").strip()
        with col:
            st.markdown(
                f"<div class='fb-card-label'>Set {lbl}</div>",
                unsafe_allow_html=True,
            )
            if text:
                st.markdown(
                    f"<div class='fb-card'>{_format_feedback_text(text)}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("(no content)")

    st.markdown("---")

    # ── Ranking UI (rank slots 1…N left→right; pick set letter per slot) ─────
    st.markdown("<div class='sec-heading-center'>🏆 Your Ranking</div>", unsafe_allow_html=True)

    existing_ranking = st.session_state.rankings.get(paper_id1, [])

    label_options: list = [None] + list(labels)
    label_fmt = {None: "— select —", **{lb: f"Set {lb}" for lb in labels}}

    draft_rankpos_key = f"draft_rankpos_{paper_id1}"
    if draft_rankpos_key not in st.session_state:
        st.session_state[draft_rankpos_key] = {}
        for rank_num in range(1, len(shuffled) + 1):
            lbl0 = None
            if rank_num - 1 < len(existing_ranking):
                m0 = existing_ranking[rank_num - 1]
                lbl0 = model_to_label.get(m0)
            st.session_state[draft_rankpos_key][rank_num] = lbl0

    rank_cols = st.columns(len(shuffled))
    rank_to_label: dict[int, str | None] = {}
    for rank_idx, rank_num in enumerate(range(1, len(shuffled) + 1)):
        n = len(shuffled)
        if rank_num == 1:
            hint = "best"
        elif rank_num == n:
            hint = "worst"
        else:
            hint = ""
        with rank_cols[rank_idx]:
            hint_part = (
                f"<div class='rank-slot-hint'>{html.escape(hint)}</div>"
                if hint
                else "<div class='rank-slot-hint'>&nbsp;</div>"
            )
            with st.container(border=True):
                st.markdown(
                    f"<div style='text-align:center;margin-bottom:0.35rem'>"
                    f"<div class='rank-slot-num'>{rank_num}</div>{hint_part}</div>",
                    unsafe_allow_html=True,
                )
                current_val = st.session_state[draft_rankpos_key].get(rank_num)
                idx = label_options.index(current_val) if current_val in label_options else 0
                chosen = st.selectbox(
                    "",
                    options=label_options,
                    format_func=lambda x, rf=label_fmt: rf.get(x, "—"),
                    index=idx,
                    key=f"rankpos_{paper_id1}_{rank_num}",
                )
            st.session_state[draft_rankpos_key][rank_num] = chosen
            rank_to_label[rank_num] = chosen

    labels_in_order = [rank_to_label[r] for r in range(1, len(shuffled) + 1)]
    all_filled = all(x is not None for x in labels_in_order)
    all_unique = len(set(labels_in_order)) == len(labels_in_order) if all_filled else False

    if all_filled and not all_unique:
        st.warning("⚠️ Each feedback set (A, B, C, …) must be chosen exactly once. Adjust before submitting.")

    is_last_paper = (nav1 == len(assigned_sets) - 1)
    btn_label = "💾 Save & Next →"

    show_end_anno = (
        has_assigned_work
        and all_annotations_complete
        and not st.session_state.get("show_completion_page", False)
    )
    if show_end_anno and not is_last_paper:
        st.session_state.task2_final_save_done = False

    task2_end_only = show_end_anno and bool(st.session_state.get("task2_final_save_done"))

    if show_end_anno:
        _maybe_render_end_annotation_banner_only(end_only_ui=task2_end_only)

    if task2_end_only:
        _, end_only_col, _ = st.columns([1, 3, 1], gap="small")
        save_c = None
        end_anno_c = None
        status_c = None
        with end_only_col:
            if st.button(
                "End annotation",
                type="primary",
                use_container_width=True,
                key="btn_end_annotation",
            ):
                st.session_state.show_completion_page = True
                st.rerun()
    elif show_end_anno:
        _, pair_col, _ = st.columns([1, 3, 1], gap="small")
        with pair_col:
            save_c, end_anno_c = st.columns(2, gap="small")
    else:
        save_c, status_c = st.columns([2, 6], gap="small")
        end_anno_c = None

    if save_c is not None:
        with save_c:
            if st.button(
                btn_label,
                type="primary",
                disabled=(not all_filled or not all_unique),
                key="submit_ranking",
            ):
                sorted_models = [label_to_model[lb] for lb in labels_in_order]
                st.session_state.rankings[paper_id1] = sorted_models
                counting_payload, count_err = _ranking_counting_payload_for_save()
                if count_err:
                    st.session_state.last_save_toast = {"ok": False, "msg": count_err, "task": "ranking"}
                    st.rerun()

                ok, err = save_ranking(annotator, paper_id1, sorted_models, counting_payload=counting_payload)
                if ok:
                    st.session_state.last_save_toast = {"ok": True, "msg": "✅ Ranking saved!", "task": "ranking"}
                elif err:
                    st.session_state.last_save_toast = {"ok": False, "msg": f"❌ Save failed: {err}", "task": "ranking"}
                else:
                    st.session_state.last_save_toast = {"ok": None, "msg": "💾 Ranking saved locally (Google Sheets not configured).", "task": "ranking"}

                if ok or err is None:
                    if not is_last_paper:
                        st.session_state.sets_nav = nav1 + 1
                    elif show_end_anno:
                        st.session_state.task2_final_save_done = True
                st.rerun()

    if end_anno_c is not None:
        with end_anno_c:
            if st.button(
                "End annotation",
                type="primary",
                use_container_width=True,
                key="btn_end_annotation_task2_pair",
            ):
                st.session_state.show_completion_page = True
                st.rerun()

    # ── Persistent save status banner ─────────────────────────────────────────
    toast = st.session_state.get("last_save_toast")
    if toast and toast.get("task") == "ranking":
        if show_end_anno:
            _, toast_c, _ = st.columns([1, 4, 1], gap="small")
        else:
            toast_c = status_c
        with toast_c:
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

else:
    st.error(
        f"Invalid task flow state (phase={task_flow_phase}). Please refresh the page or use Change name to restart."
    )
