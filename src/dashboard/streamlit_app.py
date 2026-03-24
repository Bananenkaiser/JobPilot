"""Streamlit-Dashboard für den Bewerbungsoptimizer."""

import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import yaml
from bson import ObjectId
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent

load_dotenv(ROOT / ".env")

from src.analyzer.job_matcher import AnalysisResult, analyze_job
from src.storage.database import JOBS_COLLECTION, get_session, init_db
from src.storage.models import Job, JobStatus


@st.cache_resource
def _init() -> dict:
    """Einmalige Initialisierung: Config laden + DB verbinden."""
    with open(ROOT / "config/settings.yaml") as f:
        config = yaml.safe_load(f)
    db_cfg = config.get("database", {})
    init_db(
        uri=os.environ.get("MONGODB_URI") or db_cfg.get("uri"),
        db_name=os.environ.get("MONGODB_DB") or db_cfg.get("name"),
    )
    return config


def _load_jobs() -> list[dict]:
    with get_session() as db:
        cursor = db[JOBS_COLLECTION].find(
            {},
            {
                "_id": 1, "title": 1, "company": 1, "score": 1,
                "status": 1, "fetched_at": 1, "search_profile": 1,
                "full_analysis": 1, "description": 1, "url": 1,
                "applied_at": 1, "response_received": 1, "response_at": 1,
                "invited": 1, "rejected": 1, "rejection_text": 1,
            },
        ).sort("fetched_at", -1).limit(200)
        return list(cursor)


def _save_job(result: AnalysisResult, description: str) -> None:
    content_hash = hashlib.sha256(description.encode()).hexdigest()
    job = Job(
        guid=str(uuid.uuid4()),
        content_hash=content_hash,
        title=result.job_title or "Unbekannte Stelle",
        company=result.company or "Unbekanntes Unternehmen",
        url="",
        description=description,
        score=float(result.fit_score) if result.fit_score >= 0 else None,
        fetched_at=datetime.now(timezone.utc),
        status=JobStatus.new,
        search_profile="dashboard",
    )
    doc = job.to_document()
    doc["full_analysis"] = result.full_analysis
    doc["model_used"] = result.model_used
    doc["input_tokens"] = result.input_tokens
    doc["output_tokens"] = result.output_tokens

    with get_session() as db:
        db[JOBS_COLLECTION].insert_one(doc)


def _update_status(job_id_str: str, new_status: str) -> None:
    with get_session() as db:
        db[JOBS_COLLECTION].update_one(
            {"_id": ObjectId(job_id_str)},
            {"$set": {"status": new_status}},
        )


def _update_tracking(job_id_str: str, fields: dict) -> None:
    with get_session() as db:
        db[JOBS_COLLECTION].update_one(
            {"_id": ObjectId(job_id_str)},
            {"$set": fields},
        )


def _score_color(score: int) -> str:
    if score >= 70:
        return "green"
    if score >= 40:
        return "orange"
    return "red"


def _render_score(score: float | None) -> None:
    if score is None:
        st.caption("Kein Score vorhanden")
        return
    s = int(score)
    color = _score_color(s)
    st.metric("Fit-Score", f"{s}%")
    st.progress(s / 100)
    st.markdown(f"<span style='color:{color}'>{'●' * (s // 10)}{'○' * (10 - s // 10)}</span>", unsafe_allow_html=True)


def _render_analyse_tab(config: dict) -> None:
    col_input, col_info = st.columns([2, 1])

    with col_input:
        job_title = st.text_input("Jobtitel (optional)", key="input_job_title")
        company = st.text_input("Unternehmen (optional)", key="input_company")
        description = st.text_area(
            "Stellenausschreibung",
            height=400,
            placeholder="Stellenausschreibung hier einfügen...",
            key="input_description",
        )
        analyse_btn = st.button("Analysieren", type="primary", use_container_width=True)

    with col_info:
        cv_cfg = config.get("cv", {})
        cv_name = Path(cv_cfg.get("path", "")).name
        st.info(f"Lebenslauf: `{cv_name}`")
        backend = config.get("analyzer", {}).get("backend", "anthropic")
        st.info(f"KI-Backend: `{backend}`")

    if analyse_btn:
        if not description.strip():
            st.error("Bitte eine Stellenausschreibung eingeben.")
        else:
            cv_path = ROOT / config["cv"]["path"]
            me_str = config.get("cv", {}).get("me_path", "")
            me_path = ROOT / me_str if me_str else None

            with st.spinner("KI-Analyse läuft... (kann 30–120 Sekunden dauern)"):
                result = analyze_job(
                    job_description=description,
                    cv_path=cv_path,
                    me_path=me_path,
                    job_title=job_title,
                    company=company,
                    stream_output=False,
                    config=config,
                )
            st.session_state["analysis_result"] = result
            st.session_state["analysis_done"] = True
            st.session_state["save_success"] = False

    if st.session_state.get("analysis_done") and st.session_state.get("analysis_result"):
        result: AnalysisResult = st.session_state["analysis_result"]

        st.divider()
        st.subheader("Analyseergebnis")

        col_score, col_meta = st.columns([1, 2])
        with col_score:
            _render_score(result.fit_score if result.fit_score >= 0 else None)
        with col_meta:
            if result.model_used:
                st.caption(f"Modell: {result.model_used}")
            total = result.input_tokens + result.output_tokens
            if total > 0:
                st.caption(f"Tokens: {total:,}  (In: {result.input_tokens:,} | Out: {result.output_tokens:,})")

        st.markdown(result.full_analysis)

        st.divider()
        with st.expander("In Datenbank speichern", expanded=True):
            if st.session_state.get("save_success"):
                st.success("Job erfolgreich gespeichert!")
            else:
                if st.button("Job + Analyse in MongoDB speichern", key="save_btn"):
                    _save_job(result, st.session_state.get("input_description", ""))
                    st.session_state["save_success"] = True
                    st.rerun()


def _render_statistik_tab() -> None:
    jobs = _load_jobs()

    total = len(jobs)
    analysiert = sum(1 for j in jobs if j.get("score") is not None)
    beworben = sum(1 for j in jobs if j.get("applied_at") is not None)
    rueckmeldungen = sum(1 for j in jobs if j.get("response_received"))
    einladungen = sum(1 for j in jobs if j.get("invited"))
    ablehnungen = sum(1 for j in jobs if j.get("rejected"))
    scores = [j["score"] for j in jobs if j.get("score") is not None]
    avg_score = round(sum(scores) / len(scores)) if scores else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stellen gesamt", total)
    c2.metric("Analysiert", analysiert)
    c3.metric("Beworben", beworben)
    c4.metric("Ø Score", f"{avg_score}%" if avg_score is not None else "—")

    c5, c6, c7 = st.columns(3)
    c5.metric("Rückmeldungen", rueckmeldungen)
    c6.metric("Einladungen", einladungen)
    c7.metric("Ablehnungen", ablehnungen)

    st.divider()

    # Beste Stelle
    if scores:
        best = max(jobs, key=lambda j: j.get("score") or 0)
        st.markdown("**Beste Stelle**")
        bc1, bc2 = st.columns([3, 1])
        with bc1:
            st.markdown(f"**{best.get('title', '—')}** @ {best.get('company', '—')}")
            if best.get("url"):
                st.markdown(f"[Zur Stelle]({best['url']})")
        with bc2:
            _render_score(best.get("score"))

    st.divider()

    # Status-Verteilung
    if total > 0:
        import pandas as pd
        status_counts = {}
        for j in jobs:
            s = j.get("status", "new")
            status_counts[s] = status_counts.get(s, 0) + 1
        df = pd.DataFrame(
            {"Status": list(status_counts.keys()), "Anzahl": list(status_counts.values())}
        ).sort_values("Anzahl", ascending=False)
        st.markdown("**Status-Verteilung**")
        st.bar_chart(df.set_index("Status"))


def _add_manual_job(title: str, company: str, url: str, notes: str) -> None:
    job = Job(
        guid=str(uuid.uuid4()),
        content_hash=hashlib.sha256(f"{title}{company}".encode()).hexdigest(),
        title=title,
        company=company,
        url=url,
        description=notes,
        fetched_at=datetime.now(timezone.utc),
        status=JobStatus.new,
        search_profile="manuell",
    )
    doc = job.to_document()
    with get_session() as db:
        db[JOBS_COLLECTION].insert_one(doc)


def _render_stellen_tab() -> None:
    import pandas as pd

    col_header, col_reload = st.columns([4, 1])
    with col_header:
        st.subheader("Gespeicherte Stellen")
    with col_reload:
        if st.button("Aktualisieren"):
            st.rerun()

    with st.expander("➕ Stelle manuell hinzufügen"):
        c1, c2 = st.columns(2)
        with c1:
            new_title = st.text_input("Jobtitel *", key="new_title")
            new_company = st.text_input("Unternehmen *", key="new_company")
        with c2:
            new_url = st.text_input("URL (optional)", key="new_url")
            new_notes = st.text_input("Notiz (optional)", key="new_notes")
        if st.button("Hinzufügen", type="primary", key="add_job_btn"):
            if not new_title.strip() or not new_company.strip():
                st.error("Jobtitel und Unternehmen sind Pflichtfelder.")
            else:
                _add_manual_job(new_title.strip(), new_company.strip(), new_url.strip(), new_notes.strip())
                st.success(f"'{new_title}' @ {new_company} hinzugefügt.")
                st.rerun()

    jobs = _load_jobs()
    if not jobs:
        st.info("Noch keine Jobs in der Datenbank.")
        return

    rows = []
    for doc in jobs:
        score = doc.get("score")
        fetched = doc.get("fetched_at")
        applied_at = doc.get("applied_at")
        rows.append({
            "_id": str(doc["_id"]),
            "Titel": doc.get("title", ""),
            "Firma": doc.get("company", ""),
            "Score": f"{int(score)}%" if score is not None else "—",
            "Status": doc.get("status", "new"),
            "Beworben am": applied_at.strftime("%d.%m.%Y") if applied_at else "—",
            "Eingeladen": "✓" if doc.get("invited") else "",
            "Abgelehnt": "✓" if doc.get("rejected") else "",
            "Datum": fetched.strftime("%d.%m.%Y") if fetched else "—",
        })

    df = pd.DataFrame(rows)

    selection = st.dataframe(
        df.drop(columns=["_id"]),
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
    )

    selected_rows = selection.selection.rows
    if not selected_rows:
        return

    idx = selected_rows[0]
    doc = jobs[idx]
    row = rows[idx]

    st.divider()
    title = doc.get("title", "Stelle")
    company = doc.get("company", "")
    st.subheader(f"{title}" + (f" @ {company}" if company else ""))

    col_score, col_status, col_link = st.columns(3)

    with col_score:
        _render_score(doc.get("score"))

    with col_status:
        status_values = [s.value for s in JobStatus]
        current = doc.get("status", "new")
        new_status = st.selectbox(
            "Status",
            options=status_values,
            index=status_values.index(current) if current in status_values else 0,
            key=f"status_{row['_id']}",
        )
        if new_status != current:
            if st.button("Status speichern", key=f"save_status_{row['_id']}"):
                _update_status(row["_id"], new_status)
                st.success(f"Status aktualisiert: {new_status}")
                st.rerun()

    with col_link:
        if doc.get("url"):
            st.link_button("Zur Stellenausschreibung", doc["url"])

    # Bewerbungsverlauf
    st.markdown("### Bewerbungsverlauf")
    jid = row["_id"]

    col_a, col_b = st.columns(2)
    with col_a:
        applied = st.checkbox("Beworben", value=bool(doc.get("applied_at")), key=f"applied_{jid}")
        applied_date = None
        if applied:
            default_applied = doc["applied_at"].date() if doc.get("applied_at") else datetime.now().date()
            applied_date = st.date_input("Bewerbungsdatum", value=default_applied, key=f"applied_date_{jid}")

        response = st.checkbox("Rückmeldung erhalten", value=bool(doc.get("response_received")), key=f"response_{jid}")
        response_date = None
        if response:
            default_response = doc["response_at"].date() if doc.get("response_at") else datetime.now().date()
            response_date = st.date_input("Rückmeldung am", value=default_response, key=f"response_date_{jid}")

    with col_b:
        invited = st.checkbox("Eingeladen (Vorstellungsgespräch)", value=bool(doc.get("invited")), key=f"invited_{jid}")
        rejected = st.checkbox("Abgelehnt", value=bool(doc.get("rejected")), key=f"rejected_{jid}")
        rejection_text = ""
        if rejected:
            rejection_text = st.text_area(
                "Ablehnungstext",
                value=doc.get("rejection_text", ""),
                height=120,
                key=f"rejection_text_{jid}",
            )

    if st.button("Verlauf speichern", key=f"save_tracking_{jid}", type="primary"):
        fields: dict = {
            "response_received": response,
            "invited": invited,
            "rejected": rejected,
            "rejection_text": rejection_text if rejected else "",
            "applied_at": datetime.combine(applied_date, datetime.min.time()).replace(tzinfo=timezone.utc) if applied and applied_date else None,
            "response_at": datetime.combine(response_date, datetime.min.time()).replace(tzinfo=timezone.utc) if response and response_date else None,
        }
        # Status automatisch anpassen
        if rejected:
            fields["status"] = JobStatus.rejected.value
        elif invited:
            fields["status"] = JobStatus.interview.value
        elif applied:
            fields["status"] = JobStatus.applied.value
        _update_tracking(jid, fields)
        st.success("Bewerbungsverlauf gespeichert.")
        st.rerun()

    full_analysis = doc.get("full_analysis")
    if full_analysis:
        st.markdown("### Analyse")
        st.markdown(full_analysis)
    elif doc.get("description"):
        with st.expander("Stellenbeschreibung"):
            st.text(doc["description"][:3000])


def _render_profil_tab(config: dict) -> None:
    cv_cfg = config.get("cv", {})
    cv_path = ROOT / cv_cfg.get("path", "")
    me_path = ROOT / cv_cfg.get("me_path", "")

    st.subheader("Lebenslauf")
    if cv_path.exists():
        st.success(f"Aktuell: `{cv_path.name}` ({cv_path.stat().st_size // 1024} KB)")
    else:
        st.warning(f"Kein Lebenslauf gefunden unter: `{cv_path}`")

    uploaded = st.file_uploader("Neuen Lebenslauf hochladen (PDF)", type=["pdf"])
    if uploaded:
        cv_path.parent.mkdir(parents=True, exist_ok=True)
        cv_path.write_bytes(uploaded.read())
        st.success(f"Gespeichert: `{cv_path.name}`")

    st.divider()
    st.subheader("Persönliche Informationen (me.md)")
    st.caption("Diese Datei ergänzt deinen Lebenslauf mit zusätzlichem Kontext für die KI-Analyse.")

    current_text = me_path.read_text(encoding="utf-8") if me_path.exists() else ""
    new_text = st.text_area("Inhalt bearbeiten", value=current_text, height=400, key="me_md_editor")

    if st.button("Speichern", type="primary"):
        me_path.parent.mkdir(parents=True, exist_ok=True)
        me_path.write_text(new_text, encoding="utf-8")
        st.success("me.md gespeichert.")


def main() -> None:
    config = _init()

    st.set_page_config(
        page_title="Bewerbungsoptimizer",
        page_icon="📋",
        layout="wide",
    )
    st.title("Bewerbungsoptimizer")

    if "analysis_done" not in st.session_state:
        st.session_state["analysis_done"] = False
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "save_success" not in st.session_state:
        st.session_state["save_success"] = False

    tab_statistik, tab_analyse, tab_stellen, tab_profil = st.tabs(
        ["Übersicht", "Analyse", "Gespeicherte Stellen", "Mein Profil"]
    )

    with tab_statistik:
        _render_statistik_tab()

    with tab_analyse:
        _render_analyse_tab(config)

    with tab_stellen:
        _render_stellen_tab()

    with tab_profil:
        _render_profil_tab(config)
