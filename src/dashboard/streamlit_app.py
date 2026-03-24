"""Streamlit-Dashboard für JobPilot."""

import hashlib
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import yaml
from bson import ObjectId
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent.parent

load_dotenv(ROOT / ".env")

from src.analyzer.job_matcher import AnalysisResult, analyze_job, create_candidate_profile, extract_github_skills, suggest_cv_improvements, suggest_general_improvements
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
    doc["candidate_level"] = result.candidate_level
    doc["job_level"] = result.job_level

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
        job_title = st.text_input("Jobtitel (optional – wird von KI extrahiert)", key="input_job_title")
        company = st.text_input("Unternehmen (optional – wird von KI extrahiert)", key="input_company")
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
        profile_p = ROOT / cv_cfg.get("profile_path", "")
        if profile_p.exists():
            st.success("Kandidatenprofil aktiv – schnellere Analyse")
        else:
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
            profile_str = config.get("cv", {}).get("profile_path", "")
            profile_path = ROOT / profile_str if profile_str else None

            with st.spinner("KI-Analyse läuft... (kann 30–120 Sekunden dauern)"):
                result = analyze_job(
                    job_description=description,
                    cv_path=cv_path,
                    me_path=me_path,
                    job_title=job_title,
                    company=company,
                    stream_output=False,
                    config=config,
                    profile_path=profile_path,
                )
            _save_job(result, description)
            st.session_state["analysis_result"] = result
            st.session_state["analysis_done"] = True
            st.session_state["cv_improvements"] = None

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

        # --- CV-Optimierungen ---
        st.divider()
        st.subheader("Lebenslauf-Optimierungen für diese Stelle")
        profile_str = config.get("cv", {}).get("profile_path", "")
        profile_path_opt = ROOT / profile_str if profile_str else None

        if profile_path_opt and profile_path_opt.exists():
            if st.button("Optimierungsvorschläge generieren", key="cv_improve_btn", type="primary"):
                with st.spinner("Erstelle Optimierungsvorschläge... (~30 Sekunden)"):
                    improvements = suggest_cv_improvements(
                        st.session_state.get("input_description", ""),
                        profile_path_opt,
                        config,
                    )
                st.session_state["cv_improvements"] = improvements

            if st.session_state.get("cv_improvements"):
                st.markdown(st.session_state["cv_improvements"])
        else:
            st.info("Kandidatenprofil im Tab 'Mein Profil' erstellen um CV-Optimierungen zu erhalten.")


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

    # Neu analysieren
    description = doc.get("description", "")
    if description:
        st.divider()
        col_reanalyse, col_info = st.columns([2, 3])
        with col_reanalyse:
            if st.button("Mit aktuellem Profil neu analysieren", key=f"reanalyse_{jid}", type="primary"):
                config = _init()
                cv_cfg = config.get("cv", {})
                cv_path = ROOT / cv_cfg.get("path", "")
                me_path_str = cv_cfg.get("me_path", "")
                me_path = ROOT / me_path_str if me_path_str else None
                profile_str = cv_cfg.get("profile_path", "")
                profile_path = ROOT / profile_str if profile_str else None
                with st.spinner("KI-Analyse läuft... (kann 30–120 Sekunden dauern)"):
                    result = analyze_job(
                        job_description=description,
                        cv_path=cv_path,
                        me_path=me_path,
                        job_title=doc.get("title", ""),
                        company=doc.get("company", ""),
                        stream_output=False,
                        config=config,
                        profile_path=profile_path if (profile_path and profile_path.exists()) else None,
                    )
                # Ergebnis in DB überschreiben
                with get_session() as db:
                    db[JOBS_COLLECTION].update_one(
                        {"_id": doc["_id"]},
                        {"$set": {
                            "score": float(result.fit_score) if result.fit_score >= 0 else None,
                            "full_analysis": result.full_analysis,
                            "model_used": result.model_used,
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                            "candidate_level": result.candidate_level,
                            "job_level": result.job_level,
                        }},
                    )
                st.success(f"Neu analysiert – Score: {result.fit_score}%")
                st.rerun()
        with col_info:
            config = _init()
            profile_str = config.get("cv", {}).get("profile_path", "")
            profile_path = ROOT / profile_str if profile_str else None
            if profile_path and profile_path.exists():
                st.caption("Verwendet das aktuelle Kandidatenprofil.")
            else:
                st.caption("Kein Kandidatenprofil vorhanden – Analyse nutzt den rohen Lebenslauf.")

    full_analysis = doc.get("full_analysis")
    if full_analysis:
        st.markdown("### Analyse")
        st.markdown(full_analysis)
    elif description:
        with st.expander("Stellenbeschreibung"):
            st.text(description[:3000])


def _parse_profile(text: str) -> dict:
    """Extrahiert strukturierte Felder aus dem Kandidatenprofil-Markdown."""

    def _field(key: str) -> str:
        m = re.search(r"\*\*" + re.escape(key) + r":\*\*\s*(.+)", text)
        return m.group(1).strip() if m else ""

    def _bullets(key: str) -> list[str]:
        # \s* nach dem Doppelpunkt fängt optionale Leerzeichen + \r?\n ab
        m = re.search(r"\*\*" + re.escape(key) + r":\*\*\s*\r?\n((?:[ \t]*-[ \t]*.+\r?\n?)+)", text)
        if not m:
            return []
        return [re.sub(r"^[ \t]*-[ \t]*", "", line).strip()
                for line in m.group(1).splitlines() if line.strip()]

    # GitHub-Abschnitt separat parsen (andere Feldnamen aus GITHUB_PROMPT)
    gh_tools = _bullets("Tools & Technologien (aus Projekten)")
    gh_fachgebiete = _bullets("Erkennbare Fachgebiete")
    gh_projekttypen = _bullets("Projekttypen")
    gh_staerken = _bullets("Besondere Stärken (aus Projekten erkennbar)")

    # has_github: Überschrift ODER mindestens ein geparster GitHub-Wert vorhanden
    has_github = bool(gh_tools or gh_fachgebiete or gh_projekttypen or gh_staerken) or "GitHub-Profil" in text

    return {
        "level": _field("Erfahrungslevel"),
        "fachgebiet": _field("Fachgebiet"),
        "erfahrung": _field("Berufserfahrung"),
        "soft_skills": _field("Soft Skills & Besonderheiten"),
        "kompetenzen": _bullets("Kernkompetenzen"),
        # neue granulare Kategorien (neuer Prompt)
        "sprachen": _bullets("Programmiersprachen") or [s.strip() for s in _field("Sprachen").split(",") if s.strip()],
        "ml_ki": _bullets("ML & KI"),
        "daten": _bullets("Daten & Analyse"),
        "datenbanken": _bullets("Datenbanken"),
        "tools": _bullets("Weitere Tools & Technologien") or _bullets("Tools & Technologien"),
        "has_github": has_github,
        "gh_tools": gh_tools,
        "gh_fachgebiete": gh_fachgebiete,
        "gh_projekttypen": gh_projekttypen,
        "gh_staerken": gh_staerken,
    }


def _tags_html(items: list[str], color: str = "#1f77b4") -> str:
    """Rendert eine Liste als farbige Pill-Tags."""
    tags = "".join(
        f'<span style="display:inline-block;background:{color};color:white;'
        f'border-radius:12px;padding:2px 10px;margin:2px 3px 2px 0;font-size:0.85em">{item}</span>'
        for item in items
    )
    return f'<div style="line-height:2">{tags}</div>'


def _render_profil_overview(profil_text: str, profile_path: "Path") -> None:
    """Zeigt eine visuelle Übersicht des Kandidatenprofils."""
    p = _parse_profile(profil_text)

    # Level-Farbe
    level_raw = p["level"].lower()
    if "senior" in level_raw or "lead" in level_raw:
        level_color = "#2e7d32"
    elif "mid" in level_raw:
        level_color = "#e65100"
    else:
        level_color = "#1565c0"

    # Kopfzeile
    col_lvl, col_fach, col_exp = st.columns(3)
    with col_lvl:
        st.markdown(
            f'<div style="background:{level_color};color:white;border-radius:8px;'
            f'padding:10px 16px;text-align:center;font-weight:bold;font-size:1.1em">'
            f'{p["level"] or "Level unbekannt"}</div>',
            unsafe_allow_html=True,
        )
    with col_fach:
        st.markdown("**Fachgebiet**")
        st.markdown(p["fachgebiet"] or "—")
    with col_exp:
        st.markdown("**Berufserfahrung**")
        st.markdown(p["erfahrung"] or "—")

    st.markdown("")

    # Kernkompetenzen
    if p["kompetenzen"]:
        st.markdown("**Kernkompetenzen**")
        st.markdown(_tags_html(p["kompetenzen"], "#1565c0"), unsafe_allow_html=True)

    # Programmiersprachen
    if p["sprachen"]:
        st.markdown("**Programmiersprachen**")
        st.markdown(_tags_html(p["sprachen"], "#6a1b9a"), unsafe_allow_html=True)

    # ML & KI
    if p["ml_ki"]:
        st.markdown("**ML & KI**")
        st.markdown(_tags_html(p["ml_ki"], "#00695c"), unsafe_allow_html=True)

    # Daten & Analyse
    if p["daten"]:
        st.markdown("**Daten & Analyse**")
        st.markdown(_tags_html(p["daten"], "#37474f"), unsafe_allow_html=True)

    # Datenbanken
    if p["datenbanken"]:
        st.markdown("**Datenbanken**")
        st.markdown(_tags_html(p["datenbanken"], "#4a148c"), unsafe_allow_html=True)

    # Weitere Tools
    if p["tools"]:
        st.markdown("**Weitere Tools & Technologien**")
        st.markdown(_tags_html(p["tools"], "#bf360c"), unsafe_allow_html=True)

    # Soft Skills
    if p["soft_skills"]:
        st.markdown("**Soft Skills**")
        st.caption(p["soft_skills"])

    # GitHub-Abschnitt
    if p["has_github"]:
        st.divider()
        mtime = datetime.fromtimestamp(profile_path.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
        st.markdown(
            '<span style="background:#2da44e;color:white;border-radius:12px;'
            f'padding:3px 12px;font-size:0.85em">GitHub-Skills</span>'
            f' <span style="color:gray;font-size:0.8em">Zuletzt aktualisiert: {mtime}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        if p["gh_tools"]:
            st.markdown("**Tools & Technologien (aus Projekten)**")
            st.markdown(_tags_html(p["gh_tools"], "#1b6ca8"), unsafe_allow_html=True)
        gh_col1, gh_col2 = st.columns(2)
        with gh_col1:
            if p["gh_fachgebiete"]:
                st.markdown("**Fachgebiete (aus Projekten)**")
                for item in p["gh_fachgebiete"]:
                    st.markdown(f"- {item}")
            if p["gh_projekttypen"]:
                st.markdown("**Projekttypen**")
                for item in p["gh_projekttypen"]:
                    st.markdown(f"- {item}")
        with gh_col2:
            if p["gh_staerken"]:
                st.markdown("**Besondere Stärken**")
                for item in p["gh_staerken"]:
                    st.markdown(f"- {item}")
    else:
        st.markdown("")
        mtime = datetime.fromtimestamp(profile_path.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
        st.markdown(
            '<span style="background:#6e7781;color:white;border-radius:12px;padding:3px 12px;font-size:0.85em">'
            f'Kein GitHub-Profil</span>'
            f' <span style="color:gray;font-size:0.8em">Zuletzt aktualisiert: {mtime}</span>',
            unsafe_allow_html=True,
        )


def _render_profil_tab(config: dict) -> None:
    cv_cfg = config.get("cv", {})
    cv_path = ROOT / cv_cfg.get("path", "")
    me_path = ROOT / cv_cfg.get("me_path", "")
    profile_str = cv_cfg.get("profile_path", "")
    profile_path = ROOT / profile_str if profile_str else None

    profil_text = profile_path.read_text(encoding="utf-8") if (profile_path and profile_path.exists()) else None
    has_github = profil_text is not None and "## GitHub-Profil" in profil_text
    me_filled = me_path.exists() and me_path.read_text(encoding="utf-8").strip()

    # ── Panel-Toggle via Session State ───────────────────────────────────────
    if "profil_panel" not in st.session_state:
        st.session_state["profil_panel"] = None

    def _toggle(panel: str) -> None:
        st.session_state["profil_panel"] = None if st.session_state["profil_panel"] == panel else panel

    # ── Action-Buttons ───────────────────────────────────────────────────────
    active = st.session_state["profil_panel"]
    b1, b2, b3, b4 = st.columns(4)

    cv_label = ("✓ Lebenslauf" if cv_path.exists() else "✗ Lebenslauf")
    me_label = ("✓ Persönliche Infos" if me_filled else "◎ Persönliche Infos")
    gh_label = ("✓ GitHub-Profil" if has_github else "◎ GitHub-Profil")
    profil_label = ("Profil neu erstellen" if profil_text else "Profil erstellen")

    with b1:
        if st.button(cv_label, use_container_width=True,
                     type="primary" if active == "cv" else "secondary"):
            _toggle("cv")
            st.rerun()
    with b2:
        if st.button(me_label, use_container_width=True,
                     type="primary" if active == "me" else "secondary"):
            _toggle("me")
            st.rerun()
    with b3:
        if st.button(gh_label, use_container_width=True,
                     type="primary" if active == "github" else "secondary"):
            _toggle("github")
            st.rerun()
    with b4:
        if st.button(profil_label, use_container_width=True,
                     type="primary" if active == "profil" else "secondary"):
            _toggle("profil")
            st.rerun()

    # ── Aktives Panel ────────────────────────────────────────────────────────
    if active == "cv":
        with st.container(border=True):
            if cv_path.exists():
                st.caption(f"Aktuell: **{cv_path.name}** ({cv_path.stat().st_size // 1024} KB)")
            else:
                st.warning(f"Noch kein Lebenslauf unter `{cv_path}`")
            uploaded = st.file_uploader("PDF hochladen", type=["pdf"], key="cv_uploader")
            if uploaded:
                cv_path.parent.mkdir(parents=True, exist_ok=True)
                # Alten Lebenslauf löschen (auch wenn Dateiname sich ändert)
                for old_pdf in cv_path.parent.glob("*.pdf"):
                    old_pdf.unlink()
                new_cv_path = cv_path.parent / uploaded.name
                new_cv_path.write_bytes(uploaded.read())
                # settings.yaml aktualisieren falls Dateiname geändert
                if new_cv_path != cv_path:
                    settings_path = ROOT / "config/settings.yaml"
                    settings_text = settings_path.read_text(encoding="utf-8")
                    new_rel = str(new_cv_path.relative_to(ROOT))
                    old_rel = str(cv_path.relative_to(ROOT))
                    settings_path.write_text(
                        settings_text.replace(old_rel, new_rel), encoding="utf-8"
                    )
                    _init.clear()  # Cache leeren damit neue Config geladen wird
                st.success(f"Gespeichert: {uploaded.name} – alter Lebenslauf gelöscht.")
                st.session_state["profil_panel"] = None
                st.rerun()

    elif active == "me":
        with st.container(border=True):
            st.caption("Ergänzt deinen Lebenslauf mit persönlichem Kontext für die KI-Analyse.")

            uploaded_me = st.file_uploader(
                "Datei hochladen (.md oder .txt) – überschreibt den aktuellen Inhalt",
                type=["md", "txt"],
                key="me_uploader",
            )
            if uploaded_me:
                file_text = uploaded_me.read().decode("utf-8", errors="replace")
                st.session_state["me_md_uploaded"] = file_text
                st.rerun()

            initial_text = st.session_state.pop("me_md_uploaded", None)
            if initial_text is None:
                initial_text = me_path.read_text(encoding="utf-8") if me_path.exists() else ""

            new_text = st.text_area("Inhalt", value=initial_text, height=300, key="me_md_editor")
            if st.button("Speichern", key="save_me_md", type="primary"):
                me_path.parent.mkdir(parents=True, exist_ok=True)
                me_path.write_text(new_text, encoding="utf-8")
                st.success("Gespeichert.")
                st.session_state["profil_panel"] = None
                st.rerun()

    elif active == "github":
        with st.container(border=True):
            if not profil_text:
                st.warning("Bitte zuerst ein Kandidatenprofil erstellen.")
            else:
                st.caption("Alle öffentlichen Repositories durchsuchen und Skills ins Profil einbinden.")
                github_input = st.text_input(
                    "GitHub-Benutzername oder URL",
                    placeholder="Bananenkaiser  oder  https://github.com/Bananenkaiser",
                    key="github_username",
                )
                if st.button("Skills aus allen Repos extrahieren", key="github_btn", type="primary"):
                    if not github_input.strip():
                        st.error("Bitte einen Benutzernamen eingeben.")
                    else:
                        import httpx
                        raw = github_input.strip().rstrip("/")
                        username = raw.split("github.com/")[-1].split("/")[0] if "github.com/" in raw else raw
                        try:
                            api_url = f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated"
                            with st.spinner(f"Lade Repositories von @{username}..."):
                                resp = httpx.get(api_url, timeout=15, follow_redirects=True,
                                                 headers={"Accept": "application/vnd.github+json"})
                            if resp.status_code == 403:
                                st.error("GitHub API Rate-Limit erreicht. Bitte kurz warten und erneut versuchen.")
                            elif resp.status_code == 404:
                                st.error(f"Benutzer '{username}' nicht gefunden.")
                            elif resp.status_code != 200:
                                st.error(f"GitHub API Fehler (HTTP {resp.status_code}): {resp.text[:200]}")
                            else:
                                repos = resp.json()
                                if not repos:
                                    st.warning(f"@{username} hat keine öffentlichen Repositories.")
                                else:
                                    st.caption(f"{len(repos)} Repositories gefunden – lade READMEs...")
                                    combined_parts = []
                                    progress = st.progress(0.0, text="Lade READMEs...")
                                    fetched = 0
                                    for i, repo in enumerate(repos):
                                        repo_name = repo.get("name", "")
                                        branch = repo.get("default_branch", "main")
                                        url = f"https://raw.githubusercontent.com/{username}/{repo_name}/{branch}/README.md"
                                        try:
                                            r = httpx.get(url, timeout=8, follow_redirects=True)
                                            if r.status_code == 200 and r.text.strip():
                                                combined_parts.append(f"### {repo_name}\n\n{r.text[:2000]}")
                                                fetched += 1
                                        except Exception:
                                            pass
                                        progress.progress((i + 1) / len(repos),
                                                          text=f"Lade READMEs... ({i+1}/{len(repos)})")
                                    progress.empty()
                                    if not combined_parts:
                                        st.warning(f"Keine READMEs in {len(repos)} Repositories gefunden.")
                                    else:
                                        combined_text = "\n\n---\n\n".join(combined_parts)
                                        with st.spinner(f"Extrahiere Skills aus {fetched} READMEs..."):
                                            skills_text = extract_github_skills(combined_text, config)
                                        if not skills_text.strip():
                                            st.warning("KI hat keine Skills extrahiert (leere Antwort).")
                                        else:
                                            profile_path.write_text(profil_text + "\n\n" + skills_text, encoding="utf-8")
                                            # Erfolgsmeldung in session_state speichern – überlebt st.rerun()
                                            st.session_state["github_success"] = f"GitHub-Skills aus {fetched} Repositories eingebunden."
                                            st.session_state["profil_panel"] = None
                                            st.rerun()
                        except Exception as e:
                            st.error(f"Fehler: {e}")

    elif active == "profil":
        with st.container(border=True):
            if not cv_path.exists():
                st.warning("Bitte zuerst einen Lebenslauf hochladen.")
            else:
                st.caption("Analysiert Lebenslauf + persönliche Infos einmalig und erzeugt ein kompaktes Profil (~75% weniger Tokens bei Analysen).")
                col_a, col_b, col_c = st.columns([2, 1, 1])
                with col_a:
                    if st.button("Profil jetzt erstellen / neu erstellen", type="primary", key="create_profil"):
                        with st.spinner("Analysiere Lebenslauf... (~30 Sekunden)"):
                            new_profil = create_candidate_profile(cv_path, me_path, config)
                        profile_path.parent.mkdir(parents=True, exist_ok=True)
                        profile_path.write_text(new_profil, encoding="utf-8")
                        st.success("Profil erstellt.")
                        st.session_state["profil_panel"] = None
                        st.rerun()
                if profil_text:
                    with col_b:
                        st.write("")
                    with col_c:
                        if st.button("Profil löschen", key="delete_profil", type="secondary"):
                            st.session_state["confirm_delete_profil"] = True
                    if st.session_state.get("confirm_delete_profil"):
                        st.warning("Profil wirklich löschen? Diese Aktion kann nicht rückgängig gemacht werden.")
                        cd1, cd2 = st.columns(2)
                        with cd1:
                            if st.button("Ja, löschen", key="confirm_delete_yes", type="primary"):
                                profile_path.unlink()
                                st.session_state.pop("confirm_delete_profil", None)
                                st.session_state["profil_panel"] = None
                                st.rerun()
                        with cd2:
                            if st.button("Abbrechen", key="confirm_delete_no"):
                                st.session_state.pop("confirm_delete_profil", None)
                                st.rerun()
                    st.divider()
                    st.caption("Manuell bearbeiten:")
                    edited = st.text_area("Rohtext", value=profil_text, height=350, key="profil_editor")
                    if st.button("Speichern", key="save_profil"):
                        profile_path.write_text(edited, encoding="utf-8")
                        st.success("Gespeichert.")
                        st.session_state["profil_panel"] = None
                        st.rerun()

    # Persistente Statusmeldungen (überleben st.rerun)
    if "github_success" in st.session_state:
        st.success(st.session_state.pop("github_success"))

    # ── Extrahierte Informationen ────────────────────────────────────────────
    if profil_text:
        st.divider()
        st.subheader("Extrahierte Informationen")
        _render_profil_overview(profil_text, profile_path)

        # ── Allgemeine Verbesserungsvorschläge ───────────────────────────────
        st.divider()
        st.subheader("Allgemeine Verbesserungsvorschläge")
        if st.button("Verbesserungsvorschläge generieren", key="btn_general_improvements"):
            with st.spinner("Analysiere Profil …"):
                result = suggest_general_improvements(profile_path, config)
                st.session_state["general_improvements"] = result
        if st.session_state.get("general_improvements"):
            st.markdown(st.session_state["general_improvements"])
    else:
        st.divider()
        st.info("Noch keine Informationen extrahiert. Lebenslauf hochladen und Profil erstellen.")


def main() -> None:
    config = _init()

    st.set_page_config(
        page_title="JobPilot",
        page_icon="📋",
        layout="wide",
    )
    st.title("JobPilot")

    if "analysis_done" not in st.session_state:
        st.session_state["analysis_done"] = False
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "save_success" not in st.session_state:
        st.session_state["save_success"] = False
    if "cv_improvements" not in st.session_state:
        st.session_state["cv_improvements"] = None

    tab_statistik, tab_stellen, tab_analyse, tab_profil = st.tabs(
        ["Übersicht", "Gespeicherte Stellen", "Analyse", "Mein Profil"]
    )

    with tab_statistik:
        _render_statistik_tab()

    with tab_stellen:
        _render_stellen_tab()

    with tab_analyse:
        _render_analyse_tab(config)

    with tab_profil:
        _render_profil_tab(config)
