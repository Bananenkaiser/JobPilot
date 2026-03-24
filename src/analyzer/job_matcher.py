"""Job-CV Matching Analyse — unterstützt Anthropic Claude und LM Studio."""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Standardmodelle
ANTHROPIC_MODEL = "claude-opus-4-6"

SYSTEM_PROMPT = """Du bist ein erfahrener Karriereberater und Personalvermittler.
Du analysierst, wie gut eine Stellenausschreibung zu einem Lebenslauf passt.
Antworte immer auf Deutsch, strukturiert und konkret.
Sei ehrlich – auch wenn die Passung gering ist."""

ANALYSIS_PROMPT = """Analysiere die Passung zwischen dem folgenden Lebenslauf und der Stellenausschreibung.

## Lebenslauf
{cv_content}
{me_section}
## Stellenausschreibung
{job_content}

Beginne deine Antwort immer mit diesen Zeilen (extrahiert bzw. eingeschätzt):
**Stelle:** [Jobtitel aus der Ausschreibung]
**Unternehmen:** [Unternehmensname aus der Ausschreibung]
**Kandidaten-Level:** [Einschätzung des Levels aus dem Lebenslauf, z.B. Junior, Mid-Level, Senior, Lead]
**Ausschreibungs-Level:** [Gesuchtes Level laut Stellenausschreibung, z.B. Junior, Mid-Level, Senior, Lead]

Erstelle danach eine strukturierte Analyse mit folgenden Abschnitten:

### 1. Gesamtbewertung
- Passungsgrad in Prozent (0–100%)
- Berücksichtige dabei explizit, ob das Level des Kandidaten zum gesuchten Level der Stelle passt. Eine starke Level-Abweichung (z.B. Senior-Kandidat auf Junior-Stelle oder umgekehrt) soll den Score deutlich beeinflussen.
- Kurze Begründung (2–3 Sätze)

### 2. Level-Einschätzung
- Begründe, warum du den Kandidaten als [Level] einschätzt.
- WICHTIG: Zähle als "Berufserfahrung im Fachgebiet" ausschließlich Stellen im IT/Data-Science-Bereich nach dem Studienabschluss. Bundeswehr/Militärdienst zählt NICHT als fachliche Berufserfahrung, auch wenn er im Lebenslauf unter "Berufserfahrung" steht – er ist Wehrdienst, keine Fachkarriere. Studiumsprojekte und Abschlussarbeiten zählen ebenfalls nicht als Berufserfahrung.
- Passt das Level zur Stelle? Wenn nicht: wie groß ist die Abweichung und wie wirkt sie sich aus?

### 3. Stärken (was passt gut)
Liste die konkreten Übereinstimmungen zwischen Lebenslauf und Stelle auf.

### 4. Lücken (was fehlt oder ist schwach)
Liste die Anforderungen der Stelle, die im Lebenslauf fehlen oder unzureichend dargestellt sind.

#### Fehlende Tools & Technologien
Welche konkreten Tools, Frameworks oder Technologien werden in der Ausschreibung gefordert oder erwartet, die im Profil des Kandidaten **nicht** vorkommen?
- [Tool/Technologie – z.B. Kubernetes, dbt, Terraform]

#### Fehlende Erfahrungen & Kenntnisse
Welche fachlichen Erfahrungen, Methoden oder Kenntnisgebiete werden erwartet, die beim Kandidaten **fehlen oder unzureichend** belegt sind?
- [Erfahrung/Kenntnis – z.B. Cloud-Infrastruktur, MLOps, Produktionsdeployments]

#### Sonstige Lücken
Weitere Anforderungen (Soft Skills, Zertifikate, Sprachkenntnisse, Branchenerfahrung etc.), die nicht erfüllt sind:
- [Sonstige Lücke]

### 5. Konkrete Lebenslauf-Optimierungen für diese Stelle
Zeige für jede Lücke/Schwäche **konkret**, was ich im Lebenslauf ändern oder ergänzen sollte.
Format: „Abschnitt X: [aktuelle Formulierung] → [optimierte Formulierung]" oder „Fehlend: [was ergänzen und wo]".

### 6. Empfehlung
Soll ich mich bewerben? Mit welcher Strategie?

#### Wichtige ATS-Keywords für diese Stelle
Keywords aus der Ausschreibung, die im Lebenslauf vorhanden sein sollten, damit ATS-Systeme eine höhere Trefferquote erzielen:
- **[Keyword]** — im Lebenslauf vorhanden: Ja / Nein"""


PROFILE_PROMPT = """Analysiere den folgenden Lebenslauf und extrahiere daraus ein vollständiges Kandidatenprofil.
WICHTIG: Kürze nichts ab. Liste ALLE genannten Tools, Technologien und Fähigkeiten einzeln auf – jedes Element in einer eigenen Zeile mit Bindestrich.

## Lebenslauf
{cv_content}
{me_section}
Erstelle das Profil exakt in diesem Format (Markdown). Jeder Listenpunkt steht auf einer eigenen Zeile:

## Kandidatenprofil

**Erfahrungslevel:** [Junior / Mid-Level / Senior / Lead – mit kurzer Begründung]

**Fachgebiet:** [z.B. Data Science, Backend-Entwicklung, ML Engineering]

**Berufserfahrung:** [NUR Stellen im IT/Data-Science-Fachbereich nach dem Studienabschluss in Monaten/Jahren. Bundeswehr/Militärdienst zählt hier NICHT, auch wenn er im Lebenslauf unter "Berufserfahrung" steht – separate Erwähnung unter Besonderheiten.]

**Kernkompetenzen:**
- [eine Kompetenz pro Zeile]

**Programmiersprachen:**
- [eine Sprache pro Zeile – alle aus dem Lebenslauf]

**ML & KI:**
- [ein Framework/Tool pro Zeile – alle aus dem Lebenslauf]

**Daten & Analyse:**
- [ein Tool pro Zeile]

**Datenbanken:**
- [eine Datenbank pro Zeile]

**Weitere Tools & Technologien:**
- [ein Tool pro Zeile – alles was nicht in obige Kategorien passt]

**Soft Skills & Besonderheiten:** [nur wenn klar erkennbar, sonst weglassen]"""

CV_IMPROVEMENT_PROMPT = """Du bist ein erfahrener Karriereberater. Erstelle konkrete, umsetzbare Verbesserungsvorschläge für einen Lebenslauf, um ihn optimal auf die folgende Stelle auszurichten.

## Kandidatenprofil
{profile_content}

## Stellenausschreibung
{job_content}

Erstelle eine detaillierte Optimierungsanleitung mit folgender Struktur:

### Lebenslauf-Optimierungen für diese Stelle

#### Priorität 1: Kritische Änderungen (unbedingt anpassen)
Für jede kritische Änderung:
- **Bereich:** [z.B. Berufserfahrung / Skills / Zusammenfassung]
- **Empfehlung:** [konkret was zu ändern oder ergänzen ist]
- **Warum:** [warum das für diese Stelle entscheidend ist]

#### Priorität 2: Empfohlene Ergänzungen
- **Bereich:** ...
- **Empfehlung:** ...

#### Priorität 3: Nice-to-have
- ...

#### Keywords für ATS-Systeme
Relevante Keywords aus der Ausschreibung, die im Lebenslauf vorkommen sollten, damit ATS-Systeme eine höhere Trefferquote erzielen.
Für jedes Keyword: wo genau im Lebenslauf einbauen und wie es sich natürlich in den Text integrieren lässt.
- **[Keyword]** → Einbauen in: [z.B. Skills-Abschnitt / Berufserfahrung bei Stelle X / Zusammenfassung] — Beispiel: „[kurzer Beispielsatz oder Formulierungsvorschlag]"

#### Hinweise fürs Anschreiben
Worauf sollte das Anschreiben besonders eingehen?
- ..."""

GENERAL_IMPROVEMENT_PROMPT = """Du bist ein erfahrener Karriereberater. Analysiere das folgende Kandidatenprofil und erstelle allgemeine, stellenunabhängige Verbesserungsvorschläge für Lebenslauf und Anschreiben.

## Kandidatenprofil
{profile_content}

### Allgemeine Lebenslauf-Optimierungen

#### Stärken (was bereits gut ist)
- [was im Profil positiv auffällt]

#### Verbesserungspotenzial
Für jede Schwäche: konkrete Stelle im Lebenslauf benennen und eine verbesserte Version zeigen.
- **[Schwäche, z.B. unklare Formulierung / fehlende Kennzahl]:** „[exakte Originalformulierung aus dem Profil]" → „[verbesserte Version]"

#### Fehlende Elemente für typische {field}-Stellen
- [was in diesem Bereich üblicherweise erwartet wird, aber fehlt]

#### Formatierungs- & Struktur-Tipps
- [Hinweise zu Aufbau, Länge, Lesbarkeit]

---

### Allgemeine Anschreiben-Tipps für dein Profil

#### Einstieg & Aufhänger
[Wie sollte ein überzeugender Einstieg für jemanden mit diesem Hintergrund aussehen? Konkretes Beispiel.]

#### Kernbotschaft
[Was ist dein stärkstes Argument für Arbeitgeber — was sollte immer im Anschreiben stehen?]

#### Häufige Fehler vermeiden
- [typische Schwachstellen für dieses Profil im Anschreiben]

#### Muster-Eröffnungssatz
[Ein konkreter Beispielsatz als Vorlage]"""

GITHUB_PROMPT = """Analysiere die folgenden GitHub-Projekte eines Entwicklers und extrahiere daraus ein vollständiges Bild seiner technischen Fähigkeiten.

## GitHub-Projekte (READMEs)
{readme_content}

Antworte nur mit diesem Markdown-Abschnitt (keine Einleitung, kein Kommentar):

## GitHub-Profil & Projektskills

**Tools & Technologien (aus Projekten):**
- [Tool/Framework/Bibliothek]

**Erkennbare Fachgebiete:**
- [Fachgebiet, z.B. Machine Learning, Webentwicklung]

**Projekttypen:**
- [z.B. Datenanalyse, APIs, Automatisierung]

**Besondere Stärken (aus Projekten erkennbar):**
- [Stärke]"""


@dataclass
class AnalysisResult:
    fit_score: int  # 0–100
    full_analysis: str
    job_title: str = ""
    company: str = ""
    candidate_level: str = ""   # z.B. "Senior", "Mid-Level"
    job_level: str = ""         # gesuchtes Level laut Ausschreibung
    input_tokens: int = 0
    output_tokens: int = 0

    model_used: str = ""
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    cv_improvements: list[str] = field(default_factory=list)
    recommendation: str = ""


def _read_cv(cv_path: Path) -> tuple[str, str]:
    """Liest den Lebenslauf. Gibt (content_type, content) zurück.
    content_type: 'text' oder 'pdf_base64'
    """
    suffix = cv_path.suffix.lower()
    if suffix == ".pdf":
        data = cv_path.read_bytes()
        return "pdf_base64", base64.standard_b64encode(data).decode()
    return "text", cv_path.read_text(encoding="utf-8", errors="replace")


def _pdf_to_text(cv_path: Path) -> str:
    """Extrahiert Text aus einer PDF-Datei (für LM Studio, das kein PDF-Upload unterstützt)."""
    from pypdf import PdfReader
    reader = PdfReader(cv_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_metadata(text: str) -> tuple[str, str, str, str]:
    """Extrahiert Jobtitel, Unternehmen und Level-Einschätzungen aus dem KI-Antwort-Header."""
    def _get(pattern: str) -> str:
        m = re.search(pattern, text)
        return m.group(1).strip() if m else ""

    return (
        _get(r"\*\*Stelle:\*\*\s*(.+)"),
        _get(r"\*\*Unternehmen:\*\*\s*(.+)"),
        _get(r"\*\*Kandidaten-Level:\*\*\s*(.+)"),
        _get(r"\*\*Ausschreibungs-Level:\*\*\s*(.+)"),
    )


def _extract_fit_score(text: str) -> int:
    """Extrahiert den Passungsgrad aus dem Analysetext."""
    patterns = [
        r"Passungsgrad[^\d]*(\d{1,3})\s*%",
        r"(\d{1,3})\s*%\s*(?:Passung|Übereinstimmung)",
        r"Gesamtbewertung[^%]*?(\d{1,3})\s*%",
        r"(\d{1,3})\s*(?:von\s*100|%)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return val
    return -1


def _analyze_with_anthropic(
    cv_path: Path,
    me_section: str,
    job_description: str,
    stream_output: bool,
) -> tuple[str, int, int]:
    """Analyse via Anthropic Claude API. Gibt (full_text, input_tokens, output_tokens) zurück."""
    import anthropic

    client = anthropic.Anthropic()
    cv_type, cv_content = _read_cv(cv_path)

    if cv_type == "pdf_base64":
        user_content: list = [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": cv_content,
                },
                "title": "Lebenslauf",
            },
            {
                "type": "text",
                "text": ANALYSIS_PROMPT.format(
                    cv_content="[Lebenslauf als PDF oben beigefügt]",
                    me_section=me_section,
                    job_content=job_description,
                ),
            },
        ]
    else:
        user_content = [
            {
                "type": "text",
                "text": ANALYSIS_PROMPT.format(
                    cv_content=cv_content,
                    me_section=me_section,
                    job_content=job_description,
                ),
            }
        ]

    full_text = ""
    input_tokens = 0
    output_tokens = 0

    if stream_output:
        print()
        with client.messages.stream(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        print(event.delta.text, end="", flush=True)
                        full_text += event.delta.text
            final = stream.get_final_message()
            input_tokens = final.usage.input_tokens
            output_tokens = final.usage.output_tokens
        print()
    else:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        for block in response.content:
            if block.type == "text":
                full_text = block.text
                break
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

    return full_text, input_tokens, output_tokens


def _analyze_with_lmstudio(
    cv_path: Path,
    me_section: str,
    job_description: str,
    stream_output: bool,
    base_url: str,
    model: str,
) -> tuple[str, int, int]:
    """Analyse via LM Studio (OpenAI-kompatibler Endpunkt). Gibt (full_text, input_tokens, output_tokens) zurück."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="lm-studio")

    # LM Studio unterstützt kein PDF-Upload → Text extrahieren
    if cv_path.suffix.lower() == ".pdf":
        cv_content = _pdf_to_text(cv_path)
        logger.info("PDF zu Text konvertiert für LM Studio (%d Zeichen)", len(cv_content))
    else:
        cv_content = cv_path.read_text(encoding="utf-8", errors="replace")

    prompt = ANALYSIS_PROMPT.format(
        cv_content=cv_content,
        me_section=me_section,
        job_content=job_description,
    )

    full_text = ""
    input_tokens = 0
    output_tokens = 0

    if stream_output:
        print()
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                print(delta, end="", flush=True)
                full_text += delta
        print()
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
        )
        full_text = response.choices[0].message.content or ""
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

    return full_text, input_tokens, output_tokens


def _call_llm(prompt: str, config: dict | None) -> str:
    """Einfacher LLM-Aufruf ohne Streaming, gibt den Text zurück."""
    analyzer_cfg = (config or {}).get("analyzer", {})
    backend = analyzer_cfg.get("backend", "anthropic")

    if backend == "lmstudio":
        from openai import OpenAI
        base_url = analyzer_cfg.get("lmstudio_url", "http://localhost:1234/v1")
        model = analyzer_cfg.get("lmstudio_model", "local-model")
        client = OpenAI(base_url=base_url, api_key="lm-studio")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("LM Studio fehlgeschlagen (%s) – Fallback auf Anthropic.", exc)

    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def create_candidate_profile(cv_path: Path, me_path: Path | None, config: dict | None) -> str:
    """Analysiert den Lebenslauf einmalig und erstellt ein kompaktes Kandidatenprofil."""
    cv_type, cv_content = _read_cv(cv_path)
    if cv_type == "pdf_base64":
        cv_content = _pdf_to_text(cv_path)

    me_section = ""
    if me_path and me_path.exists():
        me_text = me_path.read_text(encoding="utf-8", errors="replace").strip()
        if me_text:
            me_section = f"\n## Weitere persönliche Informationen\n{me_text}\n"

    prompt = PROFILE_PROMPT.format(cv_content=cv_content, me_section=me_section)
    return _call_llm(prompt, config)


def suggest_cv_improvements(job_description: str, profile_path: Path, config: dict | None) -> str:
    """Erstellt stellenspezifische Verbesserungsvorschläge für den Lebenslauf."""
    profile_text = profile_path.read_text(encoding="utf-8")
    prompt = CV_IMPROVEMENT_PROMPT.format(
        profile_content=profile_text,
        job_content=job_description,
    )
    return _call_llm(prompt, config)


def suggest_general_improvements(profile_path: Path, config: dict | None) -> str:
    """Allgemeine, stellenunabhängige Verbesserungsvorschläge für CV und Anschreiben."""
    profile_text = profile_path.read_text(encoding="utf-8")
    field_match = re.search(r"\*\*Fachgebiet:\*\*\s*(.+)", profile_text)
    field = field_match.group(1).strip() if field_match else "IT/Data-Science"
    prompt = GENERAL_IMPROVEMENT_PROMPT.format(profile_content=profile_text, field=field)
    return _call_llm(prompt, config)


def extract_github_skills(readme_text: str, config: dict | None) -> str:
    """Extrahiert Tools und Skills aus einem GitHub README."""
    prompt = GITHUB_PROMPT.format(readme_content=readme_text[:8000])
    return _call_llm(prompt, config)


def analyze_job(
    job_description: str,
    cv_path: Path,
    me_path: Path | None = None,
    job_title: str = "",
    company: str = "",
    stream_output: bool = True,
    config: dict | None = None,
    profile_path: Path | None = None,
) -> AnalysisResult:
    """Analysiert die Passung zwischen Stellenausschreibung und Lebenslauf.

    Wählt automatisch das Backend aus config['analyzer']['backend']:
    - 'anthropic' (Standard): Claude API
    - 'lmstudio': lokales LM Studio über OpenAI-kompatiblen Endpunkt
    """
    me_section = ""
    if me_path and me_path.exists():
        me_text = me_path.read_text(encoding="utf-8", errors="replace").strip()
        if me_text:
            me_section = f"\n## Weitere persönliche Informationen\n{me_text}\n"

    # Profil als kompakte CV-Zusammenfassung nutzen (spart ~75% Input-Tokens)
    if profile_path and profile_path.exists():
        logger.info("Kandidatenprofil gefunden – verwende kompaktes Profil statt rohem Lebenslauf.")
        profile_text = profile_path.read_text(encoding="utf-8")
        _use_profile = True
    else:
        logger.info("Lebenslauf: %s", cv_path.name)
        _use_profile = False

    analyzer_cfg = (config or {}).get("analyzer", {})
    backend = analyzer_cfg.get("backend", "anthropic")

    model_used = ""
    if _use_profile:
        # Profil-Modus: kompakten Text direkt als Prompt übergeben (kein PDF nötig)
        # Wir leiten an _analyze_with_lmstudio/_anthropic weiter, aber mit Dummy-cv_path
        # und überschreiben cv_content intern via me_section
        profile_me = f"{profile_text}\n{me_section}"
        if backend == "lmstudio":
            base_url = analyzer_cfg.get("lmstudio_url", "http://localhost:1234/v1")
            model = analyzer_cfg.get("lmstudio_model", "local-model")
            from openai import OpenAI
            client_lm = OpenAI(base_url=base_url, api_key="lm-studio")
            prompt = ANALYSIS_PROMPT.format(
                cv_content=profile_text,
                me_section=me_section,
                job_content=job_description,
            )
            try:
                if stream_output:
                    print()
                    stream = client_lm.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=4096,
                        stream=True,
                    )
                    full_text = ""
                    input_tokens = output_tokens = 0
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            print(delta, end="", flush=True)
                            full_text += delta
                    print()
                else:
                    response = client_lm.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=4096,
                    )
                    full_text = response.choices[0].message.content or ""
                    input_tokens = response.usage.prompt_tokens if response.usage else 0
                    output_tokens = response.usage.completion_tokens if response.usage else 0
                model_used = f"LM Studio – {model} (Profil)"
            except Exception as exc:
                logger.warning("LM Studio fehlgeschlagen (%s) – Fallback auf Anthropic.", exc)
                import anthropic as _ac
                client_an = _ac.Anthropic()
                resp = client_an.messages.create(
                    model=ANTHROPIC_MODEL, max_tokens=4096,
                    thinking={"type": "adaptive"},
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                full_text = next((b.text for b in resp.content if b.type == "text"), "")
                input_tokens, output_tokens = resp.usage.input_tokens, resp.usage.output_tokens
                model_used = f"Anthropic – {ANTHROPIC_MODEL} (Profil, Fallback)"
        else:
            import anthropic as _ac
            client_an = _ac.Anthropic()
            prompt = ANALYSIS_PROMPT.format(
                cv_content=profile_text,
                me_section=me_section,
                job_content=job_description,
            )
            if stream_output:
                print()
                full_text = ""
                input_tokens = output_tokens = 0
                with client_an.messages.stream(
                    model=ANTHROPIC_MODEL, max_tokens=4096,
                    thinking={"type": "adaptive"},
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    for event in stream:
                        if event.type == "content_block_delta" and event.delta.type == "text_delta":
                            print(event.delta.text, end="", flush=True)
                            full_text += event.delta.text
                    final = stream.get_final_message()
                    input_tokens = final.usage.input_tokens
                    output_tokens = final.usage.output_tokens
                print()
            else:
                resp = client_an.messages.create(
                    model=ANTHROPIC_MODEL, max_tokens=4096,
                    thinking={"type": "adaptive"},
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                full_text = next((b.text for b in resp.content if b.type == "text"), "")
                input_tokens, output_tokens = resp.usage.input_tokens, resp.usage.output_tokens
            model_used = f"Anthropic – {ANTHROPIC_MODEL} (Profil)"
    elif backend == "lmstudio":
        base_url = analyzer_cfg.get("lmstudio_url", "http://localhost:1234/v1")
        model = analyzer_cfg.get("lmstudio_model", "local-model")
        logger.info("Backend: LM Studio (%s, Modell: %s)", base_url, model)
        try:
            full_text, input_tokens, output_tokens = _analyze_with_lmstudio(
                cv_path, me_section, job_description, stream_output, base_url, model
            )
            model_used = f"LM Studio – {model}"
        except Exception as exc:
            logger.warning("LM Studio fehlgeschlagen (%s) – Fallback auf Anthropic Claude.", exc)
            print(f"\n[Fallback] LM Studio nicht erreichbar: {exc}\nVerwende Claude ...\n")
            full_text, input_tokens, output_tokens = _analyze_with_anthropic(
                cv_path, me_section, job_description, stream_output
            )
            model_used = f"Anthropic – {ANTHROPIC_MODEL} (Fallback)"
    else:
        logger.info("Backend: Anthropic Claude (%s)", ANTHROPIC_MODEL)
        full_text, input_tokens, output_tokens = _analyze_with_anthropic(
            cv_path, me_section, job_description, stream_output
        )
        model_used = f"Anthropic – {ANTHROPIC_MODEL}"

    fit_score = _extract_fit_score(full_text)
    extracted_title, extracted_company, candidate_level, job_level = _extract_metadata(full_text)

    return AnalysisResult(
        fit_score=fit_score,
        full_analysis=full_text,
        job_title=job_title or extracted_title,
        company=company or extracted_company,
        candidate_level=candidate_level,
        job_level=job_level,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_used=model_used,
    )
