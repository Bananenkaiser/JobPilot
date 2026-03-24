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
- Begründe, warum du den Kandidaten als [Level] einschätzt (Berufserfahrung in Jahren, Verantwortungsbereiche, Technologietiefe).
- Passt das Level zur Stelle? Wenn nicht: wie groß ist die Abweichung und wie wirkt sie sich aus?

### 3. Stärken (was passt gut)
Liste die konkreten Übereinstimmungen zwischen Lebenslauf und Stelle auf.

### 4. Lücken (was fehlt oder ist schwach)
Liste die Anforderungen der Stelle, die im Lebenslauf fehlen oder unzureichend dargestellt sind.

### 5. Konkrete Lebenslauf-Optimierungen für diese Stelle
Zeige für jede Lücke/Schwäche **konkret**, was ich im Lebenslauf ändern oder ergänzen sollte.
Format: „Abschnitt X: [aktuelle Formulierung] → [optimierte Formulierung]" oder „Fehlend: [was ergänzen und wo]".

### 6. Empfehlung
Soll ich mich bewerben? Mit welcher Strategie?"""


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


def analyze_job(
    job_description: str,
    cv_path: Path,
    me_path: Path | None = None,
    job_title: str = "",
    company: str = "",
    stream_output: bool = True,
    config: dict | None = None,
) -> AnalysisResult:
    """Analysiert die Passung zwischen Stellenausschreibung und Lebenslauf.

    Wählt automatisch das Backend aus config['analyzer']['backend']:
    - 'anthropic' (Standard): Claude API
    - 'lmstudio': lokales LM Studio über OpenAI-kompatiblen Endpunkt
    """
    logger.info("Lebenslauf: %s", cv_path.name)

    me_section = ""
    if me_path and me_path.exists():
        me_text = me_path.read_text(encoding="utf-8", errors="replace").strip()
        if me_text:
            me_section = f"\n## Weitere persönliche Informationen\n{me_text}\n"

    analyzer_cfg = (config or {}).get("analyzer", {})
    backend = analyzer_cfg.get("backend", "anthropic")

    model_used = ""
    if backend == "lmstudio":
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
