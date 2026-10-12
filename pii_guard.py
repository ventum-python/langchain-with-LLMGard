"""
pii_guard.py — Local PII guard (Microsoft Presidio + Indian regex)

LLM / Pinecone ko raw GST, PAN, Aadhaar, bank, phone, email, names
bhejne se pehle mask karta hai. Terminal me report print hota hai:
kya BLOCK hua, kya LLM ko gaya.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger("SmartChatbot")

# Optional Presidio — agar install nahi hai to regex fallback
_PRESIDIO_OK = False
try:
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    _PRESIDIO_OK = True
except Exception as e:
    logger.warning(f"⚠️ Presidio import failed, regex fallback on: {e}")


# Company / city / dates LLM ko jaane do — sirf secrets block
BLOCK_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IN_PAN",
    "IN_GSTIN",
    "IN_AADHAAR",
    "IN_IFSC",
    "IN_PHONE",
]

SKIP_PERSON = {
    "django", "redis", "postman", "docker", "python", "fastapi", "linux",
    "whatsapp", "firebase", "nginx", "github", "linkedin", "portfolio",
    "balance sheet", "dev tools", "vector databases", "zoom api",
    "zoho books api", "signulu", "jharkhand", "nagaland", "bill",
}

INDIAN_PATTERNS = {
    "IN_PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "IN_GSTIN": r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b",
    "IN_AADHAAR": r"\b[2-9][0-9]{3}[\s-]?[0-9]{4}[\s-]?[0-9]{4}\b",
    "IN_IFSC": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
    "IN_BANK_ACCOUNT": r"(?i)(?:account|a/c|acc(?:ount)?\s*(?:no\.?|number|#)?|bank)\s*[:\-]?\s*(\d{9,18})",
    "IN_PHONE": r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
    "EMAIL_ADDRESS": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,19}\b",
}


@dataclass
class Finding:
    entity: str
    original: str
    replacement: str
    start: int
    end: int


@dataclass
class GuardResult:
    original: str
    redacted: str
    findings: List[Finding] = field(default_factory=list)

    @property
    def blocked(self) -> List[Finding]:
        return self.findings


class PIIGuard:
    def __init__(self):
        self.analyzer = None
        self.anonymizer = None
        if _PRESIDIO_OK:
            try:
                nlp = NlpEngineProvider(nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                }).create_engine()
                self.analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
                self.anonymizer = AnonymizerEngine()
                self._register_indian_recognizers()
                logger.info("🛡️ PII Guard ready (Microsoft Presidio + Indian patterns)")
            except Exception as e:
                logger.warning(f"⚠️ Presidio engine failed ({e}). Using regex fallback.")
                self.analyzer = None
        if not self.analyzer:
            logger.info("🛡️ PII Guard ready (regex fallback)")

    def _register_indian_recognizers(self):
        specs = [
            ("IN_PAN", INDIAN_PATTERNS["IN_PAN"], 0.7),
            ("IN_GSTIN", INDIAN_PATTERNS["IN_GSTIN"], 0.8),
            ("IN_AADHAAR", INDIAN_PATTERNS["IN_AADHAAR"], 0.55),
            ("IN_IFSC", INDIAN_PATTERNS["IN_IFSC"], 0.7),
            ("IN_PHONE", INDIAN_PATTERNS["IN_PHONE"], 0.5),
        ]
        for name, regex, score in specs:
            recognizer = PatternRecognizer(
                supported_entity=name,
                patterns=[Pattern(name=name, regex=regex, score=score)],
            )
            self.analyzer.registry.add_recognizer(recognizer)

    def redact(self, text: str, stage: str) -> GuardResult:
        if not text:
            result = GuardResult(original=text or "", redacted=text or "")
            self.log_report(result, stage)
            return result

        if self.analyzer and self.anonymizer:
            result = self._redact_presidio(text)
        else:
            result = self._redact_regex(text)

        self.log_report(result, stage)
        return result

    def _redact_presidio(self, text: str) -> GuardResult:
        results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=BLOCK_ENTITIES,
        )
        cleaned = []
        for r in results:
            span = text[r.start:r.end].strip()
            if r.entity_type == "PERSON" and (
                r.score < 0.6 or span.lower() in SKIP_PERSON or len(span) < 3
            ):
                continue
            cleaned.append(r)
        results = sorted(cleaned, key=lambda r: (r.start, -(r.end - r.start)))
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)

        findings = []
        for r in results:
            original = text[r.start:r.end]
            replacement = f"<{r.entity_type}>"
            findings.append(
                Finding(r.entity_type, original, replacement, r.start, r.end)
            )

        # Extra Indian regex Presidio miss kar de to
        extra = self._redact_regex(anonymized.text)
        merged = findings + [
            f for f in extra.findings
            if f.original not in {x.original for x in findings}
        ]
        return GuardResult(original=text, redacted=extra.redacted, findings=merged)

    def _redact_regex(self, text: str) -> GuardResult:
        findings: List[Finding] = []
        redacted = text

        def collect(entity: str, pattern: str, group: int = 0):
            nonlocal redacted
            for m in re.finditer(pattern, text):
                original = m.group(group)
                replacement = f"<{entity}>"
                findings.append(
                    Finding(entity, original, replacement, m.start(group), m.end(group))
                )
                redacted = redacted.replace(original, replacement)

        for entity, pattern in INDIAN_PATTERNS.items():
            group = 1 if entity == "IN_BANK_ACCOUNT" else 0
            collect(entity, pattern, group)

        # de-dupe overlapping same span
        uniq = []
        seen = set()
        for f in findings:
            key = (f.entity, f.original)
            if key not in seen:
                seen.add(key)
                uniq.append(f)
        return GuardResult(original=text, redacted=redacted, findings=uniq)

    def log_report(self, result: GuardResult, stage: str):
        line = "=" * 64
        logger.info(line)
        logger.info("🛡️  PII GUARD REPORT")
        logger.info(f"📍 Stage: {stage}")
        logger.info(line)

        if not result.findings:
            logger.info("✅ No sensitive info found. Full text LLM/store ko ja sakta hai.")
            preview = result.redacted[:400].replace("\n", " ")
            logger.info(f"📤 Sent preview: {preview}")
            logger.info(line)
            return

        logger.info("🚫 BLOCKED — ye LLM ko NAHI diya:")
        for f in result.findings:
            logger.info(f"   • {f.entity:<16} {f.original}  →  {f.replacement}")

        blocked = ", ".join(sorted({f.original for f in result.findings}))
        logger.info(f"❌ NOT sent: {blocked}")

        preview = result.redacted[:800]
        logger.info("✅ SENT to LLM / next step (redacted):")
        for chunk in preview.splitlines() or [preview]:
            logger.info(f"   {chunk}")
        if len(result.redacted) > 800:
            logger.info("   … (truncated)")
        logger.info(line)


guard = PIIGuard()
