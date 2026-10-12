"""File-scope: summary/overview pe pehle file choose, phir sirf usi file."""

import re
from typing import Dict, List, Optional

TASK_RE = re.compile(
    r"(summar[yi]s?e?|summary|summery|samary|overview|tl;?dr|recap|gist|"
    r"file ka summary|iska summary|document (ka )?(summary|overview)|"
    r"short me bata|brief me|explain (this|the) (file|doc|document))",
    re.I,
)

ALL_RE = re.compile(
    r"\b(all files|every file|sabhi files?|sab files?|saari files?|dono files?)\b",
    re.I,
)


def is_file_task(question: str) -> bool:
    return bool(TASK_RE.search(question or ""))


def match_file(question: str, files: List[str]) -> Optional[str]:
    q = (question or "").lower()
    # longest name first so 'resume_v2.pdf' beats 'resume.pdf'
    for name in sorted(files, key=len, reverse=True):
        low = name.lower()
        stem = low.rsplit(".", 1)[0]
        if low in q or (len(stem) > 3 and stem in q):
            return name
    return None


def resolve_file_scope(
    question: str,
    files: List[str],
    requested: Optional[str] = None,
    active_file: Optional[str] = None,
) -> Dict:
    files = [f for f in files if f]

    if requested and requested in files:
        return {"source": requested, "pick": False, "files": files, "action": "summary", "pin": requested}

    if ALL_RE.search(question or ""):
        return {"source": None, "pick": False, "files": files, "action": "summary", "pin": None, "clear": True}

    named = match_file(question, files)
    if named:
        return {"source": named, "pick": False, "files": files, "action": "summary", "pin": named}

    if is_file_task(question) and len(files) > 1 and not active_file:
        return {
            "source": None,
            "pick": True,
            "files": files,
            "action": "summary",
            "pin": None,
        }

    if is_file_task(question) and len(files) == 1:
        return {"source": files[0], "pick": False, "files": files, "action": "summary", "pin": files[0]}

    if active_file and active_file in files:
        return {"source": active_file, "pick": False, "files": files, "action": "chat", "pin": active_file}

    return {"source": None, "pick": False, "files": files, "action": "chat", "pin": active_file}
