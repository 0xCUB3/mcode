from __future__ import annotations

import re


def extract_lsf_job_id(value: object) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return text
    match = re.search(r"Job\s*<(?P<job_id>\d+)>", text)
    if match:
        return match.group("job_id")
    return None
