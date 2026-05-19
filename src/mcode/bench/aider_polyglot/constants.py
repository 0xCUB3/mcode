from __future__ import annotations

import re

BENCHMARK_REPO = "https://github.com/Aider-AI/polyglot-benchmark.git"
JS_SKIP_MARKER_RE = re.compile(r"\b(xit|xtest|xdescribe)\s*\(")
JAVA_DISABLED_RE = re.compile(r"^[ \t]*@Disabled\b.*$", re.MULTILINE)
LANGUAGE_ORDER = ("python", "go", "rust", "javascript", "cpp", "java")
