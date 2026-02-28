from __future__ import annotations

import json

from mcode.llm.session import line_edits_to_patch


def test_basic_line_edit(tmp_path):
    (tmp_path / "foo.py").write_text("line1\nline2\nline3\n")
    raw = json.dumps(
        {"edits": [{"file": "foo.py", "start_line": 2, "end_line": 2, "replace": "replaced\n"}]}
    )
    patch, errors = line_edits_to_patch(raw, repo_root=str(tmp_path))
    assert errors == []
    assert "-line2" in patch
    assert "+replaced" in patch


def test_multi_line_replace(tmp_path):
    (tmp_path / "foo.py").write_text("a\nb\nc\nd\n")
    raw = json.dumps(
        {"edits": [{"file": "foo.py", "start_line": 2, "end_line": 3, "replace": "x\ny\nz\n"}]}
    )
    patch, errors = line_edits_to_patch(raw, repo_root=str(tmp_path))
    assert errors == []
    assert "-b" in patch
    assert "-c" in patch
    assert "+x" in patch
    assert "+z" in patch


def test_file_not_found(tmp_path):
    raw = json.dumps(
        {"edits": [{"file": "nope.py", "start_line": 1, "end_line": 1, "replace": "x\n"}]}
    )
    patch, errors = line_edits_to_patch(raw, repo_root=str(tmp_path))
    assert patch == ""
    assert "File not found" in errors[0]


def test_invalid_line_range(tmp_path):
    (tmp_path / "foo.py").write_text("a\nb\n")
    raw = json.dumps(
        {"edits": [{"file": "foo.py", "start_line": 5, "end_line": 6, "replace": "x\n"}]}
    )
    patch, errors = line_edits_to_patch(raw, repo_root=str(tmp_path))
    assert patch == ""
    assert "Invalid line range" in errors[0]


def test_invalid_json():
    patch, errors = line_edits_to_patch("not json", repo_root="/tmp")
    assert patch == ""
    assert errors == []


def test_empty_edits():
    patch, errors = line_edits_to_patch(json.dumps({"edits": []}), repo_root="/tmp")
    assert patch == ""


def test_normalizes_path(tmp_path):
    (tmp_path / "foo.py").write_text("line1\n")
    raw = json.dumps(
        {"edits": [{"file": "./foo.py", "start_line": 1, "end_line": 1, "replace": "new\n"}]}
    )
    patch, errors = line_edits_to_patch(raw, repo_root=str(tmp_path))
    assert errors == []
    assert "+new" in patch


def test_leading_slash_path(tmp_path):
    (tmp_path / "foo.py").write_text("line1\n")
    raw = json.dumps(
        {"edits": [{"file": "/foo.py", "start_line": 1, "end_line": 1, "replace": "new\n"}]}
    )
    patch, errors = line_edits_to_patch(raw, repo_root=str(tmp_path))
    assert errors == []
    assert "+new" in patch


def test_replace_without_trailing_newline(tmp_path):
    (tmp_path / "foo.py").write_text("a\nb\nc\n")
    raw = json.dumps(
        {"edits": [{"file": "foo.py", "start_line": 2, "end_line": 2, "replace": "replaced"}]}
    )
    patch, errors = line_edits_to_patch(raw, repo_root=str(tmp_path))
    assert errors == []
    assert "+replaced" in patch
