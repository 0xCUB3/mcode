from __future__ import annotations

import json

from mcode.llm.session import edits_to_patch


def test_edits_to_patch_basic(tmp_path):
    (tmp_path / "foo.py").write_text("def hello():\n    return 'hi'\n")
    raw = json.dumps(
        {
            "edits": [
                {
                    "file": "foo.py",
                    "search": "return 'hi'",
                    "replace": "return 'hello'",
                }
            ]
        }
    )
    patch, errors = edits_to_patch(raw, repo_root=str(tmp_path))
    assert "--- a/foo.py" in patch
    assert "+++ b/foo.py" in patch
    assert "-    return 'hi'" in patch
    assert "+    return 'hello'" in patch
    assert errors == []


def test_edits_to_patch_multiple_files(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    raw = json.dumps(
        {
            "edits": [
                {"file": "a.py", "search": "x = 1", "replace": "x = 10"},
                {"file": "b.py", "search": "y = 2", "replace": "y = 20"},
            ]
        }
    )
    patch, errors = edits_to_patch(raw, repo_root=str(tmp_path))
    assert "--- a/a.py" in patch
    assert "--- a/b.py" in patch
    assert errors == []


def test_edits_to_patch_missing_file(tmp_path):
    raw = json.dumps(
        {
            "edits": [
                {"file": "missing.py", "search": "x", "replace": "y"},
            ]
        }
    )
    patch, errors = edits_to_patch(raw, repo_root=str(tmp_path))
    assert patch == ""
    assert len(errors) == 1
    assert "File not found" in errors[0]


def test_edits_to_patch_search_not_found(tmp_path):
    (tmp_path / "foo.py").write_text("def hello():\n    pass\n")
    raw = json.dumps(
        {
            "edits": [
                {"file": "foo.py", "search": "no match here", "replace": "y"},
            ]
        }
    )
    patch, errors = edits_to_patch(raw, repo_root=str(tmp_path))
    assert patch == ""
    assert len(errors) == 1
    assert "Search text not found" in errors[0]


def test_edits_to_patch_fallback_raw_diff():
    raw = json.dumps({"patch": "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"})
    patch, errors = edits_to_patch(raw)
    assert patch.startswith("--- a/foo.py")
    assert errors == []


def test_edits_to_patch_invalid_json():
    patch, errors = edits_to_patch("not json")
    assert patch == ""
    assert errors == []


def test_edits_to_patch_empty_edits_no_fallback():
    raw = json.dumps({"edits": []})
    patch, errors = edits_to_patch(raw)
    assert patch == ""
    assert errors == []


def test_edits_to_patch_replaces_first_occurrence_only(tmp_path):
    (tmp_path / "foo.py").write_text("x = 1\nx = 1\n")
    raw = json.dumps(
        {
            "edits": [
                {"file": "foo.py", "search": "x = 1", "replace": "x = 2"},
            ]
        }
    )
    patch, errors = edits_to_patch(raw, repo_root=str(tmp_path))
    assert patch.count("+x = 2") == 1
    assert patch.count("-x = 1") == 1
    assert errors == []


def test_fuzzy_path_strips_bogus_prefix(tmp_path):
    sub = tmp_path / "pylint" / "checkers"
    sub.mkdir(parents=True)
    (sub / "base.py").write_text("x = 1\n")
    raw = json.dumps(
        {
            "edits": [
                {
                    "file": "pylint/src/pylint/checkers/base.py",
                    "search": "x = 1",
                    "replace": "x = 2",
                }
            ]
        }
    )
    patch, errors = edits_to_patch(raw, repo_root=str(tmp_path))
    assert "--- a/pylint/checkers/base.py" in patch
    assert "-x = 1" in patch
    assert "+x = 2" in patch
    assert errors == []


def test_fuzzy_path_basename_fallback(tmp_path):
    sub = tmp_path / "src" / "deep" / "nested"
    sub.mkdir(parents=True)
    (sub / "utils.py").write_text("y = 2\n")
    raw = json.dumps(
        {"edits": [{"file": "completely/wrong/utils.py", "search": "y = 2", "replace": "y = 3"}]}
    )
    patch, errors = edits_to_patch(raw, repo_root=str(tmp_path))
    assert "+y = 3" in patch
    assert errors == []


def test_fuzzy_search_text(tmp_path):
    content = "def hello():\n    x = 1\n    y = 2\n    return x + y\n"
    (tmp_path / "foo.py").write_text(content)
    raw = json.dumps(
        {
            "edits": [
                {
                    "file": "foo.py",
                    "search": "def hello():\n    x = 1\n    y = 3\n    return x + y\n",
                    "replace": "def hello():\n    x = 10\n    y = 20\n    return x + y\n",
                }
            ]
        }
    )
    patch, errors = edits_to_patch(raw, repo_root=str(tmp_path))
    assert "--- a/foo.py" in patch
    assert "+    x = 10" in patch
    assert errors == []


def test_fuzzy_search_too_different_is_skipped(tmp_path):
    (tmp_path / "foo.py").write_text("def hello():\n    pass\n")
    raw = json.dumps(
        {
            "edits": [
                {
                    "file": "foo.py",
                    "search": "completely unrelated gibberish text that matches nothing",
                    "replace": "x = 1",
                }
            ]
        }
    )
    patch, errors = edits_to_patch(raw, repo_root=str(tmp_path))
    assert patch == ""
    assert len(errors) == 1
    assert "Search text not found" in errors[0]


def test_error_feedback_includes_file_path(tmp_path):
    raw = json.dumps(
        {
            "edits": [
                {"file": "nonexistent/path/foo.py", "search": "x", "replace": "y"},
            ]
        }
    )
    _, errors = edits_to_patch(raw, repo_root=str(tmp_path))
    assert "nonexistent/path/foo.py" in errors[0]


def test_suggest_paths_on_missing_file(tmp_path):
    sub = tmp_path / "pylint" / "checkers"
    sub.mkdir(parents=True)
    (sub / "base_checker.py").write_text("x = 1\n")
    raw = json.dumps(
        {
            "edits": [
                {
                    "file": "src/pylint/checkers/base_checker.py",
                    "search": "x = 1",
                    "replace": "x = 2",
                }
            ]
        }
    )
    patch, errors = edits_to_patch(raw, repo_root=str(tmp_path))
    # The fuzzy prefix-strip resolves this, so it should succeed
    assert "--- a/pylint/checkers/base_checker.py" in patch
    assert errors == []


def test_suggest_paths_unresolvable_with_keywords(tmp_path):
    sub = tmp_path / "pylint" / "checkers"
    sub.mkdir(parents=True)
    (sub / "base_checker.py").write_text("x = 1\n")
    (sub / "messages_handler.py").write_text("y = 2\n")
    raw = json.dumps(
        {
            "edits": [
                {
                    "file": "wrong/path/messages_checker.py",
                    "search": "x = 1",
                    "replace": "x = 2",
                }
            ]
        }
    )
    _, errors = edits_to_patch(raw, repo_root=str(tmp_path))
    assert len(errors) == 1
    assert "Did you mean" in errors[0]


def test_partial_success_returns_both_patch_and_errors(tmp_path):
    (tmp_path / "good.py").write_text("x = 1\n")
    raw = json.dumps(
        {
            "edits": [
                {"file": "good.py", "search": "x = 1", "replace": "x = 2"},
                {"file": "bad.py", "search": "y = 1", "replace": "y = 2"},
            ]
        }
    )
    patch, errors = edits_to_patch(raw, repo_root=str(tmp_path))
    assert "--- a/good.py" in patch
    assert len(errors) == 1
    assert "File not found: bad.py" in errors[0]


def test_strict_mode_rejects_fuzzy_path_fallback(tmp_path):
    sub = tmp_path / "pkg"
    sub.mkdir(parents=True)
    (sub / "utils.py").write_text("y = 2\n")
    raw = json.dumps(
        {"edits": [{"file": "wrong/path/utils.py", "search": "y = 2", "replace": "y = 3"}]}
    )
    patch, errors = edits_to_patch(raw, repo_root=str(tmp_path), strict=True)
    assert patch == ""
    assert len(errors) == 1
    assert "File not found" in errors[0]


def test_strict_mode_rejects_fuzzy_search_match(tmp_path):
    (tmp_path / "foo.py").write_text("def hello():\n    x = 1\n    y = 2\n    return x + y\n")
    raw = json.dumps(
        {
            "edits": [
                {
                    "file": "foo.py",
                    "search": "def hello():\n    x = 1\n    y = 3\n    return x + y\n",
                    "replace": "def hello():\n    x = 10\n    y = 20\n    return x + y\n",
                }
            ]
        }
    )
    patch, errors = edits_to_patch(raw, repo_root=str(tmp_path), strict=True)
    assert patch == ""
    assert len(errors) == 1
    assert "Search text not found" in errors[0]


def test_strict_mode_requires_unique_search_text(tmp_path):
    (tmp_path / "foo.py").write_text("x = 1\nx = 1\n")
    raw = json.dumps(
        {
            "edits": [
                {"file": "foo.py", "search": "x = 1", "replace": "x = 2"},
            ]
        }
    )
    patch, errors = edits_to_patch(raw, repo_root=str(tmp_path), strict=True)
    assert patch == ""
    assert len(errors) == 1
    assert "must match exactly once" in errors[0]


def test_strict_mode_accepts_leading_slash_path(tmp_path):
    sub = tmp_path / "dynaconf" / "loaders"
    sub.mkdir(parents=True)
    (sub / "base.py").write_text("x = 1\n")
    raw = json.dumps(
        {
            "edits": [
                {"file": "/dynaconf/loaders/base.py", "search": "x = 1", "replace": "x = 2"},
            ]
        }
    )
    patch, errors = edits_to_patch(raw, repo_root=str(tmp_path), strict=True)
    assert "--- a/dynaconf/loaders/base.py" in patch
    assert "+x = 2" in patch
    assert errors == []
