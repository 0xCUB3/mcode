#!/usr/bin/env bash
set -euo pipefail

# Exact run 5 chunk commands used for the failure-report-snippets sweep.

OPENAI_BASE_URL=http://127.0.0.1:18325/v1 OPENAI_API_KEY=dummy MCODE_CONTEXT_WINDOW=32768 MCODE_MAX_NEW_TOKENS=4096 MCODE_REACT_TIMEOUT=2400 uv run mcode bench aider-polyglot --model Qwen/Qwen3.6-35B-A3B --backend openai --temperature 0.3 --loop-budget 20 --task-ids python/bowling,python/connect,python/dot-dsl,python/food-chain,python/forth,python/grep,python/ledger,python/markdown,python/meetup,python/ocr-numbers,python/paasio,python/palindrome-products,python/pig-latin,python/poker,python/pov,python/protein-translation,python/pythagorean-triplet,python/rectangles,python/rest-api,python/robot-simulator,python/satellite,python/scale-generator,python/sgf-parsing,python/simple-cipher,python/simple-linked-list,python/spiral-matrix,python/sublist,python/tournament,python/tree-building,python/variable-length-quantity,python/word-count,python/word-search,python/wordy,python/zebra-puzzle --benchmark-root /Users/skula/Documents/polyglot-benchmark --db research/2026-04-24-aider-polyglot-workspace-context-full/run5-failure-reports/python.db

OPENAI_BASE_URL=http://127.0.0.1:18325/v1 OPENAI_API_KEY=dummy MCODE_CONTEXT_WINDOW=32768 MCODE_MAX_NEW_TOKENS=4096 MCODE_REACT_TIMEOUT=2400 uv run mcode bench aider-polyglot --model Qwen/Qwen3.6-35B-A3B --backend openai --temperature 0.3 --loop-budget 20 --task-ids python/affine-cipher,python/beer-song,python/book-store,python/bottle-song,python/dominoes,python/go-counting,python/grade-school,python/hangman,python/list-ops,python/phone-number,python/proverb,python/react,python/robot-name,python/transpose,python/two-bucket,python/zipper --benchmark-root /Users/skula/Documents/polyglot-benchmark --db research/2026-04-24-aider-polyglot-workspace-context-full/run5-failure-reports/python-remainder.db

OPENAI_BASE_URL=http://127.0.0.1:18325/v1 OPENAI_API_KEY=dummy MCODE_CONTEXT_WINDOW=32768 MCODE_MAX_NEW_TOKENS=4096 MCODE_REACT_TIMEOUT=2400 uv run mcode bench aider-polyglot --model Qwen/Qwen3.6-35B-A3B --backend openai --temperature 0.3 --loop-budget 20 --task-ids go/alphametics,go/beer-song,go/book-store,go/bottle-song,go/bowling,go/connect,go/counter,go/crypto-square,go/dnd-character,go/dominoes,go/error-handling,go/food-chain,go/forth,go/hexadecimal,go/kindergarten-garden,go/ledger,go/markdown,go/matrix,go/octal,go/paasio,go/palindrome-products,go/pig-latin,go/poker,go/pov,go/protein-translation,go/react,go/robot-simulator,go/say,go/scale-generator,go/simple-linked-list,go/sublist,go/transpose,go/tree-building,go/trinary,go/two-bucket,go/variable-length-quantity,go/word-search,go/wordy,go/zebra-puzzle --benchmark-root /Users/skula/Documents/polyglot-benchmark --db research/2026-04-24-aider-polyglot-workspace-context-full/run5-failure-reports/go.db

OPENAI_BASE_URL=http://127.0.0.1:18325/v1 OPENAI_API_KEY=dummy MCODE_CONTEXT_WINDOW=32768 MCODE_MAX_NEW_TOKENS=4096 MCODE_REACT_TIMEOUT=2400 uv run mcode bench aider-polyglot --model Qwen/Qwen3.6-35B-A3B --backend openai --temperature 0.3 --loop-budget 20 --task-ids cpp/all-your-base,cpp/allergies,cpp/bank-account,cpp/binary-search-tree,cpp/circular-buffer,cpp/clock,cpp/complex-numbers,cpp/crypto-square,cpp/diamond,cpp/dnd-character,cpp/gigasecond,cpp/grade-school,cpp/kindergarten-garden,cpp/knapsack,cpp/linked-list,cpp/meetup,cpp/parallel-letter-frequency,cpp/perfect-numbers,cpp/phone-number,cpp/queen-attack,cpp/robot-name,cpp/space-age,cpp/spiral-matrix,cpp/sublist,cpp/yacht,cpp/zebra-puzzle --benchmark-root /Users/skula/Documents/polyglot-benchmark --db research/2026-04-24-aider-polyglot-workspace-context-full/run5-failure-reports/cpp.db

OPENAI_BASE_URL=http://127.0.0.1:18325/v1 OPENAI_API_KEY=dummy MCODE_CONTEXT_WINDOW=32768 MCODE_MAX_NEW_TOKENS=4096 MCODE_REACT_TIMEOUT=2400 uv run mcode bench aider-polyglot --model Qwen/Qwen3.6-35B-A3B --backend openai --temperature 0.3 --loop-budget 20 --task-ids rust/accumulate,rust/acronym,rust/alphametics,rust/book-store,rust/bowling,rust/decimal,rust/dot-dsl,rust/doubly-linked-list,rust/fizzy,rust/forth,rust/gigasecond,rust/grade-school,rust/grep,rust/luhn-from,rust/macros,rust/nucleotide-codons,rust/ocr-numbers,rust/parallel-letter-frequency,rust/pig-latin,rust/poker,rust/react,rust/robot-name,rust/say,rust/scale-generator,rust/simple-cipher,rust/two-bucket,rust/variable-length-quantity,rust/word-count,rust/wordy,rust/xorcism --benchmark-root /Users/skula/Documents/polyglot-benchmark --db research/2026-04-24-aider-polyglot-workspace-context-full/run5-failure-reports/rust.db

OPENAI_BASE_URL=http://127.0.0.1:18325/v1 OPENAI_API_KEY=dummy MCODE_CONTEXT_WINDOW=32768 MCODE_MAX_NEW_TOKENS=4096 MCODE_REACT_TIMEOUT=2400 uv run mcode bench aider-polyglot --model Qwen/Qwen3.6-35B-A3B --backend openai --temperature 0.3 --loop-budget 20 --task-ids javascript/affine-cipher,javascript/alphametics,javascript/beer-song,javascript/binary,javascript/book-store,javascript/bottle-song,javascript/bowling,javascript/complex-numbers,javascript/connect,javascript/food-chain,javascript/forth,javascript/go-counting,javascript/grade-school,javascript/grep,javascript/house,javascript/killer-sudoku-helper,javascript/ledger,javascript/list-ops,javascript/meetup,javascript/ocr-numbers,javascript/palindrome-products,javascript/parallel-letter-frequency,javascript/phone-number,javascript/pig-latin,javascript/poker,javascript/promises,javascript/queen-attack,javascript/rational-numbers,javascript/react,javascript/rectangles,javascript/resistor-color-trio,javascript/rest-api,javascript/robot-name,javascript/say,javascript/scale-generator,javascript/simple-linked-list,javascript/space-age,javascript/state-of-tic-tac-toe,javascript/sum-of-multiples,javascript/tournament,javascript/transpose,javascript/triangle,javascript/twelve-days,javascript/two-bucket,javascript/variable-length-quantity,javascript/word-search,javascript/wordy,javascript/zebra-puzzle,javascript/zipper --benchmark-root /Users/skula/Documents/polyglot-benchmark --db research/2026-04-24-aider-polyglot-workspace-context-full/run5-failure-reports/javascript.db

OPENAI_BASE_URL=http://127.0.0.1:18325/v1 OPENAI_API_KEY=dummy MCODE_CONTEXT_WINDOW=32768 MCODE_MAX_NEW_TOKENS=4096 MCODE_REACT_TIMEOUT=2400 uv run mcode bench aider-polyglot --model Qwen/Qwen3.6-35B-A3B --backend openai --temperature 0.3 --loop-budget 20 --task-ids java/affine-cipher,java/all-your-base,java/alphametics,java/bank-account,java/book-store,java/bottle-song,java/bowling,java/change,java/circular-buffer,java/connect,java/custom-set,java/dominoes,java/food-chain,java/forth,java/go-counting,java/hangman,java/house,java/kindergarten-garden,java/ledger,java/mazy-mice,java/ocr-numbers,java/palindrome-products,java/phone-number,java/pig-latin,java/poker,java/pov,java/protein-translation,java/pythagorean-triplet,java/queen-attack,java/rational-numbers,java/react,java/resistor-color-trio,java/rest-api,java/satellite,java/series,java/sgf-parsing,java/simple-linked-list,java/state-of-tic-tac-toe,java/transpose,java/tree-building,java/twelve-days,java/two-bucket,java/variable-length-quantity,java/word-search,java/wordy,java/zebra-puzzle,java/zipper --benchmark-root /Users/skula/Documents/polyglot-benchmark --db research/2026-04-24-aider-polyglot-workspace-context-full/run5-failure-reports/java.db

uv run python - <<'PY'
import sqlite3
from pathlib import Path
folder = Path('research/2026-04-24-aider-polyglot-workspace-context-full/run5-failure-reports')
paths = [folder/'python.db', folder/'python-remainder.db', folder/'go.db', folder/'cpp.db', folder/'rust.db', folder/'javascript.db', folder/'java.db']
out = folder/'results.db'
if out.exists():
    out.unlink()
src = sqlite3.connect(paths[0])
dst = sqlite3.connect(out)
for (sql,) in src.execute("select sql from sqlite_master where type='table' and name in ('runs','task_results') order by name"):
    dst.execute(sql)
run_cols = [r[1] for r in src.execute('pragma table_info(runs)')]
task_cols = [r[1] for r in src.execute('pragma table_info(task_results)')]
run = dict(zip(run_cols, src.execute('select * from runs limit 1').fetchone()))
run['id'] = 1
dst.execute(f"insert into runs ({','.join(run_cols)}) values ({','.join('?' for _ in run_cols)})", [run.get(c) for c in run_cols])
seen = set()
row_id = 1
for path in paths:
    con = sqlite3.connect(path)
    for values in con.execute('select * from task_results order by task_id'):
        row = dict(zip(task_cols, values))
        if row['task_id'] in seen:
            continue
        seen.add(row['task_id'])
        row['id'] = row_id
        row['run_id'] = 1
        row_id += 1
        dst.execute(f"insert into task_results ({','.join(task_cols)}) values ({','.join('?' for _ in task_cols)})", [row.get(c) for c in task_cols])
dst.commit()
PY
