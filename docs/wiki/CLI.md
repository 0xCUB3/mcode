# CLI Reference

## `mcode --help`

```text
                                                                                                                        
 Usage: mcode [OPTIONS] COMMAND [ARGS]...                                                                               
                                                                                                                        
 mCode benchmarking harness.                                                                                            
                                                                                                                        
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --verbose  -v        Show Mellea INFO logs                                                                           │
│ --help               Show this message and exit.                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ results   Query pass rates from the results DB.                                                                      │
│ bench                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## `mcode results --help`

```text
                                                                                                                        
 Usage: mcode results [OPTIONS]                                                                                         
                                                                                                                        
 Query pass rates from the results DB.                                                                                  
                                                                                                                        
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --db                     PATH                  SQLite DB path [default: experiments/results/results.db]              │
│ --benchmark              TEXT                                                                                        │
│ --model                  TEXT                                                                                        │
│ --backend                TEXT                                                                                        │
│ --samples                INTEGER RANGE [x>=1]                                                                        │
│ --debug-iters            INTEGER RANGE [x>=0]                                                                        │
│ --timeout                INTEGER RANGE [x>=1]                                                                        │
│ --compare-samples                                                                                                    │
│ --retrieval              TEXT                                                                                        │
│ --help                                         Show this message and exit.                                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## `mcode bench --help`

```text
                                                                                                                        
 Usage: mcode bench [OPTIONS] COMMAND [ARGS]...                                                                         
                                                                                                                        
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                          │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ humaneval                                                                                                            │
│ mbpp                                                                                                                 │
│ swebench-lite                                                                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## `mcode bench humaneval --help`

```text
                                                                                                                        
 Usage: mcode bench humaneval [OPTIONS]                                                                                 
                                                                                                                        
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --model                            TEXT                  Mellea model id [required]                               │
│    --backend                          TEXT                  Mellea backend name [default: ollama]                    │
│    --samples                          INTEGER RANGE [x>=1]  [default: 1]                                             │
│    --debug-iters                      INTEGER RANGE [x>=0]  [default: 0]                                             │
│    --timeout                          INTEGER RANGE [x>=1]  [default: 60]                                            │
│    --retrieval      --no-retrieval                          [default: no-retrieval]                                  │
│    --db                               PATH                  [default: experiments/results/results.db]                │
│    --limit                            INTEGER RANGE [x>=1]                                                           │
│    --help                                                   Show this message and exit.                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## `mcode bench mbpp --help`

```text
                                                                                                                        
 Usage: mcode bench mbpp [OPTIONS]                                                                                      
                                                                                                                        
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --model                            TEXT                  Mellea model id [required]                               │
│    --backend                          TEXT                  Mellea backend name [default: ollama]                    │
│    --samples                          INTEGER RANGE [x>=1]  [default: 1]                                             │
│    --debug-iters                      INTEGER RANGE [x>=0]  [default: 0]                                             │
│    --timeout                          INTEGER RANGE [x>=1]  [default: 60]                                            │
│    --retrieval      --no-retrieval                          [default: no-retrieval]                                  │
│    --db                               PATH                  [default: experiments/results/results.db]                │
│    --limit                            INTEGER RANGE [x>=1]                                                           │
│    --help                                                   Show this message and exit.                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
