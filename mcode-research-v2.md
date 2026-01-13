# mCode: Small Model Agentic Coding Research

**IBM Watson AI Lab UROP Project**

---

## Executive Summary

mCode is a research project exploring whether small local LLMs (1B-8B parameters) can achieve near-frontier performance on agentic coding tasks through inference-time techniques—without requiring end-user fine-tuning.

The approach is evidence-driven: build benchmarking infrastructure first, run ablations to determine which techniques actually matter, then build a user-facing tool using only validated techniques.

---

## Part 1: Research Findings

### 1.1 Inference-Time Scaling

The "Large Language Monkeys" paper (Brown et al., 2024) found that coverage scales log-linearly with sample count over four orders of magnitude. On SWE-bench Lite, DeepSeek-Coder-V2 jumped from 15.9% (1 sample) to 56% (250 samples)—beating single-sample SOTA of 43%.

Key finding: on fixed compute budget, Llama-3-8B with many samples outperforms Llama-3-70B with fewer samples.

**Critical requirement:** automatic verification. Without execution feedback, majority voting plateaus at ~100 samples. With test-based verification, gains keep scaling.

| Technique | Improvement | Verification Required |
|-----------|-------------|----------------------|
| Best-of-N (50 samples) | 3-10× coverage | Yes |
| Self-debugging | 12% Pass@1 | Yes |
| Majority voting | Plateaus at ~100 | No |

### 1.2 Self-Debugging

ICLR 2024 Self-Debug paper shows up to 12% improvement on MBPP with unit test feedback. 3-5 iterations is the sweet spot.

Caveat for small models: they give poor self-feedback. The pattern that works is external verification—execute code, capture real errors, feed those back. Don't ask the model to evaluate itself.

### 1.3 Multi-Agent Decomposition

MapCoder (4 specialized agents) achieves 93.9% on HumanEval. More relevant: Llama 3.1 8B quantized to int4 shows 23% improvement with multi-agent decomposition vs single-agent.

Intuition: task decomposition reduces cognitive load per step, which benefits small models more than large ones.

### 1.4 Retrieval

Pinecone study: with sufficient retrieval, RAG equalizes performance across model sizes.

For code:
- AST-based chunking (tree-sitter) beats fixed-size by 5-6 points
- Hybrid retrieval (BM25 + dense) captures both exact matches and semantic similarity
- Selective retrieval: 80% of the time retrieval hurts. Learning when to retrieve gives 70% speedup with no accuracy loss

### 1.5 Existing Tool Patterns

| Tool | Key Pattern |
|------|-------------|
| Aider | Repo map via tree-sitter (compressed codebase representation) |
| Cline | Plan/Act mode separation, XML tool calls |
| Claude Code | Single-threaded loop, grep instead of embeddings |
| OpenHands | Fine-tuned 32B on agent trajectories, 7B in development |

### 1.6 Failure Modes

- Self-correction without execution feedback doesn't work for small models
- Long context (>8K tokens) degrades quality
- Tool calling reliability is poor out-of-the-box—needs constrained decoding

---

## Part 2: Mellea Integration

### What Mellea Provides

- Instruct-validate-repair loop with rejection sampling
- Structured output via Outlines (constrained decoding)
- Backend abstraction (Ollama, watsonx, HuggingFace, OpenAI)
- aLoRA adapters for lightweight task adaptation

### What We Build

- Benchmarking harness (SWE-Bench, HumanEval, MBPP)
- Docker sandbox for code execution
- Repository context (tree-sitter repo map, retrieval)
- Ablation framework for controlled experiments

---

## Part 3: Project Approach

### Philosophy

**Evidence-driven development.** Don't build features hoping they help—prove they help with benchmarks, then ship them.

```
Phase 1: Harness     →  Run benchmarks on pluggable models
Phase 2: Ablations   →  Which techniques actually matter?
Phase 3: Tool        →  Build Claude Code with validated techniques only
Phase 4: CI          →  Every change runs benchmarks, catch regressions
```

The harness doesn't get thrown away—it becomes the test suite for the tool.

---

## Part 4: Architecture

### 4.1 Tech Stack

```
Language:           Python 3.11+
CLI:                typer
LLM:                mellea
Code parsing:       tree-sitter (py-tree-sitter)
Embeddings:         sentence-transformers + CodeSage
Vector store:       faiss-cpu
Execution:          docker-py
Data:               SQLite + pandas (results storage/analysis)
Packaging:          uv
```

or anything better you find

### 4.3 CLI Interface

**Phase 1 & 2 (Primary):**

```bash
# Run a benchmark
mcode bench humaneval --model granite-8b --samples 5

# Run SWE-Bench subset
mcode bench swebench-lite --model qwen-7b --samples 10 --retrieval on

# Compare sample counts
mcode bench humaneval --model granite-8b --samples 1,5,10,20 --compare

# Run ablation experiment
mcode ablate --config experiments/configs/sample_scaling.yaml

# Analyze results
mcode analyze --experiment sample_scaling --output charts/
```

**Phase 3 (Later):**

```bash
# Interactive agent (built after we know what works)
mcode run "fix the bug in parser.py"
mcode chat
mcode index ./repo
```

### 4.4 Core Components

#### Benchmark Runner (`bench/runner.py`)

```python
from dataclasses import dataclass
from pathlib import Path
from mcode.llm.session import LLMSession
from mcode.execution.sandbox import DockerSandbox
from mcode.bench.tasks import Task, load_benchmark
from mcode.bench.results import ResultsDB

@dataclass
class BenchConfig:
    model_id: str
    samples: int = 1
    retrieval: bool = False
    max_debug_iterations: int = 3
    timeout: int = 60

@dataclass 
class BenchResult:
    task_id: str
    passed: bool
    samples_generated: int
    iterations_used: int
    time_ms: int
    error: str | None = None

class BenchmarkRunner:
    def __init__(self, config: BenchConfig):
        self.config = config
        self.llm = LLMSession(model_id=config.model_id)
        self.sandbox = DockerSandbox()
        self.results_db = ResultsDB()
    
    def run_benchmark(self, benchmark: str) -> list[BenchResult]:
        tasks = load_benchmark(benchmark)
        results = []
        
        for task in tasks:
            result = self.run_task(task)
            results.append(result)
            self.results_db.save(result, self.config)
        
        return results
    
    def run_task(self, task: Task) -> BenchResult:
        import time
        start = time.time()
        
        # Build context if retrieval enabled
        context = ""
        if self.config.retrieval and task.repo_path:
            context = self._build_context(task)
        
        # Generate with N samples (Mellea handles sampling)
        generation = self.llm.generate_code(
            task=task.prompt,
            context=context,
            samples=self.config.samples,
        )
        
        # Execute and debug loop
        code = generation.code
        for iteration in range(self.config.max_debug_iterations):
            result = self.sandbox.run_tests(
                code=code,
                test_command=task.test_command,
                timeout=self.config.timeout,
            )
            
            if result.success:
                return BenchResult(
                    task_id=task.id,
                    passed=True,
                    samples_generated=self.config.samples,
                    iterations_used=iteration + 1,
                    time_ms=int((time.time() - start) * 1000),
                )
            
            # Try to fix
            code = self.llm.debug_code(code, result.stderr, context).code
        
        return BenchResult(
            task_id=task.id,
            passed=False,
            samples_generated=self.config.samples,
            iterations_used=self.config.max_debug_iterations,
            time_ms=int((time.time() - start) * 1000),
            error=result.stderr,
        )
    
    def _build_context(self, task: Task) -> str:
        from mcode.context.repo_map import RepoMap
        from mcode.context.retriever import Retriever
        
        repo_map = RepoMap().build_map(task.repo_path)
        relevant = Retriever().search(task.prompt, task.repo_path, top_k=5)
        
        return f"{repo_map}\n\n{relevant}"
```

#### Ablation Runner (`ablation/runner.py`)

```python
import yaml
import itertools
from dataclasses import dataclass
from mcode.bench.runner import BenchmarkRunner, BenchConfig

@dataclass
class AblationConfig:
    name: str
    benchmark: str
    model_id: str
    vary: dict[str, list]  # param name -> values to try
    fixed: dict[str, any]  # params that stay constant

class AblationRunner:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        self.config = AblationConfig(**raw)
    
    def run(self):
        # Generate all combinations
        vary_keys = list(self.config.vary.keys())
        vary_values = list(self.config.vary.values())
        
        for combination in itertools.product(*vary_values):
            params = dict(zip(vary_keys, combination))
            params.update(self.config.fixed)
            params["model_id"] = self.config.model_id
            
            bench_config = BenchConfig(**params)
            runner = BenchmarkRunner(bench_config)
            
            print(f"Running: {params}")
            results = runner.run_benchmark(self.config.benchmark)
            
            pass_rate = sum(r.passed for r in results) / len(results)
            print(f"  Pass rate: {pass_rate:.1%}")
```

#### Example Ablation Config (`experiments/configs/sample_scaling.yaml`)

```yaml
name: sample_scaling
benchmark: humaneval
model_id: qwen2.5-coder-7b

vary:
  samples: [1, 2, 5, 10, 20, 50]

fixed:
  retrieval: false
  max_debug_iterations: 3
  timeout: 60
```

#### Results Storage (`bench/results.py`)

```python
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from mcode.bench.runner import BenchResult, BenchConfig

class ResultsDB:
    def __init__(self, db_path: str = "experiments/results/results.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
    
    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                benchmark TEXT,
                model_id TEXT,
                config JSON
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY,
                run_id INTEGER,
                task_id TEXT,
                passed BOOLEAN,
                samples_generated INTEGER,
                iterations_used INTEGER,
                time_ms INTEGER,
                error TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
        """)
        self.conn.commit()
    
    def start_run(self, benchmark: str, config: BenchConfig) -> int:
        cursor = self.conn.execute(
            "INSERT INTO runs (timestamp, benchmark, model_id, config) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), benchmark, config.model_id, json.dumps(config.__dict__))
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def save_result(self, run_id: int, result: BenchResult):
        self.conn.execute(
            """INSERT INTO results 
               (run_id, task_id, passed, samples_generated, iterations_used, time_ms, error)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, result.task_id, result.passed, result.samples_generated,
             result.iterations_used, result.time_ms, result.error)
        )
        self.conn.commit()
    
    def get_pass_rates(self, benchmark: str = None, model_id: str = None) -> list[dict]:
        query = """
            SELECT r.model_id, r.config, 
                   COUNT(*) as total,
                   SUM(res.passed) as passed
            FROM runs r
            JOIN results res ON r.id = res.run_id
            WHERE 1=1
        """
        params = []
        if benchmark:
            query += " AND r.benchmark = ?"
            params.append(benchmark)
        if model_id:
            query += " AND r.model_id = ?"
            params.append(model_id)
        
        query += " GROUP BY r.id"
        
        rows = self.conn.execute(query, params).fetchall()
        return [
            {"model_id": r[0], "config": json.loads(r[1]), 
             "pass_rate": r[3] / r[2] if r[2] > 0 else 0}
            for r in rows
        ]
```

### 4.5 pyproject.toml

```toml
[project]
name = "mcode"
version = "0.1.0"
description = "Small model agentic coding research"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.9.0",
    "mellea>=0.2.0",
    "tree-sitter>=0.21.0",
    "tree-sitter-python>=0.21.0",
    "sentence-transformers>=2.2.0",
    "faiss-cpu>=1.7.4",
    "docker>=7.0.0",
    "pydantic>=2.0.0",
    "jinja2>=3.1.0",
    "rich>=13.0.0",
    "pyyaml>=6.0.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.scripts]
mcode = "mcode.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## Part 5: Implementation Roadmap

### Phase 1: Benchmarking Harness (Weeks 1-2)

- [ ] Project scaffolding with uv
- [ ] HumanEval task loader and evaluator
- [ ] MBPP task loader and evaluator  
- [ ] Basic Docker sandbox (run code, capture output)
- [ ] Mellea session wrapper with configurable sampling
- [ ] SQLite results storage
- [ ] CLI: `mcode bench humaneval --model X --samples N`
- [ ] End-to-end test: run HumanEval on one model, get pass rate

### Phase 2: Ablation Framework (Weeks 3-4)

- [ ] YAML config for ablation experiments
- [ ] Ablation runner (iterate over param combinations)
- [ ] Tree-sitter repo map (for retrieval ablation)
- [ ] Basic retrieval (BM25 or dense, not both yet)
- [ ] SWE-Bench Lite integration
- [ ] CLI: `mcode ablate --config X`
- [ ] Analysis notebook template

### Phase 3: Core Ablations (Weeks 5-6)

Run experiments, collect data:

- [ ] Sample scaling: 1, 2, 5, 10, 20, 50 samples
- [ ] Model comparison: Granite 3B vs 8B vs Qwen 7B vs CodeLlama 7B
- [ ] Retrieval: off vs repo-map-only vs full RAG
- [ ] Debug iterations: 0, 1, 3, 5
- [ ] Mellea validation: on vs off (just use test results)

### Phase 4: Analysis & Tool (Weeks 7-8)

- [ ] Statistical analysis of ablation results
- [ ] Identify which techniques matter (and which don't)
- [ ] Build user-facing agent using validated techniques only
- [ ] CLI: `mcode run "task description"`
- [ ] Documentation and writeup

### Stretch Goals

- [ ] Hybrid retrieval (BM25 + dense + reranking)
- [ ] Multi-file editing with diff generation
- [ ] Plan/Act mode separation
- [ ] Full SWE-Bench (not just Lite)
- [ ] CI pipeline that runs benchmarks on PRs

---

## Part 6: Research Questions

### Primary

1. **Sample scaling curve for small models.** Where do diminishing returns kick in? Is it 5? 10? 50?

2. **Minimum viable model size.** Can Granite 3B work, or is 7B the floor?

3. **Retrieval ROI.** How much does full RAG help vs just a repo map? Is it worth the complexity?

4. **Debug iteration value.** How many iterations before gains plateau?

### Secondary

5. **Mellea validation utility.** Does LLM-as-judge add anything over just re-running tests?

6. **Model-specific behavior.** Do different small models respond differently to these techniques?

7. **Task difficulty interaction.** Do techniques help more on hard tasks or easy tasks?

---

## Part 7: Success Criteria

**Minimum:** Working harness that can run HumanEval and MBPP on pluggable models with configurable sample count. Clear data on sample scaling curve.

**Target:** Ablation data across multiple techniques and models. Clear recommendations on what works. User-facing tool that implements validated techniques.

**Stretch:** SWE-Bench results competitive with published baselines. Publishable findings on small model agentic coding.

---

## References

- Brown et al. (2024). "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling"
- Chen et al. (2024). "Teaching Large Language Models to Self-Debug" (ICLR 2024)
- Islam et al. (2024). "MapCoder: Multi-Agent Code Generation for Competitive Problem Solving"
- Shrivastava et al. (2024). "Repoformer: Selective Retrieval for Repository-Level Code Completion"
- Mellea Documentation: https://docs.mellea.ai
- SWE-Bench: https://www.swebench.com
