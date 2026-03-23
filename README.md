# Parameter Golf Working Notes

Prepared on March 22, 2026 from the official `openai/parameter-golf` repository and official Google Colab docs. This is a working brief for our local effort, not the official rules source. If the leaderboard changes after March 22, 2026, this file will be stale.

## What This Competition Is

OpenAI Model Craft Challenge: Parameter Golf is a competition to train the best language model that:

- fits in a `16,000,000` byte artifact,
- trains in under `10 minutes` on `8xH100 SXM`,
- and is scored on FineWeb validation compression using tokenizer-agnostic `bits per byte` (`val_bpb`).

Lower `val_bpb` is better.

The challenge is framed as a parameter-constrained optimization problem: given a tiny artifact budget, how much modeling quality can you squeeze out through architecture, compression, training, and evaluation tricks?

The challenge window in the official repo is `March 18, 2026` through `April 30, 2026`.

## What The Point Of The Competition Is

The point is not just to make a small model. It is to explore the frontier of what becomes optimal when parameters and artifact bytes are the bottleneck.

The official writeup explicitly encourages:

- unusual architectures,
- aggressive quantization and compression,
- tokenizer changes,
- test-time compute,
- long-context evaluation,
- recurrence / parameter tying / low-rank ideas,
- and other "weird" but valid approaches.

This challenge is heavily inspired by NanoGPT speedrunning, but here the pressure is on parameter efficiency and artifact size rather than only wallclock.

## What We Are Given

The official repo gives participants a starter kit:

- `train_gpt.py`: the main CUDA training baseline
- `train_gpt_mlx.py`: a simple Apple Silicon MLX path for local iteration
- `data/cached_challenge_fineweb.py`: downloader for the published challenge dataset/tokenizer export
- `data/README.md`: dataset and retokenization workflow notes
- `records/`: public record and non-record submissions, including code, logs, and README notes
- `requirements.txt`: reference dependencies for self-setup

The starter scripts are described as launch points, not SOTA recipes. The best ideas are expected to live in `records/`, not in the root baseline script.

## Official Objective And Metric

The official score is based on compression performance on the FineWeb validation set:

- metric: `val_bpb` (`bits per byte`)
- lower is better
- evaluation is tokenizer-agnostic
- validation uses the fixed `fineweb_val_*` split
- the official README says the validation split is the fixed first `50k` documents

Important implication: tokenizer tricks are allowed, but you must still compute `val_bpb` correctly.

## Hard Rules And Constraints

### Artifact Limit

- Total artifact cap is decimal `16MB`, meaning `16,000,000` bytes.
- This is not `16 MiB` / `16,777,216` bytes.
- Counted artifact size is `code bytes + compressed model bytes`.
- The official FAQ says all counted code should live in `train_gpt.py`.
- Evaluation cannot rely on network calls, extra downloads, or access to training data.
- The submission must be self-contained and reproducible.

### Compute Limit

- Leaderboard record submissions must train in under `10 minutes` on `8xH100 SXM`.
- Evaluation also has its own `10 minute` limit on `8xH100`; this is in addition to training time.
- Non-record submissions are allowed for interesting approaches that do not beat SOTA.
- There is also an unlimited-compute non-record track.

### Reproducibility And Fairness

- OpenAI is not auto-verifying every result immediately, but top entries may be checked later.
- Non-reproducible submissions can be disqualified.
- The repo explicitly reserves the right to reject runs that violate the spirit of the competition through unfair external compute.
- Hyperparameter tuning is allowed.
- Brute-forcing seeds or sneaking in hidden compute is not.

### Evaluation Restrictions

- Evaluation may use any sequence length as long as it stays within the eval budget.
- You cannot access training data during evaluation unless you "pay for those bits" inside the artifact.
- You cannot train on the validation set before evaluation.
- Test-time training is only allowed on validation tokens that have already been evaluated.

## Record Submission Requirements

To claim a new SOTA record, the official repo says a submission must:

1. Beat the existing SOTA by at least `0.005 nats`.
2. Show enough run logs for `p < 0.01` significance because of run-to-run variance.
3. Reproducibly run in under `10 minutes` on `8xH100`.

If a submission changes the tokenizer or dataset, it must prove the `val_bpb` calculation is correct.

Each submission PR should only add a new folder under the appropriate `records/` track and include:

- `README.md`
- `submission.json`
- training logs
- `train_gpt.py`
- any other needed dependencies

Broken or non-compiling record folders are not accepted.

## Dataset And Tokenizer Notes

The official data workflow notes say:

- `python3 data/cached_challenge_fineweb.py --variant sp1024` downloads the published cached export
- default download includes the full validation split plus `80` training shards, which is `8B` training tokens
- current shard size is `100,000,000` tokens
- `10B` retokenized training tokens corresponds to `100` train shards
- the default published repo is `willdepueoai/parameter-golf`
- the baseline tokenizer family in the starter path uses a `1024`-token vocabulary (`sp1024`)

The main README uses:

- a smaller `--train-shards 10` example for local MLX smoke tests
- the full default `sp1024` published export for remote CUDA work

## Official Starter Workflow

### Local Iteration

The repo offers an Apple Silicon MLX path for quick local iteration. The example smoke run uses:

- `ITERATIONS=200`
- `TRAIN_BATCH_TOKENS=8192`
- `VAL_LOSS_EVERY=0`
- `VAL_BATCH_SIZE=8192`

### Remote CUDA Baseline

The official remote example:

- clones the repo onto a GPU machine,
- downloads the cached FineWeb export,
- and runs `torchrun --standalone --nproc_per_node=1 train_gpt.py` on a single H100.

The official README says the baseline should end around:

- `val_bpb ~ 1.2`
- compressed artifact under `16MB`

OpenAI also mentions:

- `8xH100` leaderboard submissions are expensive,
- cheaper GPU SKUs are recommended for iteration first,
- and OpenAI is sponsoring `$1,000,000` in compute credits to help participants get started.

There is also an optional challenge participant form for attribution and recruiting visibility.

## Current Leaderboard Snapshot

Snapshot taken from the official repo on March 22, 2026. The latest leaderboard rows visible there are dated `March 20, 2026`.

| Rank | Run | Score | Author | Summary | Date |
|---|---|---:|---|---|---|
| 1 | 10L Int5-MLP + BigramHash(10240) | 1.1428 | thwu1 | 10 layers, mixed int5/int6 quantization, BigramHash(10240), SWA(0.4), WD=0.04 | 2026-03-20 |
| 2 | Int6 MLP3x + SmearGate + BigramHash | 1.1458 | Raahil Shah | 3x MLP, SmearGate, BigramHash, OrthoInit, Muon WD, SWA | 2026-03-20 |
| 3 | 11L MLP3x + Int6 QAT | 1.1502 | aruniyer | 11 layers, 3x MLP, int6 QAT, zstd-22, WD=0.04, sliding eval | 2026-03-20 |
| 4 | SmearGate + OrthoInit + Muon WD | 1.1556 | aquariouseworkman | SmearGate, BigramHash, 3x MLP, int6 STE QAT, sliding eval | 2026-03-19 |
| 5 | 10L Int6 QAT + Zstd MLP2.6x | 1.1586 | yahya010 | 10 layers, int6 QAT, zstd-22, MLP 1344, Muon 0.99, sliding eval | 2026-03-19 |
| 6 | Mixed Quant + Sliding Window Eval | 1.1630 | aquariouseworkman | int6 block weights, int8 embeddings, 3x MLP, sliding eval | 2026-03-19 |
| 7 | Muon WD + 10 layer | 1.1748 | notapplica | spectral embed init, residual mix, previous wins | 2026-03-19 |
| 8 | Sliding Window Eval | 1.1925 | Matthew Li | sliding-window evaluation at stride 64, increasing eval context | 2026-03-19 |
| 9 | Lora TTT | 1.1928 | samacqua | test-time training with LoRA adapters | 2026-03-19 |
| 10 | 4k seq length | 1.2014 | Spokane Way | 4k context length and better hyperparameters | 2026-03-19 |
| 11 | 2048 seq length | 1.2060 | Spokane Way | 2048 sequence length in train and val | 2026-03-18 |
| 12 | int6 mixed precision | 1.2147 | Nan Liu | 10 layers, mixed int8/int6 | 2026-03-18 |
| 13 | fp16 Embed | 1.2197 | Renier Velazco | fp16 tied embedding plus LR / warmdown tuning | 2026-03-18 |
| 14 | Naive Baseline | 1.2244 | Baseline | 9 layers, 512 dim, 1024 vocab, tied embeddings, 4 KV heads | 2026-03-18 |

### Notable Non-Record Run

| Run | Score | Author | Summary | Date |
|---|---:|---|---|---|
| 4-Hour Baseline | 1.2074 | Will DePue | unlimited-compute test, 4 hours on 8xH100 | 2026-03-18 |

## What The Leaderboard Currently Looks Like

A few strong patterns are already obvious:

- The board moved very quickly between March 18 and March 20, 2026.
- The baseline `1.2244` was improved to `1.1428` in about two days, a gain of roughly `0.0816 bpb`.
- The top of the board is now crowded with heavily quantized, quantization-aware, wider-than-baseline models.
- The current frontier is not just "train longer". The 4-hour non-record baseline still loses badly to the top 10-minute records.
- A large amount of early gain came from evaluation tricks, especially sliding-window evaluation.
- The current top runs then stack better quantization, more capacity, and better token-pair features on top of those eval gains.

## Strategy Landscape

### 1. Sliding-Window Evaluation Was The First Big Free Win

The `Sliding Window Eval` record showed that evaluation alone could move the score from `1.2244` to `1.1925`, roughly a `0.032` improvement, by scoring tokens with much richer left context.

This matters because:

- it costs no artifact bytes beyond code,
- it is a pure eval-side improvement,
- and it appears in many later records.

Current takeaway: sliding eval is table stakes.

### 2. Longer Context Helped Early, But It Is Not The Whole Story

Early records improved by increasing training context:

- `2048` sequence length got to about `1.206`
- `4096` sequence length plus tuning got to about `1.2014`

But later winners often moved back toward `1024` or `2048` train length while using better throughput, better quantization, and better evaluation. That suggests raw context length alone is not the main frontier now.

Current takeaway: long context helps, but throughput-efficient training plus better eval may dominate.

### 3. Quantization Is Central, Not Optional

The strongest runs are all aggressively compression-aware:

- int6 per-row quantization on block weights
- int8 or fp16 treatment for more sensitive tensors like embeddings
- QAT / STE fake-quant during training
- zstd-22 instead of zlib in many top entries
- one current leader uses int5 for MLP weights and int6 for attention weights

Why it matters:

- the artifact budget is the game,
- compression savings buy extra depth and width,
- and some training choices are explicitly made to make weights quantize better later.

Current takeaway: we should assume artifact-aware training is a first-class objective.

### 4. Capacity Is Being Spent On MLP Width And Extra Depth

Repeated winning pattern:

- keep model dimension at `512`
- widen MLP from `2x` to `3x`
- add depth from `9` layers to `10` or `11` layers
- fund the extra capacity through stronger quantization and compression

The current #1 run explicitly says int5 MLP compression saved enough bytes to buy a 10th layer.

Current takeaway: capacity increases are working when quantization pays for them.

### 5. Bigram-Level Features Are Showing Up Repeatedly

Several top entries add token-pair structure near the embedding layer:

- `SmearGate`
- `BigramHash`

The #1 and #2 runs both use bigram-style features, and the #1 run uses `BigramHash(10240)`.

This is a strong signal that tiny explicit token-pair inductive bias is useful under tight parameter budgets.

### 6. Optimizer And Regularization Details Matter A Lot

Frequently repeated knobs in the top runs:

- Muon optimizer
- higher momentum (`0.99`) with warmup
- decoupled weight decay, especially around `0.04`
- longer warmdown schedules
- lower learning rates than the baseline
- SWA over the late-training checkpoints
- orthogonal initialization / muP-style scaling

These are not minor polish anymore. In the top record notes, weight decay and SWA are treated as direct contributors to better post-quantization quality.

### 7. Mixed Precision For Sensitive Tensors Is A Major Theme

One particularly important pattern:

- embeddings are often left as `fp16` or moved to `int8`
- attention / MLP block weights are quantized more aggressively
- some late-layer projections are also protected in `fp16`

This seems to be a recurring answer to the fact that embeddings are more quantization-sensitive than many block weights.

### 8. Test-Time Training Exists, But It Is Not The Current SOTA

`LoRA TTT` showed that test-time training can help and is valid when done without label leakage. But the later leaderboard leaders are dominated more by:

- sliding eval,
- quantization-aware training,
- wider MLPs,
- added depth,
- and bigram-feature engineering.

Current takeaway: TTT is interesting, but it is not where the board is currently winning.

### 9. Unlimited Compute Alone Is Not Enough

The non-record `4-Hour Baseline` reached `1.2074`, which is better than the initial baseline but still much worse than the top compressed 10-minute runs.

Current takeaway: better ideas beat just running the baseline longer.

## Concrete Notes From The Top Public Runs

### Current #1: 10L Int5-MLP + BigramHash(10240)

Public notes from the March 20, 2026 leader:

- mean `val_bpb = 1.14276` across `3` seeds
- `10` layers, `512` model dim, `8` heads, `4` KV heads
- `3x` MLP expansion with `relu^2`
- mixed quantization:
  - `int5` for MLP weights
  - `int6` for attention weights
  - `fp16` for tied embeddings and last-layer key projections
- `BigramHash(10240)` with `dim=128`
- `SmearGate`
- orthogonal init, muP-style output scaling, U-Net skip connections
- Muon with `WD=0.04`
- `seq_len=2048`
- `TRAIN_BATCH_TOKENS=786,432`
- SWA over the last `40%` of warmdown
- sliding eval with `stride=64`

Important public ablation note: the record README says moving from a 9-layer int6 base to `int5 MLP + 10th layer` was one of the biggest gains, and increasing BigramHash from `8192` to `10240` still helped.

### Current #2: Int6 MLP3x + SmearGate + BigramHash

Public notes from the March 20, 2026 second-place run:

- mean `val_bpb = 1.1458` across `3` seeds
- `9` layers, `512` dim
- `3x` MLP expansion
- int6 per-row quantization on block weights
- `fp16` tied embeddings
- zstd level `22`
- `SmearGate`
- `BigramHash(4096)`
- orthogonal init
- Muon + decoupled `WD=0.04`
- SWA every `50` steps over the last `50%` of training
- `seq_len=2048`
- artifact size about `15.86MB`

Important public note: this README says the `3x` MLP expansion was the single largest contributor.

### Current #3: 11L MLP3x + Int6 QAT

Public notes from the March 20, 2026 third-place run:

- mean `val_bpb = 1.1502` across `3` seeds
- `11` layers
- `3x` MLP expansion
- int6 quantization-aware training with STE
- all block weights int6, tied embedding exported in `fp16`
- zstd level `22`
- sliding eval with `stride=64`
- Muon / Adam weight decay at `0.04`
- model size about `26.5M` params
- compressed artifact about `15.4MB`

Important public note: this run shows that even `11` layers can fit if compression is strong enough.

## Practical Conclusions For Our Own Attempt

If we want to be competitive, the public record notes suggest these are the most promising first areas:

- reproduce the baseline cleanly
- add sliding-window evaluation
- make the export quantization-aware from the start
- try mixed quantization for embeddings vs block weights
- spend saved bytes on MLP width and then depth
- explore bigram features like SmearGate or BigramHash
- tune Muon momentum, weight decay, warmdown, and SWA with quantization in mind

The public leaderboard also suggests we should not assume that:

- longer training alone will carry us,
- a pure baseline reproduction is enough,
- or that context length alone is the main remaining lever

## Colab + VS Code Working Plan For This Repo

We plan to use Google Colab as an accessible GPU-backed experimentation environment while keeping the project code in this repo and editing from VS Code.

### What The Official Colab Docs Say

From the official Colab docs:

- notebooks can be stored in Google Drive or loaded from GitHub
- code runs inside a VM private to the account
- that VM has a maximum lifetime and can be deleted after idle time
- free Colab notebooks generally run for at most `12 hours`, with dynamic limits and idle timeouts
- Drive I/O can be slow, so it is better to copy active data to local runtime disk when possible
- Colab officially documents hosted runtimes and also Colab UI connections to local/custom runtimes
- as of March 22, 2026, the official FAQ also says free managed runtimes may terminate `remote control such as SSH shells` and workflows that `bypass the notebook UI to interact primarily via a web UI`

### What That Means For Us

For this project, the safest practical workflow is:

- keep source of truth in Git / this repo
- use Colab notebooks for GPU execution and quick experiments
- sync code through GitHub or notebook uploads rather than treating Drive as the primary dataset store
- copy datasets from Drive or remote storage into the Colab VM's local filesystem before training
- treat Colab as a prototyping environment, not a faithful replacement for official `8xH100 SXM` leaderboard verification
- avoid assuming that a direct VS Code attachment to a managed Colab runtime will be stable on the free tier

Inference from the official docs:

- a notebook-first Colab workflow is clearly supported
- a VS Code centered workflow against a managed Colab backend may be fragile unless we use a paid/dedicated setup or switch to a different remote machine we control

Important limitation:

- the official Parameter Golf leaderboard target is `8xH100 SXM in 10 minutes`
- Colab is useful for iteration and ablations, but it is not the official reference environment

### Recommended Project Workflow

1. Keep training code script-first so it can run both from terminal and notebook cells.
2. Use Colab to reproduce a smaller baseline or ablation quickly.
3. Log `val_loss`, `val_bpb`, artifact size, and quantization penalty for every run.
4. Once an idea looks promising, move it to a more faithful multi-GPU environment for serious verification.

## Sources

- Official challenge repo: https://github.com/openai/parameter-golf
- Official challenge README: https://github.com/openai/parameter-golf/blob/main/README.md
- Official records folder: https://github.com/openai/parameter-golf/tree/main/records
- Official data workflow notes: https://github.com/openai/parameter-golf/blob/main/data/README.md
- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
- Google Colab local runtimes: https://research.google.com/colaboratory/local-runtimes.html
