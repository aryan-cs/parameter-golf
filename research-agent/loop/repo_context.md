Current layout for this fresh restart:
- automation, controller state, prompts, and launch scripts live under `research-agent/`
- live experiment code, helper scripts, cached references, workbench snapshots, and the append-only journal live under `research-experiments/`
- mirrored external public repos now live under `research-experiments/cache/external/`

This repo's current worktree is loop-harness-first. The older training/search files still exist in git history at `HEAD`, but many are locally deleted right now. Do not blindly restore or overwrite those deletions.

Mirrored public leaderboard repos available locally:
- first-place starting point: `research-experiments/cache/external/thwu1-parameter-golf/` at `45bbccff356439d2f0b0dbae06cc3fa58b9576ed`
- third-place reference: `research-experiments/cache/external/aruniyer-parameter-golf/` at `954a158102ec64c292ad82b2442e387e505a9388`
- refresh command when needed: `bash research-experiments/scripts/sync_external_public_repos.sh`

Most recent meaningful repo-local frontier from `HEAD:DRAFT3.md` and `HEAD:STRATEGY.md`:
- best completed local exact run: `bytelevel24k_d640_gqa_softcap_cd05_s3200`
- score: `final_val_bpb=1.6017072436714903`
- artifact: `26,942,603` bytes
- status: sample-data local proxy only, far above the `16,000,000` byte cap, not record-track valid and must not be treated as real leaderboard progress

Old prepared follow-ups in git history:
- accumulation queue: `bytelevel24k_d640_gqa_softcap_cd05_b32k_s4800`
- width fallback: `bytelevel24k_d576_gqa_softcap_cd05_s4800`
- knob queue: `bytelevel24k_d640_gqa_softcap_cd08_w100_s3200`
- knob queue: `bytelevel24k_d640_gqa_qg175_cd05_s1600`
- knob queue: `bytelevel24k_d640_gqa_softcap_nosmear_cd05_s1600`

Capabilities already present in `HEAD:train_gpt.py`:
- grouped-query attention
- softcapped logits
- smear block
- delayed int8 QAT
- tied embeddings
- checkpoint save/resume

Missing relative to the current March 22, 2026 record frontier:
- no BigramHash path
- no mixed int5/int6 artifact pipeline
- no MLP3x / wider top-record block family
- no sliding-window evaluation path
- no official `records/track_10min_16mb/...` record packaging

Record-only implication:
- do not spend turns extending the old sample-data `d640` queue unless the work also closes the missing top-record features above
- prefer rehydrating or adapting actual top-record script ideas over continuing the generic local baseline lineage
- use the mirrored first-place public repo as the default base when preparing new record-track experiments
