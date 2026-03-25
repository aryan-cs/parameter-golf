# Scripts Index

This directory mixes stable launchers, current record-push helpers, and exploratory one-offs. The list below points to the scripts that matter right now.

## Start Here

- [`icrn_h200_record_push.sh`](icrn_h200_record_push.sh): orchestrates the current H200-first search flow.
- [`record_push_status.py`](record_push_status.py): ranks completed candidates and prints the exact `8xH100` handoff commands.
- [`record_push_candidate_lib.sh`](record_push_candidate_lib.sh): shared candidate naming, env application, and log-path helpers.

## H200 Artifact-Only Search

- [`icrn_h200_artifact_ttt_portfolio.sh`](icrn_h200_artifact_ttt_portfolio.sh): prints the ordered artifact-only legal-TTT sweep.
- [`icrn_h200_artifact_ttt_candidate.sh`](icrn_h200_artifact_ttt_candidate.sh): runs one artifact-only TTT candidate.
- [`icrn_h200_artifact_ngram_portfolio.sh`](icrn_h200_artifact_ngram_portfolio.sh): artifact-only n-gram evaluation sweep.
- [`icrn_h200_artifact_ttt_ngram_portfolio.sh`](icrn_h200_artifact_ttt_ngram_portfolio.sh): hybrid TTT + n-gram artifact sweep.

## H200 Proxy Training

- [`icrn_h200_ttt_h100_proxy.sh`](icrn_h200_ttt_h100_proxy.sh): baseline H100-step proxy launcher on H200.
- [`icrn_h200_ttt_h100_proxy_candidate.sh`](icrn_h200_ttt_h100_proxy_candidate.sh): runs one promoted architecture/TTT combo.
- [`icrn_h200_ttt_recordstack.sh`](icrn_h200_ttt_recordstack.sh): recovered March 23 stack reproduction.
- [`icrn_h200_resume_legal_ttt.sh`](icrn_h200_resume_legal_ttt.sh): resumed legal-TTT eval over a saved artifact.

## 8xH100 Handoff

- [`h100_record_push_candidate.sh`](h100_record_push_candidate.sh): exact promoted-candidate run for one seed.
- [`h100_record_push_candidate_3seed.sh`](h100_record_push_candidate_3seed.sh): exact three-seed wrapper once the first seed clears.
- [`h100_parallel_candidate_portfolio.sh`](h100_parallel_candidate_portfolio.sh): spread several high-signal candidates across multiple nodes.

## Packaging And Analysis

- [`prepare_submission_metadata.py`](prepare_submission_metadata.py): parse logs and build submission-style metadata.
- [`summarize_record_runs.py`](summarize_record_runs.py): summarize one or more run logs, optionally merged.
- [`salvage_legal_ttt_eval.py`](salvage_legal_ttt_eval.py): re-score a saved artifact with legal TTT settings.
- [`eval_ngram_cache_artifact.py`](eval_ngram_cache_artifact.py): artifact-only n-gram/cache evaluation.
- [`eval_ngram_ttt_artifact.py`](eval_ngram_ttt_artifact.py): artifact-only hybrid n-gram + TTT evaluation.

## Infrastructure

- [`icrn_h200_setup.sh`](icrn_h200_setup.sh): H200 setup/bootstrap helper.
- [`queue_h200_credit_prep.sh`](queue_h200_credit_prep.sh): prepare follow-up jobs before a current run finishes.
- [`queue_h200_followups_after_current.sh`](queue_h200_followups_after_current.sh): queue H200 follow-ups.

## Naming Notes

- `artifact` scripts work from an already-trained artifact and change only evaluation-time behavior.
- `proxy` scripts train on H200 in a way intended to mimic the `8xH100` step budget.
- `record_push` scripts are the current submission path.
- Older one-off launchers remain here because they are still useful references, but they are not the default workflow unless the current lane fails.
