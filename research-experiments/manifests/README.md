# Manifest Staging

Stage real jobs in `pending/`.

- `job_kind=experiment` is for actual train/eval runs that should count toward model progress.
- support jobs like data download or analysis should use a different `job_kind`.
- the controller ingests only manifests whose runtime prerequisites are currently satisfied.
