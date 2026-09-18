# Portable test note

The frozen paired experiment imports the shared game adapter as `run`; its
logic is unchanged. To run the deterministic branch checks from the full
bundle, put the public `until-win` core on `PYTHONPATH`:

```sh
cd /path/to/extracted/cycle-aware-v1
PYTHONPATH=/path/to/jev-nethack/until-win python \
  -W error::ResourceWarning -m unittest -v test_branch_experiment
```

The branch test is deterministic and read-only. It requires the public
`until-win` Python core; it does not launch NetHack, call Jev, or mutate the
live runtime. The full branch artifacts and frozen receipts remain the source
of truth for the already completed 64+64 run.
