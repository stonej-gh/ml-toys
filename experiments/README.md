# Experiments

Each numbered directory is one question about this environment, packaged the
same way: a `README.md` (question, method, expected result), a `run.py` (do
the full thing), and for the graded ones a `grade.py` (a seeded short run
against a locked threshold, collected by pytest). Two are notebook-style
READMEs with inline snippets instead of scripts.

| # | question | graded |
|---|---|---|
| [01-survive](01-survive/README.md) | can a from-scratch DQN learn to rescue a doomed orbit? | yes |
| [02-reward-hacking](02-reward-hacking/README.md) | hand a fresh agent a broken spec: does it cheat? (the flagship) | yes |
| [03-forgetting](03-forgetting/README.md) | curricula climb; do leagues actually retain? | yes |
| [04-fuel-economics](04-fuel-economics/README.md) | what does a hard fuel tank do that a thrust tax cannot? | notebook |
| [05-curved-lasers](05-curved-lasers/README.md) | how much of a policy is an aim model fitted to its era's ballistics? | notebook |
| [06-spotter-port](06-spotter-port/README.md) | can a stranger retrain the spotter and walk the whole deploy ladder? | yes |

## Grading

```
python -m pytest -m grade_cheap -q          # all graders, from the repo root
python -m pytest experiments/01-survive/grade.py -m grade_cheap -q   # just one
```

The full quick test suite stays `pytest tests/ -q`; a bare `pytest` runs both.

Two kinds of determinism, stated per grader:

* **Platform-exact.** Everything evaluated through the pure-Python reference
  forward ([orbitduel/netpilot.py](../orbitduel/netpilot.py)) or the seeded
  renderer produces identical numbers on every OS and Python; those graders
  can assert exact win counts and hashes.
* **Machine-deterministic.** Graders that *train* (02's live run, 06's micro
  retrain) run torch seeded on one CPU thread: two runs on the same machine
  are identical, but across platforms only thresholds hold, so their bars sit
  far below the reference outcomes (the margins are quoted in each README).

Shared plumbing lives in [exputil.py](exputil.py). Thresholds and locked
golden values live in each `grade.py`, deliberately: an experiment should
read whole from its own directory.
