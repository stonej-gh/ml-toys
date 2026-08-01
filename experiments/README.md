# Experiments

Each numbered directory is one question about this environment, packaged the
same way: a `README.md` (question, method, expected result), a `run.py` (do
the full thing), and for the graded ones a `grade.py` (a seeded short run
against a locked threshold, collected by pytest). Two are notebook-style
READMEs with inline snippets instead of scripts.

| # | question | level | graded |
|---|---|---|---|
| [01-survive](01-survive/README.md) | can a from-scratch DQN learn to rescue a doomed orbit? | beginner | yes |
| [02-reward-hacking](02-reward-hacking/README.md) | hand a fresh agent a broken spec: does it cheat? (the flagship) | beginner | yes |
| [03-generalization](03-generalization/README.md) | curricula climb; does a league generalize where a ladder does not? | intermediate | yes |
| [04-fuel-economics](04-fuel-economics/README.md) | what does a hard fuel tank do that a thrust tax cannot? | intermediate | notebook |
| [05-curved-lasers](05-curved-lasers/README.md) | how much of a policy is an aim model fitted to its era's ballistics? | advanced | notebook |
| [06-spotter-port](06-spotter-port/README.md) | can a stranger retrain the spotter and walk the whole deploy ladder? | intermediate | yes |

The level column marks how much vocabulary a question assumes, and each README
opens with a plain-English one-liner. New to the vocabulary itself? Start at
[docs/START-HERE.md](../docs/START-HERE.md) and the
[glossary](../docs/GLOSSARY.md); [02-reward-hacking](02-reward-hacking/README.md)
is the friendliest first run. Anything that trains needs the one-time
install from the [README quickstart](../README.md#quickstart-the-pilot)
first: `python -m venv .venv` then `.venv/bin/pip install -e ".[train,viz,dev]"`.

## Grading

```
python -m pytest -m grade_cheap -q          # all graders, from the repo root
python -m pytest experiments/01-survive/grade.py -m grade_cheap -q   # just one
```

The full quick test suite stays `pytest tests/ -q`; a bare `pytest` runs both.

Two kinds of determinism, stated per grader:

* **Platform-exact.** The pure-Python reference forward
  ([orbitduel/netpilot.py](../orbitduel/netpilot.py)) and the seeded renderer
  produce identical numbers on every OS and Python; hashes and the pinned
  golden-eval outcomes are asserted exactly, and CI proves them on Linux and
  macOS. One honest caveat: a full episode can sit on a knife edge where an
  interpreter's math library flips a single game, so matrix-style graders
  assert margins around a reference measurement, not the exact matrix.
* **Machine-deterministic.** Graders that *train* (02's live run, 06's micro
  retrain) run torch seeded on one CPU thread: two runs on the same machine
  are identical, but across platforms only thresholds hold, so their bars sit
  far below the reference outcomes (the margins are quoted in each README).

Shared plumbing lives in [exputil.py](exputil.py). Thresholds and locked
golden values live in each `grade.py`, deliberately: an experiment should
read whole from its own directory.
