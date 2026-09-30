# The experiment layer for a curvature-based cognitive field

The Phase 0–6 implementation of the *curvature cognitive field* design, plus ten method validations.

[中文](README.md) · Design declaration in [`DECLARATION.md`](DECLARATION.md)

**It was split out of `rl-scaffold` into its own repository.** Not because "code shouldn't live
next to knowledge", but because **an experiment's version should not ride on a product's version
number**. It used to live in `rl-scaffold/field/`, so that repository cut a *product* release
(v0.0.5) for this experiment — which was wrong.

**It reads the corpus's `nodes/` and does not write a single byte.** Pinned by the snapshot
comparison in `phase5.py` and by check `B-D5` in `checks.py`.

---

## How it gets its corpus

This repository **does not contain** `nodes/`. It reads an `rl-scaffold` checkout (or any
isomorphic Scaffold corpus). The location is decided by `corpus.py`, in this order:

```
1. the FIELD_CORPUS environment variable
2. default: ../rl-scaffold, side by side with this repository
```

```bash
# default layout (the two repositories side by side) -> nothing to set
python checks.py

# somewhere else
FIELD_CORPUS=/path/to/rl-scaffold python checks.py
```

⚠️ **It never silently passes when the corpus is missing.** The six checks that need a corpus
(B-D1 / B-D2 / B-D7 / B-D11 / B-D14 / B-D15) are marked **skipped** and counted separately —
because **skipped is not passed; they verified nothing.**

### Half of this repository never touches the corpus at all

| Half | Modules | Needs a corpus? |
|---|---|---|
| **Mechanism** | `spectral` `geometry` `connectivity` `curvature` `curvature_split` `ollivier` `field` `ppr` `metrics` `conformal` `knockout` `synthetic` `direction` `directed` `attack` `invariants` | **No** — usable standalone against any graph |
| **Experiment** | `graph` `checks` and `phase*.py` | Yes (though `graph`'s pure graph utilities still do not) |

⚠️ That "the mechanism half never touches the corpus" claim **was killed once by its own test**:
`graph.py` used to attach the corpus at **module import time**, so `curvature.py` — which only
wants `degrees` — was dragged into needing a corpus too. It only became true after the attach
was made **lazy**. See the note in `graph.py:load_nodes`.

---

## What it is not

- **Not a product — an experiment log.** It reads the corpus's `nodes/` and **writes not a
  single byte** — it changes no validation rule, no retrieval, no view.
- **The conclusions are only answerable for this one graph** (36 nodes / 112 situations /
  47 inter-node relations). Swap the graph and every number below is void.
- **Synthetic graphs are for mechanism validation only**, never for reporting performance.
- **This repository cannot be run in full**: without a corpus, six checks are
  **explicitly skipped**. **Skipped is not passed** — they verified nothing.
- **`B-D7` is red, and this repository does not fix it.** Its target is upstream data
  (8 of 17 `HELD_OUT` entries are cue text); the fix needs an upstream decision.
  This repository only reports — it **never loosens a check and never deletes data**.

---

## Results (in one breath)

**Of four kill tests, two pass (spectrum, control), one passes on weak evidence (decidability),
and one does not hold (half of forgetting). Two supporting experiments failed
(the lesion test, and "determinate → multi-valued"). The curvature line was reclassified
from "falsified" to "not falsified, but resting on a single edge".**

| Claim | Result | Key number |
|---|---|---|
| A metric exists (spectrum does not degenerate) | **Holds** | effective dimension 19.1 / 35, λ_max/λ₁ = 85.9 |
| Intent can control the construction | **Holds** (arc length t ≥ 2) | Mantel p = 0.0005 (12/12 settings) |
| Construction is deterministic | **Holds** | element-wise identical across runs |
| Entrance robustness | **Holds** | tolerates ~2 hops (diameter 8); rank degrades smoothly |
| Decidability | **Holds**, but "better than the plain score" is **refuted** | R1 alone AUC 1.000 / p = 0.0028; **paired test R1 vs strong control p = 0.7852 (not significant)** |
| C5 lesion experiment | **Failed** | the effect was its own confound |
| Forgetting (determinate → multi-valued) | **Does not hold** | relaxation makes constructions *more* alike: distinct readouts 2.67 → 1.11 |
| C7 usability | **Half holds** | parameter-free criterion recall 9/9 vs baseline 7/9 |
| Forman curvature = bridge detection | **Refuted** | degree term carries 1.054 of the variance — it is a degree proxy |
| Ollivier-Ricci = interface detection | **Not falsified, on one edge** | the single interface bridge κ = −0.500 |
| Interface bridges (synthetic, n = 10) | **Holds** | interface vs pendant AUC = 1.000; all four graphs separate perfectly |

---

## How to run

```
python checks.py              # 20 B-D falsification checks (19 product + 1 data; 6 skipped without a corpus)
python phase1.py              # graph / spectrum / effective resistance / Forman / Fiedler cut / geometry
python curvature_compare.py   # Ollivier-Ricci vs Forman (with three hand-computed self-tests)
python phase2c.py             # can intent control the construction (Mantel permutation)
python phase3.py              # entrance robustness + matcher ordering diagnosis
python phase4.py              # decidability (go / no-go)
python phase5.py              # forgetting
python phase6.py              # usability + ablation

# probes brought in by method validations (5)-(10)
python phase2d.py             # five readouts compared head to head (verdict on (1))
python phase4b.py             # sweep conductance as a decision statistic + paired test (2)(3)
python phase4c.py             # spectral mode knockout (5)
python phase7a.py             # direction consistency + majority immunity (7)
python phase7b.py             # synthetic-graph interface validation (6; needs no corpus)
python phase7c.py             # directed spectrum and directed OR (8)
python phase7d.py             # attack typology + random-graph invariants (9)(10)
```

The only dependency is **PyYAML** (to reuse the upstream front-matter parser rather than rewrite
it). Everything is deterministic and runs in seconds.

(`phase2.py` and `phase2b_readout.py` are process records: the former is the **voided** version of
a test, the latter the probe that diagnosed why it was void. They are kept for auditability —
do not read them as conclusions.)

---

## Files

**Code sits flat at the repository root**, matching `arena` and
`nested-traceable-discussion-graph`. **All reports and verdicts live in `docs/`**
(31 files: 19 reports + 6 verdicts + 6 baselines), so the root holds only code and facade
documents.

### Facade

| File | What it is |
|---|---|
| [`DECLARATION.md`](DECLARATION.md) | **The design declaration**: the problem, the structure, the four rules of discipline, the results, the three self-corrections, the per-method verdicts |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to contribute (**read this first**) — including "what is not allowed" and what to do when the corpus ref moves |
| [`CHANGELOG.md`](CHANGELOG.md) | Change log |
| [`NOTICE`](NOTICE) | Relationship to `rl-scaffold`, extraction method, third parties |
| `.github/workflows/ci.yml` | Two jobs: **without a corpus** (a gate) and **with a corpus** (product checks gate; the upstream data check only reports) |

### Mechanism modules (corpus-free)

| File | What it does |
|---|---|
| `spectral.py` | Laplacian, Jacobi eigendecomposition, effective resistance, spectral description |
| `geometry.py` | Diffusion coordinates `Ψ_k = u_k/√λ_k`, Fiedler cut, cut conductance |
| `connectivity.py` | Bridges, 2-edge-connected components (Tarjan lowlink) |
| `curvature.py` / `curvature_split.py` | Forman curvature and its term decomposition (used to refute Forman) |
| `ollivier.py` | **Ollivier-Ricci** (exact W₁ via integer-scaled min-cost flow) + three hand-computed self-tests |
| `field.py` | Field evolution `K(t) = e^{−tL}δ_e + α(I − e^{−tL})L⁺S`, readout, distances |
| `ppr.py` | **PPR + sweep cut**: local mass distribution, per-prefix conductance, calibration against the graph's own node-seed null |
| `metrics.py` | Probability of superiority (AUC), **exact paired permutation test**, Holm correction, rank/recall |
| `conformal.py` | Quantile calibration replacing a hand-written line + coverage mechanism check |
| `knockout.py` | **Spectral mode knockout** (the graph never moves; replaces the node-deletion lesion test) |
| `synthetic.py` | Synthetic graph generators (clique rings, pendants, random trees) + grouping by planted edge |
| `direction.py` | **Direction consistency**: mean / first principal component / max-min + immunity test |
| `directed.py` | Directed adjacency, stationary distribution, strong connectivity, topological depth, reachability, **directed OR** |
| `attack.py` | ASPIC+/ABA three-way attack typology + **edge reification** |
| `invariants.py` | **Invariants tested on random graphs** (six, with counterexamples reported) |

### Experiment modules (need a corpus)

`graph.py` builds the graph from `nodes/` (taking only id/type/relations/cues — **no body text**),
`corpus.py` locates the corpus, `paths.py` decides where reports land,
`checks.py` holds the 20 falsification checks and `phase*.py` are the probes.

---

## Four rules of discipline (all of them earned)

1. **Write the criterion down before looking at results.** Changing it afterwards is not allowed;
   you may only change **the object being tested**, and you must record that as a correction.
2. **Diagnose which layer failed before deciding whether to change anything.**
   **Do not keep tuning until it looks good.**
3. **No exemptions — give it a name.** When a check catches a magic number, the fix is always
   rename / name it / change the code, never an exemption list entry.
4. **Distance between two numbers ≠ a test of whether they differ.**
   Any two statistics computed on the **same sample** must be compared **pairwise**
   (this repository uses an exact sign-flip test), with Holm correction for multiple comparisons.
   This rule is what refuted a conclusion that had already been written down.

⚠️ All four are stated in full, with the cases that produced them, in
[`DECLARATION.md`](DECLARATION.md) §三 and [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Open items

1. **A larger positive/negative set** (50 positives / 20 negatives) to re-run the C5 criterion.
   The current sample is 9 vs 4, and the paired test already showed **no detectable difference at
   that size**.
2. **Conformal prediction needs an exchangeable calibration set.** The mechanism is verified
   (nominal 0.95/0.90/0.80 → measured 0.9640/0.9167/0.8027, and it **refuses to give a cut** when
   the sample is too small), but exchangeability is not attainable on the real data.
   **The obstacle is sample size, not the method** — the same point as item 1.
3. **Direction semantics:** upstream `SPEC.md` **contradicts itself** — its prose rule says
   "A appearing in B's `relations` means A points to B", while its own direction table says
   "argument → stance" with the argument writing into `relations`. The measured data supports the
   **table**. Three options are registered in `rl-scaffold`'s `GAPS.md` §7.2.
4. **`relations` is a directed acyclic graph** (36 strongly connected components, every node its
   own) is worth writing into the upstream docs: it decides which directed tools are usable
   (Chung's Laplacian is not; DAG depth / reachability are).
5. **The max-min solver is a local method.** Only the minimum projection it reached is reported,
   not the gap to the global optimum.
6. **Assert the skipped state in CI.** CI currently only asserts "exit code 0", and the exit code is
   also 0 when six checks are skipped. **"Skipped" should be asserted separately**, otherwise a check
   that quietly becomes "always skipped" would go unnoticed.

⚠️ One item is shared with upstream `rl-scaffold`: check `B-D7` fires — **8 of the 17 `HELD_OUT`
rows are verbatim `cues` of the expected node**. Both fixes (add a verbatim/paraphrase marker, or
rename the metric) must be decided upstream. **This repository only reports it; it does not relax
the check, delete data, or modify the corpus.**

---

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

<!-- Copyright 2026 AIC-123 · SPDX-License-Identifier: Apache-2.0 -->
