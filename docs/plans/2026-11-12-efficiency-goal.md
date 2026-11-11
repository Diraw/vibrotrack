# Correctness-preserving efficiency Goal

## Scope

Optimize vibrotrack detection, orientation, linking, refine and I/O without reducing recall or scientific correctness. Choose technologies by evidence, not by language preference. Do not change detector thresholds, interpolate positions, modify raw/frozen data or publish remotely. Refine remains the standard workflow; manual curation is optional fallback.

Baseline commit: 1feaa28 (0.9.2). The old G0 refine report records single-run measurements (21.3 s versus 142 s), not repeated guarantees. Its timestamp was manually assigned and background-load assertions were not measured. Preserve this historical record but do not treat these details as verified facts.

## Verifiable acceptance criteria

1. Correctness: existing tests pass plus adversarial tests for partial recovery, ambiguity, tails, source identity inconsistencies and cross-chunk references. G0 remains 516 x 9001 = 4,644,516 rows, 516 tracks and zero gap rows. No interpolation or loosened acceptance gates.
2. Detection/orientation: initially compare six fixed representative frames, then at least 300 sequential real frames covering ordinary/contact/wall/gap/shard-boundary cases. Prefer exact field equality. Any tolerance requires explicit justification plus identical accepted detections, validity and identities; no silently accepted drift.
3. Link/refine: accelerated defaults must reproduce the full reference or conservatively fall back. G0 alone cannot establish general splice equivalence. Compare every chunk array including global predecessors and angles.
4. Performance: warmup plus at least three paired repetitions; record individual timings, median/p95, environment, input identity, config, commit and dirty state. Measure cold startup/JIT, I/O/transfers, peak memory, worker scaling and GPU memory separately. Accept a production optimization at >=20% targeted-stage reduction and >=10% relevant end-to-end reduction, otherwise document rejection or limited applicability. Microbenchmarks are not whole-run evidence.
5. Routes: assess CPU scaling, native/JIT, GPU feasibility and workflow/I/O with profiling or bounded prototypes. Keep only profitable routes; reject a route when evidence demonstrates poor value. Using every technology is not required.
6. Reproducibility: checked-in benchmark tools/tests; reports outside code; untouched raw/frozen data; scoped clean diffs; optional dependencies and fallbacks documented. Final report accounts for every requirement, rejected experiments and remaining limitations.

## Design and alternatives

Prefer profiling and redundant-work removal for portability and exactness. Native/JIT is second for remaining numerical hotspots. GPU is a bounded experiment: RTX 3060 Laptop 6 GiB, approximately 3.2 GiB free at discovery; transfers and cold compilation count.

The first read-only detector profile shows _marker called twice: candidate photometric veto, then the selected subset. Prototype reuse of the first computation without modifying production; compare all arrays and repeated timings. Independently audit incremental splice before accepting its general correctness.

## Execution plan

1. tools/benchmark_detector.py: fixed representative frames, fixed stored prior, warmup, repeated timings, per-array hashes, cProfile and actual timestamps/environment. Explicitly label as microbenchmark rather than sequential regression.
2. Temporary marker-reuse experiment, exact differential tests and paired benchmark. Only adopt after demonstrated gain and test-first production change.
3. Reproduce independent splice findings; add conservative correctness gates/fallback before prioritizing speed.
4. Expand to 300 sequential frames and paired end-to-end profiling; assess worker/GPU/native/I/O routes.
5. Commit validated work, record evidence and continue until every acceptance criterion is met or an external blocker is genuine.
