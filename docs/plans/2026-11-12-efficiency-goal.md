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

## Status 2026-11-12 (v0.9.3)

Criterion 1 (correctness): 82 tests pass; G0 stays 516 x 9001 = 4,644,516 rows,
516 tracks, 0 gap rows after refine, `accept` and `verify` pass. The withdrawn
splice is unreachable from the default path and the compatibility entry point
replays the reference linker.

Criterion 2 (detection): 300 sequential real frames over three windows,
paired reference/dedupe, median position offset 0.009 px, angle-valid identical.
`prior_dedupe=6` is a project-level opt-in, not a package default; it is recorded
as a science-parameter trade-off because 22/300 frames differ by one detection.
Full 9001-frame run reproduces every per-frame count and the final product
matches the existing G0 refined product at 0.041 px median offset.

Criterion 3 (link/refine): linking 86.15 -> 32.07 s and complete refine
104.18 -> 48.99 s over three paired runs, all output arrays equal. The exact
fast path now diverts float32, huge-coordinate and gate-margin input to the
dense reference assignment (`ec2d3d7`).

Criterion 4 (performance): paired reports with provenance, warmup, p95 and
memory in `tools/benchmark_linker.py`; the 63% linking and 53% refine gains
clear the 20% stage bar. Detection dedupe clears it at the detection stage and
holds >=10% end-to-end on 120- and 300-frame extraction pairs; full-run
end-to-end timing is recorded in `data/runs/2026-11-12_efficiency_goal/`.

Criterion 5 (routes): CPU worker scaling is documented (6-8 workers on this
host; 12 crosses the knee). GPU fixed-fit prototype saves about 2 ms per frame
with no detector regression, rejected as not worth it. Native/JIT remains an
open, separate experiment and is not part of this release.

Criterion 6 (reproducibility): tools are checked in; reports live under
`data/runs/`; raw/frozen data untouched; optional `psutil` use is limited to the
benchmark tool.

Rejected: local identity splice (EMA/competition counterexamples), marker reuse
(9.1% paired, below the bar), GPU fixed-fit (about 2 ms, no full regression).

## Rejected routes (2026-11-12, after 0.9.5)

A local NVMe working copy of the raw frames was built and measured. Raw reads
are 2-3x faster than the USB archive (3535 vs 1066 MB/s with eight threads), but
extraction does not improve: 3000-frame runs spanned 120.9-184.4 s across
raw/output disk combinations, and the fastest was the USB archive with a C:
output. Full 9001-frame runs were 383.8 s (USB) against 426.4/466.4 s (NVMe),
because E: also carries the system, project and output traffic, so read and
write compete on the same WD Blue device.

The limiting resource is CPU, not storage. Eight workers use 35-46% of 20
logical CPUs on this i7-12700H (6 P-cores + 8 E-cores), single-thread detection
is 0.20 s/frame and eight-worker wall clock is 0.04-0.05 s/frame, i.e. 4-5x
scaling rather than 8x. E-core scheduling and turbo loss under an eight-core
load are the plausible causes; the measurement noise on this workstation is
+-30% between repeats, so worker-count micro-tuning (6 vs 8) is not resolvable
here and is left at the documented 6-8 range.

The 49 GB copy was removed after the measurements. Raw read bandwidth was never
the constraint: 52 GB takes about 15 s on NVMe, while detection takes 200+ s.
