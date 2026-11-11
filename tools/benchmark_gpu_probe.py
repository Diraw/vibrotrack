"""Isolated fixed-ray-fit CUDA probe; never a full detector or a production backend.

Run with PYTHONDONTWRITEBYTECODE=1. Writes only a new --out JSON (refuses overwrite).
The ~3000-candidate workload tiles one real frame; it is not 3000 independent discs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

# Local process only. Do not install packages or change shared configuration.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
sys.dont_write_bytecode = True


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf8")).hexdigest()


def array_hash(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def command_output(command, cwd=None):
    try:
        return subprocess.check_output(command, cwd=cwd, text=True,
                                       stderr=subprocess.STDOUT, timeout=20).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return {"unavailable": str(exc)}


def memory_snapshot(torch):
    free, total = torch.cuda.mem_get_info()
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "global_free_bytes": free, "global_total_bytes": total,
            "nvidia_smi": command_output([
                "nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu",
                "--format=csv,noheader"])}


def summary(values):
    return {"samples_s": values, "median_s": statistics.median(values),
            "min_s": min(values), "max_s": max(values)}


def numpy_trace(strength, rr, r0, A2, min_strength=20, tol=2.2):
    """Diagnostic copy, checked bit-for-bit against the imported production fit."""
    import numpy as np
    valid = (strength > min_strength) & (np.abs(rr - r0) < tol)
    trace = [valid.copy()]
    for _ in range(3):
        w = valid.astype(np.float64)
        M = np.einsum("nr,ri,rj->nij", w, A2, A2) + np.eye(2)[None] * 1e-4
        b = np.einsum("nr,ri,nr->ni", w, A2, rr - r0)
        c = np.linalg.solve(M, b[..., None])[..., 0]
        resid = rr - (c @ A2.T + r0)
        valid = (strength > min_strength) & (np.abs(resid) < tol)
        trace.append(valid.copy())
    ns = valid.sum(1)
    rmse = np.sqrt((resid ** 2 * valid).sum(1) / np.maximum(ns, 1))
    return (c, ns, rmse), trace


def torch_fit(strength, rr, r0, A2, eye, trace=None,
              min_strength=20, tol=2.2):
    """Three original iterations, float64 fit with NumPy's float32 rr-r0 semantics.

    Input profile rr is float32. NumPy 1.x treats this Python scalar subtraction
    as float32 before its einsum promotes to float64. Prematurely converting rr
    would silently change the problem by about 1e-6 pixels.
    """
    import torch
    delta32 = rr - r0
    delta = delta32.to(dtype=torch.float64)
    rr64 = rr.to(dtype=torch.float64)
    valid = (strength > min_strength) & (delta32.abs() < tol)
    if trace is not None:
        trace.append(valid)
    for _ in range(3):
        w = valid.to(dtype=torch.float64)
        M = torch.einsum("nr,ri,rj->nij", w, A2, A2) + eye
        b = torch.einsum("nr,ri,nr->ni", w, A2, delta)
        c = torch.linalg.solve(M, b.unsqueeze(-1)).squeeze(-1)
        resid = rr64 - (c @ A2.T + r0)
        valid = (strength > min_strength) & (resid.abs() < tol)
        if trace is not None:
            trace.append(valid)
    ns = valid.sum(1)
    rmse = ((resid.square() * valid).sum(1) / ns.clamp_min(1)).sqrt()
    return c, ns, rmse


def gates(result, params):
    import numpy as np
    c, ns, rmse = result
    shift = np.linalg.norm(c, axis=1)
    flags = {"min_rays": ns >= params["min_rays"],
             "rmse": rmse < params["r_tol"],
             "max_shift": shift < params["max_shift"]}
    flags["accepted_by_fixed_fit_only"] = (
        flags["min_rays"] & flags["rmse"] & flags["max_shift"])
    return flags


def compare(reference, candidate, params, xy):
    import numpy as np
    record = {}
    for name, a, b in zip(("c", "ns", "rmse"), reference, candidate):
        delta = np.abs(a - b)
        record[name] = {"bit_equal": bool(np.array_equal(a, b)),
                        "max_abs_error": float(delta.max(initial=0)),
                        "within_atol_1e-9_rtol_1e-10": bool(np.allclose(
                            a, b, atol=1e-9, rtol=1e-10, equal_nan=False)),
                        "reference_sha256": array_hash(a),
                        "candidate_sha256": array_hash(b)}
    a_flags, b_flags = gates(reference, params), gates(candidate, params)
    record["gate_mismatches"] = {
        k: int(np.count_nonzero(a_flags[k] != b_flags[k])) for k in a_flags}
    record["accepted_reference"] = int(a_flags["accepted_by_fixed_fit_only"].sum())
    record["accepted_candidate"] = int(b_flags["accepted_by_fixed_fit_only"].sum())
    record["float32_centres_equal"] = bool(np.array_equal(
        xy + reference[0].astype(np.float32), xy + candidate[0].astype(np.float32)))
    record["float32_rmse_equal"] = bool(np.array_equal(
        reference[2].astype(np.float32), candidate[2].astype(np.float32)))
    record["margins_to_acceptance_threshold"] = {
        "rays_min_abs": float(np.abs(reference[1] - params["min_rays"]).min()),
        "rmse_min_abs": float(np.abs(reference[2] - params["r_tol"]).min()),
        "shift_min_abs": float(np.abs(np.linalg.norm(reference[0], axis=1)
                                         - params["max_shift"]).min())}
    record["numeric_and_gate_match"] = bool(
        all(record[k]["within_atol_1e-9_rtol_1e-10"] for k in ("c", "ns", "rmse"))
        and record["ns"]["bit_equal"] and not any(record["gate_mismatches"].values()))
    return record


def timed_gpu(strength, rr, r0, A2, eye):
    """Wall time with synchronization at each stage, pageable host inputs/outputs."""
    import torch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ds = torch.from_numpy(strength).to("cuda")
    dr = torch.from_numpy(rr).to("cuda")
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    result = torch_fit(ds, dr, r0, A2, eye)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    host = tuple(t.cpu().numpy().copy() for t in result)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    del ds, dr, result
    return host, {"h2d": t1 - t0, "compute": t2 - t1,
                  "d2h": t3 - t2, "total": t3 - t0}


def run(args, report):
    import_start = time.perf_counter()
    import cv2
    import numpy as np
    import scipy
    from vibrotrack import detect, detect2
    report["cold_start"] = {"cpu_dependencies_import_s": time.perf_counter() - import_start}
    import_start = time.perf_counter()
    import torch
    report["cold_start"]["torch_import_s"] = time.perf_counter() - import_start
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    report["environment"] = {
        "platform": platform.platform(), "python": sys.version,
        "executable": sys.executable, "numpy": np.__version__,
        "scipy": scipy.__version__, "opencv": cv2.__version__,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "opencv_threads": cv2.getNumThreads(), "torch_cpu_threads": torch.get_num_threads(),
        "thread_environment": {k: os.environ.get(k) for k in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "driver": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                  "--format=csv,noheader"]),
        "cupy": "not imported or used; task reported ABI/CUDA DLL import failure"}
    report["provenance"]["production_sources"] = {
        str(Path(m.__file__).resolve()): file_hash(m.__file__) for m in (detect, detect2)}
    report["provenance"]["functions_sha256"] = {
        fn.__name__: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
        for fn in (detect2._peaks, detect2._profile, detect2._fit_free, detect2._fit_fixed)}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no GPU timings fabricated")
    report["cold_start"]["cuda_initialized_before_explicit_init"] = torch.cuda.is_initialized()
    report["cold_start"]["nvidia_smi_before_init"] = command_output([
        "nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"])
    t0 = time.perf_counter()
    torch.cuda.init()
    torch.cuda.synchronize()
    report["cold_start"]["cuda_init_and_first_sync_s"] = time.perf_counter() - t0
    prop = torch.cuda.get_device_properties(0)
    cap = args.memory_cap_mib * 1024 ** 2
    torch.cuda.set_per_process_memory_fraction(min(cap / prop.total_memory, 0.15))
    report["environment"].update({"gpu": prop.name, "gpu_total_bytes": prop.total_memory,
                                   "compute_capability": [prop.major, prop.minor],
                                   "allocator_cap_bytes": min(cap, int(prop.total_memory * 0.15)),
                                   "allocator_cap_excludes_context_and_library_allocations": True})
    report["environment_sha256"] = json_hash(report["environment"])
    report["memory_after_init"] = memory_snapshot(torch)
    cfg_path = args.case / "run_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf8"))
    if [float(x) for x in cfg.get("scale_levels", [])] != [128.0, 28224.0]:
        raise ValueError("Unexpected source scale levels; investigate instead of silently changing input")
    params = dict(detect2.DEFAULTS)
    params.update(cfg.get("detector_defaults", {}))
    # Frame zero has no temporal priors; the real workload is exactly its masked peaks.
    files = detect.files_for({"path": str(args.raw)})
    frame_path = Path(files[args.frame])
    t0 = time.perf_counter()
    im = detect.read_image(str(frame_path), cfg["scale_levels"])
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != im.shape:
        raise ValueError("Mask unreadable or mask/frame shape mismatch")
    peaks, confidence = detect2._peaks(im, None, params["thresh"])
    keep = mask[np.clip(peaks[:, 1].astype(int), 0, mask.shape[0] - 1),
                np.clip(peaks[:, 0].astype(int), 0, mask.shape[1] - 1)] > 0
    xy, confidence = peaks[keep], confidence[keep]
    if not len(xy):
        raise ValueError("No real candidate profiles")
    strength, rr = detect2._profile(im, xy)
    coef, ns1, rmse1 = detect2._fit_free(strength, rr)
    solid = ((ns1 >= params["solid_ns"]) & (rmse1 < 1.8)
             & (np.linalg.norm(coef[:, :2], axis=1) < params["max_shift"])
             & (coef[:, 2] > 30) & (coef[:, 2] < 42))
    r0 = float(np.median(coef[solid, 2])) if solid.sum() >= 20 else params["radius_fallback"]
    report["input"] = {
        "case": str(args.case.resolve()), "raw": str(args.raw), "frame": args.frame,
        "frame_path": str(frame_path), "frame_sha256": file_hash(frame_path),
        "image_u8_sha256": array_hash(im), "image_shape": list(im.shape),
        "mask": str(args.mask.resolve()), "mask_sha256": file_hash(args.mask),
        "config_sha256": file_hash(cfg_path), "scale_levels": cfg["scale_levels"],
        "candidate_count": len(xy), "unmasked_peak_count": len(peaks),
        "solid_for_radius_count": int(solid.sum()), "radius_ref": r0,
        "strength_dtype": str(strength.dtype), "rr_dtype": str(rr.dtype),
        "strength_sha256": array_hash(strength), "rr_sha256": array_hash(rr),
        "xy_sha256": array_hash(xy), "confidence_sha256": array_hash(confidence),
        "preparation_including_hashes_s": time.perf_counter() - t0,
        "bbox": None, "prior_xy": None, "search_mask": "the supplied interior mask"}
    effective = {"detector_defaults": params, "radius_ref": r0, "min_strength": 20,
                 "ray_tolerance": 2.2, "rounds": 3, "regularizer": 1e-4,
                 "fit_dtype": "float64", "profile_and_precast_delta_dtype": "float32",
                 "A2_sha256": array_hash(detect2.A2),
                 "warmups": args.warmups, "repeats": args.repeats,
                 "sizes": [len(xy), 512, 3000], "atol": 1e-9, "rtol": 1e-10}
    report["parameters"] = effective
    report["parameters_sha256"] = json_hash(effective)
    # Guard the dtype semantic the port explicitly preserves.
    if (rr - r0).dtype != np.float32:
        raise RuntimeError("Reference scalar promotion changed; this port is not validated")
    t0 = time.perf_counter()
    A2 = torch.from_numpy(detect2.A2).to(device="cuda", dtype=torch.float64)
    eye = torch.eye(2, device="cuda", dtype=torch.float64).unsqueeze(0) * 1e-4
    torch.cuda.synchronize()
    report["cold_start"]["constant_upload_and_setup_s"] = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    report["workloads"] = []
    for label, size in (("real_frame", len(xy)), ("first_512", 512), ("tiled_3000", 3000)):
        if label == "first_512" and len(xy) < 512:
            raise ValueError("Too few candidates for an untiled N=512 subset")
        indices = np.arange(size, dtype=np.int64) % len(xy)
        ss, radii, points = (np.ascontiguousarray(v[indices]) for v in (strength, rr, xy))
        row = {"label": label, "n": size,
               "construction": "identity" if label == "real_frame" else
                               "first 512" if label == "first_512" else
                               "deterministic cyclic repetition of real profiles; not independent samples",
               "source_indices_sha256": array_hash(indices),
               "strength_sha256": array_hash(ss), "rr_sha256": array_hash(radii),
               "input_bytes": ss.nbytes + radii.nbytes}
        report["workloads"].append(row)
        expected = detect2._fit_fixed(ss, radii, r0)
        traced, trace_cpu = numpy_trace(ss, radii, r0, detect2.A2)
        if not all(np.array_equal(a, b) for a, b in zip(expected, traced)):
            raise AssertionError("Diagnostic NumPy trace differs from production fit")
        # First call remains separate from warmed timing (library init is included).
        result, cold = timed_gpu(ss, radii, r0, A2, eye)
        row["gpu_first_call_s"] = cold
        row["first_call_comparison"] = compare(expected, result, params, points)
        # Full per-ray decisions from all three rounds, outside timed output copies.
        ds = torch.from_numpy(ss).to("cuda")
        dr = torch.from_numpy(radii).to("cuda")
        trace_gpu = []
        check = torch_fit(ds, dr, r0, A2, eye, trace=trace_gpu)
        torch.cuda.synchronize()
        row["ray_flag_mismatches_initial_and_rounds_1_to_3"] = [
            int(np.count_nonzero(a != b.cpu().numpy())) for a, b in zip(trace_cpu, trace_gpu)]
        del ds, dr, check, trace_gpu
        for _ in range(args.warmups):
            detect2._fit_fixed(ss, radii, r0)
            timed_gpu(ss, radii, r0, A2, eye)
        cpu_times = []
        gpu_times = {key: [] for key in ("h2d", "compute", "d2h", "total")}
        all_numeric = row["first_call_comparison"]["numeric_and_gate_match"]
        flags_differ = any(row["ray_flag_mismatches_initial_and_rounds_1_to_3"])
        nondeterministic_cpu = 0
        worst_c = worst_rmse = 0.0
        float32_centres_differ = 0
        for repeat in range(args.repeats):
            # Alternate order to reduce monotone drift/temperature bias.
            def cpu_once():
                t = time.perf_counter()
                value = detect2._fit_fixed(ss, radii, r0)
                cpu_times.append(time.perf_counter() - t)
                return value
            if repeat % 2 == 0:
                cpu_result = cpu_once()
                gpu_result, timings = timed_gpu(ss, radii, r0, A2, eye)
            else:
                gpu_result, timings = timed_gpu(ss, radii, r0, A2, eye)
                cpu_result = cpu_once()
            nondeterministic_cpu += int(not all(np.array_equal(a, b) for a, b
                                                in zip(expected, cpu_result)))
            comparison = compare(expected, gpu_result, params, points)
            all_numeric &= comparison["numeric_and_gate_match"]
            flags_differ |= any(comparison["gate_mismatches"].values())
            float32_centres_differ += int(not comparison["float32_centres_equal"])
            worst_c = max(worst_c, comparison["c"]["max_abs_error"])
            worst_rmse = max(worst_rmse, comparison["rmse"]["max_abs_error"])
            for key in gpu_times:
                gpu_times[key].append(timings[key])
        row["cpu_reference"] = summary(cpu_times)
        row["gpu"] = {key: summary(values) for key, values in gpu_times.items()}
        row["speedup_cpu_over_gpu_including_transfers"] = (
            row["cpu_reference"]["median_s"] / row["gpu"]["total"]["median_s"])
        row["speedup_compute_only_not_deployment_metric"] = (
            row["cpu_reference"]["median_s"] / row["gpu"]["compute"]["median_s"])
        row["validation"] = {"all_repeats_numeric_and_gate_match": bool(all_numeric),
                             "any_ray_or_acceptance_flag_difference": bool(flags_differ),
                             "cpu_nondeterministic_repeats": nondeterministic_cpu,
                             "float32_centres_differ_repeats": float32_centres_differ,
                             "worst_c_abs_error": worst_c, "worst_rmse_abs_error": worst_rmse}
        row["eligible_for_further_research_only"] = bool(
            all_numeric and not flags_differ and not nondeterministic_cpu)
        row["memory"] = memory_snapshot(torch)
        print(label, size, "cpu_ms", row["cpu_reference"]["median_s"] * 1000,
              "gpu_total_ms", row["gpu"]["total"]["median_s"] * 1000,
              "speedup", row["speedup_cpu_over_gpu_including_transfers"], flush=True)
    report["memory_final"] = memory_snapshot(torch)
    mismatch = any(not r["eligible_for_further_research_only"] for r in report["workloads"])
    report["decision"] = {
        "any_flag_or_numeric_failure_rejects_gpu_candidate": mismatch,
        "production_adoption": "REJECTED / NOT AUTHORIZED: isolated research only",
        "local_fit_correctness_on_tested_inputs": "FAIL" if mismatch else "PASS within recorded tolerances",
        "warm_speedup_including_transfers_by_size": {
            str(r["n"]): r["speedup_cpu_over_gpu_including_transfers"] for r in report["workloads"]},
        "end_to_end_validated": False,
        "limits": ["One frame only; no complete detection/linking/quality_bits regression",
                   "Acceptance checks are fixed-fit gates, not final detections or recall",
                   "N=3000 tiles existing profiles and does not represent independent discs",
                   "Concurrent GPU use and Windows/WDDM clocks can affect timing",
                   "Torch allocator cap does not cap driver/context/library GPU memory",
                   "Warm timing includes H2D + synchronized compute + D2H; no image I/O or upstream stages"]}
    report["status"] = "measured_rejected_on_correctness" if mismatch else "measured_isolated_only"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--memory-cap-mib", type=int, default=768)
    args = parser.parse_args()
    if args.warmups < 5 or args.repeats < 5:
        parser.error("At least five warmups and five measured repetitions are required")
    if args.memory_cap_mib < 128 or args.memory_cap_mib > 768:
        parser.error("Allocator cap must be 128..768 MiB (context/library memory is additional)")
    if args.out.exists():
        parser.error("Refusing to overwrite an existing output")
    repo = Path(__file__).resolve().parents[1]
    report = {"schema": "fixed-ray-fit-gpu-probe/v1",
              "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "status": "started", "scope": "isolated fixed-ray-fit only; NOT end-to-end",
              "provenance": {"script": str(Path(__file__).resolve()),
                             "script_sha256": file_hash(__file__), "argv": sys.argv,
                             "cwd": str(Path.cwd()),
                             "git_commit": command_output(["git", "rev-parse", "HEAD"], repo),
                             "git_status_before": command_output(["git", "status", "--short"], repo)}}
    started = time.perf_counter()
    exit_code = 0
    try:
        run(args, report)
    except Exception as exc:
        report["status"] = "blocked_or_failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
        report["production_adoption"] = "REJECTED; probe failed, no fallback invented"
        exit_code = 1
        print(traceback.format_exc(), file=sys.stderr)
    report["elapsed_s"] = time.perf_counter() - started
    report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print("report:", args.out, "status:", report["status"], flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
