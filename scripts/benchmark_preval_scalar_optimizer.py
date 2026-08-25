"""Benchmark Brent calibration against the frozen BFGS PreVal baseline.

The baseline class is loaded directly from the ``preval-hc2-experiment`` Git
revision, so the benchmark does not require branch switching or two checkouts.
The candidate implementation is loaded from the current working tree.
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASELINE_REVISION = "preval-hc2-experiment"
MODULE_PATH = "aeon/classification/sklearn/_prevalidated_ridge_classifier.py"
LAMBDAS = np.logspace(-3, 3, 10, dtype=np.float32)
CASES = (
    # Progress n while keeping p small, then increase p in both SVD branches.
    ("n64_p8_c2", 64, 8, 2, 101),
    ("n256_p8_c3", 256, 8, 3, 102),
    ("n1024_p8_c5", 1_024, 8, 5, 103),
    ("n64_p64_c3", 64, 64, 3, 104),
    ("n256_p64_c5", 256, 64, 5, 105),
    ("n1024_p64_c2", 1_024, 64, 2, 106),
    ("n64_p256_c3", 64, 256, 3, 107),
    ("n256_p512_c5", 256, 512, 5, 108),
)
REAL_CASES = (
    # MiniRocket feature counts deliberately vary as well as UCR train size/classes.
    ("Coffee", 1_000),
    ("FaceFour", 2_500),
    ("MedicalImages", 5_000),
    ("SwedishLeaf", 7_500),
    ("ShapesAll", 10_000),
)


def _git(repo, *args):
    """Run a read-only Git command in the selected repository."""
    return subprocess.run(
        ["git", "-c", f"safe.directory={repo.as_posix()}", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_classes(repo):
    """Load the frozen baseline module and current working-tree class."""
    sys.path.insert(0, str(repo))
    from aeon.classification.sklearn import PrevalidatedRidgeClassifier
    from aeon.classification.sklearn import (
        _prevalidated_ridge_classifier as candidate_module,
    )

    source = _git(repo, "show", f"{BASELINE_REVISION}:{MODULE_PATH}")
    baseline_module = types.ModuleType("preval_frozen_bfgs")
    baseline_module.__file__ = f"{BASELINE_REVISION}:{MODULE_PATH}"
    exec(compile(source, baseline_module.__file__, "exec"), baseline_module.__dict__)
    return (
        baseline_module.PrevalidatedRidgeClassifier,
        baseline_module,
        PrevalidatedRidgeClassifier,
        candidate_module,
    )


def _make_data(n_cases, n_atts, n_classes, seed):
    """Create deterministic, balanced data with a learnable class signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_cases, n_atts)).astype(np.float32)
    weights = rng.normal(size=n_atts).astype(np.float32)
    order = np.argsort(X @ weights)
    y = np.empty(n_cases, dtype=np.int64)
    y[order] = np.arange(n_cases) * n_classes // n_cases
    return X, y


def _synthetic_cases():
    """Return the original deterministic synthetic benchmark cases."""
    for name, n_cases, n_atts, n_classes, seed in CASES:
        X, y = _make_data(n_cases, n_atts, n_classes, seed)
        yield {
            "case": name,
            "suite": "synthetic",
            "dataset": "",
            "transform": "synthetic",
            "seed": seed,
            "requested_features": n_atts,
            "X": X,
            "y": y,
        }


def _real_cache_paths(cache_dir, dataset, n_kernels):
    """Return paths for one transformed and scaled UCR feature cache."""
    target = cache_dir / "MiniRocket" / f"{dataset}_k{n_kernels}"
    return {
        "directory": target,
        "train": target / "train.npy",
        "test": target / "test.npy",
        "labels": target / "labels.npz",
        "metadata": target / "metadata.json",
    }


def _prepare_real_cache(dataset, n_kernels, paths, data_dir, n_jobs):
    """Transform and scale a UCR dataset once, outside all timed fits."""
    from aeon.datasets import load_classification
    from aeon.transformations.collection.convolution_based import MiniRocket

    print(f"Preparing cached MiniRocket features for {dataset}...", flush=True)
    X_train, y_train = load_classification(
        dataset, split="train", extract_path=data_dir
    )
    X_test, y_test = load_classification(dataset, split="test", extract_path=data_dir)
    X_train, X_test = np.asarray(X_train), np.asarray(X_test)
    if X_train.ndim == 2:
        X_train, X_test = X_train[:, None, :], X_test[:, None, :]

    encoder = LabelEncoder().fit(np.concatenate((y_train, y_test)))
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    transform = MiniRocket(
        n_kernels=n_kernels,
        max_dilations_per_kernel=32,
        n_jobs=n_jobs,
        random_state=0,
    )
    transform.fit(X_train, y_train)
    raw_train = np.asarray(transform.transform(X_train), dtype=np.float32)
    raw_test = np.asarray(transform.transform(X_test), dtype=np.float32)
    scaler = StandardScaler(with_mean=False)
    feature_train = scaler.fit_transform(raw_train).astype(np.float32, copy=False)
    feature_test = scaler.transform(raw_test).astype(np.float32, copy=False)

    paths["directory"].mkdir(parents=True, exist_ok=True)
    np.save(paths["train"], feature_train)
    np.save(paths["test"], feature_test)
    np.savez(
        paths["labels"],
        y_train=y_train,
        y_test=y_test,
        classes=encoder.classes_,
    )
    metadata = {
        "dataset": dataset,
        "transform": "MiniRocket",
        "n_kernels_requested": n_kernels,
        "n_features": int(feature_train.shape[1]),
        "n_train": int(feature_train.shape[0]),
        "n_test": int(feature_test.shape[0]),
        "n_classes": int(np.unique(y_train).size),
        "random_state": 0,
        "scaler": "StandardScaler(with_mean=False)",
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _real_cases(cache_dir, data_dir, prepare_cache, n_jobs):
    """Load cached MiniRocket UCR features without timing transformation or I/O."""
    for dataset, n_kernels in REAL_CASES:
        paths = _real_cache_paths(cache_dir, dataset, n_kernels)
        required = (paths["train"], paths["test"], paths["labels"], paths["metadata"])
        if not all(path.is_file() for path in required):
            if not prepare_cache:
                raise FileNotFoundError(
                    f"Missing transformed cache for {dataset}: {paths['directory']}. "
                    "Re-run with --prepare-cache."
                )
            _prepare_real_cache(dataset, n_kernels, paths, data_dir, n_jobs)

        # Materialise before timing so neither memory-map paging nor disk I/O is timed.
        X = np.asarray(np.load(paths["train"]), dtype=np.float32).copy()
        labels = np.load(paths["labels"], allow_pickle=True)
        y = np.asarray(labels["y_train"]).copy()
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        yield {
            "case": f"MiniRocket:{dataset}",
            "suite": "ucr",
            "dataset": dataset,
            "transform": "MiniRocket",
            "seed": 0,
            "requested_features": n_kernels,
            "X": X,
            "y": y,
            "cache_metadata": metadata,
        }


@contextmanager
def _time_optimizer(module, attribute):
    """Measure cumulative wall time inside optimizer calls during one fit."""
    original = getattr(module, attribute)
    measurement = {"seconds": 0.0, "calls": 0}

    def timed_optimizer(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            measurement["seconds"] += time.perf_counter() - start
            measurement["calls"] += 1

    setattr(module, attribute, timed_optimizer)
    try:
        yield measurement
    finally:
        setattr(module, attribute, original)


def _timed_fit(estimator_class, module, optimizer_attribute, X, y):
    """Time total fit and the optimizer portion of that fit."""
    estimator = estimator_class(lambdas=LAMBDAS.copy())
    with _time_optimizer(module, optimizer_attribute) as optimizer:
        start = time.perf_counter()
        estimator.fit(X.copy(), y.copy())
        fit_seconds = time.perf_counter() - start
    return estimator, fit_seconds, optimizer


def _correctness(baseline, candidate, X):
    """Return output-difference diagnostics for one fitted estimator pair."""
    baseline_proba = baseline.predict_proba(X.copy())
    candidate_proba = candidate.predict_proba(X.copy())
    return {
        "lambda_equal": bool(candidate.lambda_ == baseline.lambda_),
        "baseline_lambda": float(baseline.lambda_),
        "candidate_lambda": float(candidate.lambda_),
        "baseline_scale": float(baseline.scale_),
        "candidate_scale": float(candidate.scale_),
        "scale_abs_difference": float(abs(candidate.scale_ - baseline.scale_)),
        "probability_max_abs_difference": float(
            np.max(np.abs(candidate_proba - baseline_proba))
        ),
        "probabilities_exact": bool(np.array_equal(candidate_proba, baseline_proba)),
        "predictions_equal": bool(
            np.array_equal(candidate.predict(X.copy()), baseline.predict(X.copy()))
        ),
    }


def _write_csv(path, rows, fieldnames):
    """Write rows using a stable, analysis-friendly column order."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aeon-repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Aeon checkout containing the candidate working tree (default: %(default)s)",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--suite",
        choices=("synthetic", "ucr", "both"),
        default="synthetic",
        help="Case suite to run (default: %(default)s)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("benchmark-data/preval-ucr-features"),
        help="Persistent transformed-feature cache (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmark-data/ucr"),
        help="UCR download/extraction directory used only to prepare missing caches",
    )
    parser.add_argument(
        "--prepare-cache",
        action="store_true",
        help="Create any missing UCR feature caches before benchmark timing",
    )
    parser.add_argument(
        "--transform-jobs",
        type=int,
        default=1,
        help="MiniRocket jobs used only while preparing missing caches",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark-results/preval-minimize-scalar"),
    )
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 5:
        parser.error("use at least one warm-up and five measured repetitions")

    repo = args.aeon_repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_class, baseline_module, candidate_class, candidate_module = _load_classes(
        repo
    )

    implementations = {
        "baseline_bfgs": (baseline_class, baseline_module, "minimize"),
        "candidate_brent": (
            candidate_class,
            candidate_module,
            "minimize_scalar",
        ),
    }
    cases = []
    if args.suite in {"synthetic", "both"}:
        cases.extend(_synthetic_cases())
    if args.suite in {"ucr", "both"}:
        cases.extend(
            _real_cases(
                args.cache_dir.resolve(),
                args.data_dir.resolve(),
                args.prepare_cache,
                args.transform_jobs,
            )
        )

    total_runs = len(cases) * (args.warmups + args.repeats) * len(implementations)
    completed = 0
    raw_rows = []
    summary_rows = []

    print(
        f"Comparing {BASELINE_REVISION} (BFGS) with the current working tree "
        f"(Brent): {total_runs} fits"
    )
    for case_index, case in enumerate(cases):
        name, X, y = case["case"], case["X"], case["y"]
        n_cases, n_atts = X.shape
        n_classes = len(np.unique(y))
        measured = {implementation: [] for implementation in implementations}
        last_estimators = {}

        for iteration in range(args.warmups + args.repeats):
            phase = "warmup" if iteration < args.warmups else "measured"
            repeat = iteration if phase == "warmup" else iteration - args.warmups
            order = list(implementations)
            if (case_index + iteration) % 2:
                order.reverse()

            for implementation in order:
                estimator_class, module, optimizer_attribute = implementations[
                    implementation
                ]
                estimator, fit_seconds, optimizer = _timed_fit(
                    estimator_class, module, optimizer_attribute, X, y
                )
                last_estimators[implementation] = estimator
                completed += 1
                print(
                    f"[{completed:03d}/{total_runs:03d}] "
                    f"{100 * completed / total_runs:5.1f}%  {name:<16} "
                    f"{phase:<8} {implementation:<16} fit={fit_seconds:.6f}s",
                    flush=True,
                )
                if phase == "measured":
                    row = {
                        "case": name,
                        "suite": case["suite"],
                        "dataset": case["dataset"],
                        "transform": case["transform"],
                        "n_cases": n_cases,
                        "n_atts": n_atts,
                        "n_classes": n_classes,
                        "seed": case["seed"],
                        "requested_features": case["requested_features"],
                        "repeat": repeat,
                        "implementation": implementation,
                        "fit_seconds": fit_seconds,
                        "optimizer_seconds": optimizer["seconds"],
                        "optimizer_calls": optimizer["calls"],
                    }
                    raw_rows.append(row)
                    measured[implementation].append(row)

        differences = _correctness(
            last_estimators["baseline_bfgs"],
            last_estimators["candidate_brent"],
            X,
        )
        baseline_fit = np.median(
            [row["fit_seconds"] for row in measured["baseline_bfgs"]]
        )
        candidate_fit = np.median(
            [row["fit_seconds"] for row in measured["candidate_brent"]]
        )
        baseline_optimizer = np.median(
            [row["optimizer_seconds"] for row in measured["baseline_bfgs"]]
        )
        candidate_optimizer = np.median(
            [row["optimizer_seconds"] for row in measured["candidate_brent"]]
        )
        summary_rows.append(
            {
                "case": name,
                "suite": case["suite"],
                "dataset": case["dataset"],
                "transform": case["transform"],
                "n_cases": n_cases,
                "n_atts": n_atts,
                "n_classes": n_classes,
                "requested_features": case["requested_features"],
                "baseline_fit_median_seconds": baseline_fit,
                "candidate_fit_median_seconds": candidate_fit,
                "fit_ratio_candidate_over_baseline": candidate_fit / baseline_fit,
                "baseline_optimizer_median_seconds": baseline_optimizer,
                "candidate_optimizer_median_seconds": candidate_optimizer,
                "optimizer_ratio_candidate_over_baseline": (
                    candidate_optimizer / baseline_optimizer
                ),
                **differences,
            }
        )

    raw_path = output_dir / "raw_timings.csv"
    summary_path = output_dir / "summary.csv"
    metadata_path = output_dir / "metadata.json"
    _write_csv(raw_path, raw_rows, list(raw_rows[0]))
    _write_csv(summary_path, summary_rows, list(summary_rows[0]))
    metadata = {
        "baseline_revision": BASELINE_REVISION,
        "baseline_commit": _git(repo, "rev-parse", BASELINE_REVISION),
        "candidate_branch": _git(repo, "branch", "--show-current"),
        "candidate_head": _git(repo, "rev-parse", "HEAD"),
        "candidate_worktree_diff": _git(repo, "diff", "--", MODULE_PATH),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "suite": args.suite,
        "real_cases": [
            {"dataset": dataset, "minirocket_kernels": kernels}
            for dataset, kernels in REAL_CASES
        ],
        "cache_dir": str(args.cache_dir.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "prepare_cache": args.prepare_cache,
        "lambdas": LAMBDAS.tolist(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved raw timings to {raw_path}")
    print(f"Saved summaries and correctness diagnostics to {summary_path}")
    print(f"Saved environment and revision metadata to {metadata_path}")


if __name__ == "__main__":
    main()
