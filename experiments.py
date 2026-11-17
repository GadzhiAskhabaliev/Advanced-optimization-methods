"""Fair comparison of SGD and R-SGD on the two required simulations.

Run from the repository root: python experiments.py

Tuning uses one dataset and one training seed. The selected learning rate and
radius are frozen before the noise level changes. Evaluation uses five new
datasets. Error bars are the standard deviation across those datasets.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from robust_sgd.data import GENERATORS, train_val_test
from robust_sgd.models import MODELS
from robust_sgd.train import train_model

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results"
FIG = RESULT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PROTOCOL = {
    "logistic": {
        "title": "Simulation 1: logistic regression",
        "n_train": 1000,
        "n_val": 500,
        "n_test": 1000,
        "steps": 300,
        "batch_size": 32,
        "alphas": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        "rhos": [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0],
    },
    "mlp": {
        "title": "Simulation 2: two-layer network",
        "n_train": 1000,
        "n_val": 500,
        "n_test": 1000,
        "steps": 400,
        "batch_size": 32,
        "alphas": [1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 21.0, 27.0],
        "rhos": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
    },
}

EVAL_SEEDS = [1, 2, 3, 4, 5]
TUNE_DATA_SEED = 0
TUNE_INIT_SEED = 1000
TUNE_TRAIN_SEED = 2000
SGD_COLOR = "#0072B2"
RSGD_COLOR = "#D55E00"


def _init_theta(model_name, init_seed):
    return MODELS[model_name]["init"](np.random.default_rng(init_seed))


def _load_data(model_name, data_seed):
    cfg = PROTOCOL[model_name]
    return train_val_test(
        GENERATORS[model_name],
        cfg["n_train"],
        cfg["n_val"],
        cfg["n_test"],
        data_seed,
    )


def gradient_scale(model_name):
    """Median per-example gradient norm at the tuning initialization."""
    data = _load_data(model_name, TUNE_DATA_SEED)
    theta0 = _init_theta(model_name, TUNE_INIT_SEED)
    grads = MODELS[model_name]["example_gradients"](theta0, data["x_train"], data["y_train"])
    norms = np.linalg.norm(grads, axis=1)
    median_norm = float(np.median(norms))
    coordinate_rms = float(np.sqrt(np.mean(grads ** 2)))
    # σ √p is the RMS norm of the perturbation. Levels 0, 0.5 and 1 match
    # that RMS norm to 0, half, and one median clean-gradient norm.
    unit = median_norm / np.sqrt(theta0.size)
    sigmas = [0.0, 0.5 * unit, 1.0 * unit]
    return {
        "median_gradient_norm": median_norm,
        "coordinate_rms": coordinate_rms,
        "parameter_count": int(theta0.size),
        "sigmas": sigmas,
        "tune_sigma": sigmas[1],
    }


def _run(job):
    data = _load_data(job["model"], job["data_seed"])
    theta0 = _init_theta(job["model"], job["init_seed"])
    out = train_model(
        job["model"],
        data,
        theta0,
        job["method"],
        job["alpha0"],
        job["rho"],
        job["sigma"],
        job["steps"],
        job["batch_size"],
        job["train_seed"],
    )
    keep_curves = job.get("keep_curves", False)
    keep_theta = job.get("keep_theta", False)
    payload = {
        "method": job["method"],
        "alpha0": job["alpha0"],
        "rho": job["rho"],
        "sigma": job["sigma"],
        "data_seed": job["data_seed"],
        "init_seed": job["init_seed"],
        "train_seed": job["train_seed"],
        "tag": job["tag"],
        "selection_val_loss": out["selection_val_loss"],
        "final_train_loss": out["final_train_loss"],
        "final_val_loss": out["final_val_loss"],
        "test_loss": out["test_loss"],
        "test_accuracy": out["test_accuracy"],
        "train_accuracy": out["train_accuracy"],
        "zero_step_frequency": out["zero_step_frequency"],
        "mean_shrinkage": out["mean_shrinkage"],
        "mean_angle_deg": out["mean_angle_deg"],
        "update_seconds": out["update_seconds"],
        "gradient_evaluations": out["gradient_evaluations"],
    }
    if keep_curves:
        payload["train_loss"] = out["train_loss"]
        payload["val_loss"] = out["val_loss"]
    if keep_theta:
        payload["theta"] = out["theta"]
    return payload


def parallel_map(jobs):
    if len(jobs) == 1:
        return [_run(jobs[0])]
    workers = min(len(jobs), os.cpu_count() or 1)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if done % 10 == 0 or done == len(futures):
                print(f"  finished {done}/{len(futures)}", flush=True)
    return results


def _base_job(model_name, sigma, tag):
    cfg = PROTOCOL[model_name]
    return {
        "model": model_name,
        "sigma": sigma,
        "steps": cfg["steps"],
        "batch_size": cfg["batch_size"],
        "tag": tag,
        "data_seed": TUNE_DATA_SEED,
        "init_seed": TUNE_INIT_SEED,
        "train_seed": TUNE_TRAIN_SEED,
        "keep_curves": False,
        "keep_theta": False,
    }


def _select(rows, method):
    pool = [row for row in rows if row["method"] == method]
    if method == "sgd":
        return min(pool, key=lambda row: (row["selection_val_loss"], row["alpha0"]))
    return min(pool, key=lambda row: (row["selection_val_loss"], row["rho"], row["alpha0"]))


def tune(model_name, sigma):
    cfg = PROTOCOL[model_name]
    alphas = list(cfg["alphas"])
    rhos = list(cfg["rhos"])
    jobs = []
    for alpha in alphas:
        job = _base_job(model_name, sigma, "tune")
        job.update(method="sgd", alpha0=alpha, rho=0.0)
        jobs.append(job)
        for rho in rhos:
            job = _base_job(model_name, sigma, "tune")
            job.update(method="rsgd", alpha0=alpha, rho=rho)
            jobs.append(job)
    print(f"tuning {model_name}: {len(jobs)} runs at σ={sigma:.4g}")
    rows = parallel_map(jobs)

    def at_edge(values, chosen):
        return chosen <= min(values) + 1e-15 or chosen >= max(values) - 1e-15

    # One expansion if a selected value sits on the boundary of the grid.
    for _ in range(1):
        sgd_best = _select(rows, "sgd")
        rsgd_best = _select(rows, "rsgd")
        extra_alphas = []
        for best in (sgd_best, rsgd_best):
            if best["alpha0"] >= max(alphas) - 1e-15:
                extra_alphas.append(max(alphas) * 3.0)
            if best["alpha0"] <= min(alphas) + 1e-15:
                extra_alphas.append(min(alphas) / 3.0)
        extra_rhos = []
        if rsgd_best["rho"] >= max(rhos) - 1e-15:
            extra_rhos.append(max(rhos) * 2.0)
        extra_alphas = sorted(set(extra_alphas) - set(alphas))
        extra_rhos = sorted(set(extra_rhos) - set(rhos))
        if not extra_alphas and not extra_rhos:
            break
        extra_jobs = []
        for alpha in extra_alphas:
            job = _base_job(model_name, sigma, "tune-extend")
            job.update(method="sgd", alpha0=alpha, rho=0.0)
            extra_jobs.append(job)
            for rho in rhos + extra_rhos:
                job = _base_job(model_name, sigma, "tune-extend")
                job.update(method="rsgd", alpha0=alpha, rho=rho)
                extra_jobs.append(job)
        for rho in extra_rhos:
            for alpha in alphas:
                job = _base_job(model_name, sigma, "tune-extend")
                job.update(method="rsgd", alpha0=alpha, rho=rho)
                extra_jobs.append(job)
        print(f"  extending grid with {len(extra_jobs)} runs")
        rows.extend(parallel_map(extra_jobs))
        alphas = sorted(set(alphas).union(extra_alphas))
        rhos = sorted(set(rhos).union(extra_rhos))

    sgd_best = _select(rows, "sgd")
    rsgd_best = _select(rows, "rsgd")
    return {
        "alphas": alphas,
        "rhos": rhos,
        "runs": len(rows),
        "sgd": _public(sgd_best),
        "rsgd": _public(rsgd_best),
        "sgd_at_alpha_edge": at_edge(alphas, sgd_best["alpha0"]),
        "rsgd_at_alpha_edge": at_edge(alphas, rsgd_best["alpha0"]),
        "rsgd_at_rho_edge": at_edge(rhos, rsgd_best["rho"]),
        "grid": [_public(row) for row in rows],
    }


def _public(row):
    keep = [
        "method",
        "alpha0",
        "rho",
        "sigma",
        "selection_val_loss",
        "final_train_loss",
        "final_val_loss",
        "test_loss",
        "test_accuracy",
        "train_accuracy",
        "zero_step_frequency",
        "mean_shrinkage",
        "mean_angle_deg",
        "update_seconds",
        "gradient_evaluations",
        "data_seed",
        "train_seed",
    ]
    return {key: row[key] for key in keep if key in row}


def _eval_job(model_name, method, alpha0, rho, sigma, data_seed, tag, keep_curves, keep_theta, train_seed=None):
    cfg = PROTOCOL[model_name]
    return {
        "model": model_name,
        "method": method,
        "alpha0": alpha0,
        "rho": rho,
        "sigma": sigma,
        "steps": cfg["steps"],
        "batch_size": cfg["batch_size"],
        "data_seed": data_seed,
        "init_seed": 1000 + data_seed,
        "train_seed": 2000 + data_seed if train_seed is None else train_seed,
        "tag": tag,
        "keep_curves": keep_curves,
        "keep_theta": keep_theta,
    }


def evaluate(model_name, chosen, sigmas):
    jobs = []
    for data_seed in EVAL_SEEDS:
        for sigma in sigmas:
            for method in ("sgd", "rsgd"):
                pick = chosen[method]
                jobs.append(
                    _eval_job(
                        model_name,
                        method,
                        pick["alpha0"],
                        pick["rho"],
                        sigma,
                        data_seed,
                        tag="eval",
                        keep_curves=True,
                        keep_theta=(data_seed == EVAL_SEEDS[0]),
                    )
                )
    print(f"evaluating {model_name}: {len(jobs)} runs")
    return parallel_map(jobs)


def sensitivity(model_name, alpha0, sigmas):
    rhos = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    jobs = []
    for sigma in sigmas:
        for rho in rhos:
            jobs.append(
                _eval_job(
                    model_name,
                    "rsgd",
                    alpha0,
                    rho,
                    sigma,
                    EVAL_SEEDS[0],
                    tag="sensitivity",
                    keep_curves=False,
                    keep_theta=False,
                )
            )
    print(f"sensitivity {model_name}: {len(jobs)} runs")
    rows = parallel_map(jobs)
    return rhos, rows


def training_variability(model_name, chosen, sigma):
    """Same dataset and initialization, five training streams."""
    jobs = []
    for train_seed in range(3001, 3006):
        for method in ("sgd", "rsgd"):
            pick = chosen[method]
            job = _eval_job(
                model_name,
                method,
                pick["alpha0"],
                pick["rho"],
                sigma,
                EVAL_SEEDS[0],
                tag="train-variability",
                keep_curves=False,
                keep_theta=False,
                train_seed=train_seed,
            )
            # Keep the dataset and initialization fixed. Only the training seed changes.
            job["init_seed"] = 1000 + EVAL_SEEDS[0]
            jobs.append(job)
    print(f"training variability {model_name}: {len(jobs)} runs")
    return parallel_map(jobs)


def step_length_control(model_name, chosen, sigma):
    """SGD at the R-SGD learning rate, and SGD with that rate scaled by shrinkage.

    The second control asks whether a shorter SGD step reproduces R-SGD.
    """
    alpha = chosen["rsgd"]["alpha0"]
    rho = chosen["rsgd"]["rho"]
    probe = _run(
        _eval_job(
            model_name,
            "rsgd",
            alpha,
            rho,
            sigma,
            EVAL_SEEDS[0],
            tag="shrink-probe",
            keep_curves=False,
            keep_theta=False,
        )
    )
    shrinkage = probe["mean_shrinkage"] if probe["mean_shrinkage"] not in (None, 0.0) else 1.0
    jobs = [
        _eval_job(model_name, "sgd", alpha, 0.0, sigma, EVAL_SEEDS[0], "step-control", False, False),
        _eval_job(
            model_name,
            "sgd",
            alpha * shrinkage,
            0.0,
            sigma,
            EVAL_SEEDS[0],
            "step-control",
            False,
            False,
        ),
    ]
    rows = parallel_map(jobs)
    return {"probe": _public(probe), "shrinkage_used": shrinkage, "sgd": [_public(row) for row in rows]}


def _aggregate(rows):
    def nanmean(values):
        array = np.array([np.nan if value is None else value for value in values], dtype=float)
        if np.all(np.isnan(array)):
            return None
        return float(np.nanmean(array))

    keys = [
        "final_train_loss",
        "final_val_loss",
        "test_loss",
        "test_accuracy",
        "train_accuracy",
        "zero_step_frequency",
        "mean_shrinkage",
        "mean_angle_deg",
        "update_seconds",
    ]
    summary = {}
    for key in keys:
        values = [row[key] for row in rows]
        summary[key + "_mean"] = nanmean(values)
        array = np.array([np.nan if value is None else value for value in values], dtype=float)
        summary[key + "_std"] = float(np.nanstd(array, ddof=1)) if np.sum(~np.isnan(array)) > 1 else 0.0
    summary["n"] = len(rows)
    summary["gradient_evaluations"] = rows[0]["gradient_evaluations"]
    summary["failed"] = int(sum(not np.isfinite(row["test_loss"]) for row in rows))
    return summary


def plot_losses(model_name, rows, sigmas, suffix="", zoom=False):
    cfg = PROTOCOL[model_name]
    methods = ("sgd", "rsgd")
    colors = {"sgd": SGD_COLOR, "rsgd": RSGD_COLOR}
    styles = {"sgd": "-", "rsgd": "--"}
    labels = {"sgd": "SGD", "rsgd": "R-SGD"}
    fig, axes = plt.subplots(2, len(sigmas), figsize=(3.4 * len(sigmas), 5.6), sharex=True, squeeze=False)
    tails = []
    for col, sigma in enumerate(sigmas):
        for row_i, (field, ylabel) in enumerate((("train_loss", "training loss"), ("val_loss", "validation loss"))):
            ax = axes[row_i, col]
            for method in methods:
                selected = [
                    row[field]
                    for row in rows
                    if row["method"] == method and abs(row["sigma"] - sigma) < 1e-12
                ]
                stack = np.vstack(selected)
                mean = stack.mean(axis=0)
                std = stack.std(axis=0, ddof=1)
                updates = np.arange(1, mean.size + 1)
                ax.plot(
                    updates,
                    mean,
                    color=colors[method],
                    linestyle=styles[method],
                    label=labels[method],
                    linewidth=1.6,
                )
                ax.fill_between(updates, mean - std, mean + std, color=colors[method], alpha=0.16, linewidth=0)
                tails.append(mean[mean.size // 5 :])
            if row_i == 0:
                ax.set_title(f"σ = {sigma:.3g}")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_i == 1:
                ax.set_xlabel("update")
            ax.set_ylim(bottom=0.0)
            ax.grid(True, alpha=0.25)
    if zoom:
        steps = rows[0]["train_loss"].size
        hi = float(np.percentile(np.concatenate(tails), 99.5) * 1.6)
        for ax in axes.ravel():
            ax.set_xlim(steps // 5, steps)
            ax.set_ylim(0.0, max(hi, 1e-3))
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(cfg["title"] + (" (late updates)" if zoom else ""), y=1.06, fontsize=12)
    fig.tight_layout()
    path = FIG / f"{model_name}_losses{suffix}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_boundaries(model_name, rows, sigmas):
    data = _load_data(model_name, EVAL_SEEDS[0])
    raw = np.vstack([data["x_train_raw"], data["x_test_raw"]])
    mins = raw.min(axis=0) - 0.35
    maxs = raw.max(axis=0) + 0.35
    xx, yy = np.meshgrid(
        np.linspace(mins[0], maxs[0], 220),
        np.linspace(mins[1], maxs[1], 220),
    )
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid = (grid - data["mu"]) / data["sd"]
    logits_fn = MODELS[model_name]["logits"]
    methods = ("sgd", "rsgd")
    fig, axes = plt.subplots(len(methods), len(sigmas), figsize=(3.2 * len(sigmas), 6.0), sharex=True, sharey=True)
    for col, sigma in enumerate(sigmas):
        for row_i, method in enumerate(methods):
            match = [
                item
                for item in rows
                if item["method"] == method
                and item["data_seed"] == EVAL_SEEDS[0]
                and abs(item["sigma"] - sigma) < 1e-12
            ][0]
            values = logits_fn(match["theta"], grid).reshape(xx.shape)
            ax = axes[row_i, col]
            ax.contourf(xx, yy, values, levels=[-1e9, 0.0, 1e9], colors=["#c6dbef", "#fdd0a2"])
            ax.contour(xx, yy, values, levels=[0.0], colors="black", linewidths=1.0)
            test = data["x_test_raw"]
            labels = data["y_test"]
            ax.scatter(test[labels == 0, 0], test[labels == 0, 1], s=7, c="#2171b5", alpha=0.75, linewidths=0)
            ax.scatter(test[labels == 1, 0], test[labels == 1, 1], s=7, c="#d94801", alpha=0.75, linewidths=0)
            if row_i == 0:
                ax.set_title(f"σ = {sigma:.3g}")
            if col == 0:
                ax.set_ylabel("SGD" if method == "sgd" else "R-SGD")
            if row_i == len(methods) - 1:
                ax.set_xlabel("feature 1")
    fig.suptitle(f"{PROTOCOL[model_name]['title']}: decision boundary, data seed {EVAL_SEEDS[0]}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = FIG / f"{model_name}_boundary.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_sensitivity(model_name, rhos, rows, sigmas):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for sigma in sigmas:
        selected = [row for row in rows if abs(row["sigma"] - sigma) < 1e-12]
        selected = sorted(selected, key=lambda row: row["rho"])
        rho = [row["rho"] for row in selected]
        axes[0].plot(rho, [row["final_val_loss"] for row in selected], "o-", label=f"σ = {sigma:.3g}")
        axes[1].plot(rho, [row["zero_step_frequency"] for row in selected], "o-", label=f"σ = {sigma:.3g}")
    axes[0].set_xlabel(r"robustness radius $\rho$")
    axes[0].set_ylabel("final validation loss")
    axes[1].set_xlabel(r"robustness radius $\rho$")
    axes[1].set_ylabel("zero-step frequency")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{PROTOCOL[model_name]['title']}: radius sensitivity, data seed {EVAL_SEEDS[0]}")
    fig.tight_layout()
    path = FIG / f"{model_name}_sensitivity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _curves_to_npz(model_name, rows):
    path = RESULT / f"{model_name}_curves.npz"
    payload = {}
    for row in rows:
        key = f"s{row['data_seed']}_{row['method']}_sig{row['sigma']:.6g}"
        payload[key + "_train"] = row["train_loss"]
        payload[key + "_val"] = row["val_loss"]
    np.savez_compressed(path, **payload)
    return str(path)


def paired_differences(rows, sigmas):
    """R-SGD minus SGD on matched seeds. Negative loss means R-SGD is lower."""
    summary = {}
    for sigma in sigmas:
        deltas = []
        for seed in EVAL_SEEDS:
            matched = {}
            for method in ("sgd", "rsgd"):
                matched[method] = next(
                    row
                    for row in rows
                    if row["data_seed"] == seed
                    and row["method"] == method
                    and abs(row["sigma"] - sigma) < 1e-12
                )
            deltas.append(
                {
                    "test_loss": matched["rsgd"]["test_loss"] - matched["sgd"]["test_loss"],
                    "test_accuracy": matched["rsgd"]["test_accuracy"] - matched["sgd"]["test_accuracy"],
                    "train_loss": matched["rsgd"]["final_train_loss"] - matched["sgd"]["final_train_loss"],
                }
            )
        block = {"n": len(deltas)}
        for key in ("test_loss", "test_accuracy", "train_loss"):
            values = np.array([item[key] for item in deltas], dtype=float)
            block[key + "_mean"] = float(values.mean())
            block[key + "_std"] = float(values.std(ddof=1))
            # How many seeds favor R-SGD.
            if key == "test_accuracy":
                block[key + "_rsgd_better"] = int(np.sum(values > 0))
            else:
                block[key + "_rsgd_better"] = int(np.sum(values < 0))
        summary[f"{sigma:.6g}"] = block
    return summary


def run_model(model_name):
    scale = gradient_scale(model_name)
    sigmas = scale["sigmas"]
    print(model_name, json.dumps(scale))
    chosen = tune(model_name, scale["tune_sigma"])
    eval_rows = evaluate(model_name, chosen, sigmas)
    metrics = {}
    for sigma in sigmas:
        metrics[f"{sigma:.6g}"] = {}
        for method in ("sgd", "rsgd"):
            subset = [
                row
                for row in eval_rows
                if row["method"] == method and abs(row["sigma"] - sigma) < 1e-12
            ]
            metrics[f"{sigma:.6g}"][method] = _aggregate(subset)
    rho_grid, sens_rows = sensitivity(model_name, chosen["rsgd"]["alpha0"], [sigmas[0], sigmas[-1]])
    variability = training_variability(model_name, chosen, scale["tune_sigma"])
    variability_summary = {
        method: _aggregate([row for row in variability if row["method"] == method])
        for method in ("sgd", "rsgd")
    }
    control = {
        f"{sigma:.6g}": step_length_control(model_name, chosen, sigma) for sigma in (sigmas[1], sigmas[2])
    }
    loss_path = plot_losses(model_name, eval_rows, sigmas)
    zoom_path = plot_losses(model_name, eval_rows, sigmas, suffix="_zoom", zoom=True)
    boundary_path = plot_boundaries(model_name, eval_rows, sigmas)
    sensitivity_path = plot_sensitivity(model_name, rho_grid, sens_rows, [sigmas[0], sigmas[-1]])
    curve_path = _curves_to_npz(model_name, eval_rows)
    # Drop bulky curves from the JSON summary. Thetas are kept for the boundary seed.
    eval_public = []
    for row in eval_rows:
        item = _public(row)
        item["data_seed"] = row["data_seed"]
        if "theta" in row:
            item["theta"] = row["theta"].tolist()
        eval_public.append(item)
    return {
        "scale": scale,
        "tuning": {
            "alphas": chosen["alphas"],
            "rhos": chosen["rhos"],
            "runs": chosen["runs"],
            "tune_sigma": scale["tune_sigma"],
            "sgd": chosen["sgd"],
            "rsgd": chosen["rsgd"],
            "sgd_at_alpha_edge": chosen["sgd_at_alpha_edge"],
            "rsgd_at_alpha_edge": chosen["rsgd_at_alpha_edge"],
            "rsgd_at_rho_edge": chosen["rsgd_at_rho_edge"],
            "grid": chosen["grid"],
        },
        "metrics": metrics,
        "sensitivity": [_public(row) for row in sens_rows],
        "training_variability": variability_summary,
        "step_length_control": control,
        "figures": {
            "losses": loss_path,
            "losses_zoom": zoom_path,
            "boundary": boundary_path,
            "sensitivity": sensitivity_path,
            "curves": curve_path,
        },
        "paired": paired_differences(eval_rows, sigmas),
        "evaluation_runs": eval_public,
    }


def main():
    summary = {}
    for model_name in ("logistic", "mlp"):
        summary[model_name] = run_model(model_name)
    path = RESULT / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {path}")
    for model_name in ("logistic", "mlp"):
        print(f"\n== {model_name} ==")
        tuning = summary[model_name]["tuning"]
        print(
            "SGD α={alpha0} val={selection_val_loss:.4f}; R-SGD α={a} ρ={rho} val={v:.4f}".format(
                alpha0=tuning["sgd"]["alpha0"],
                selection_val_loss=tuning["sgd"]["selection_val_loss"],
                a=tuning["rsgd"]["alpha0"],
                rho=tuning["rsgd"]["rho"],
                v=tuning["rsgd"]["selection_val_loss"],
            )
        )
        for sigma, methods in summary[model_name]["metrics"].items():
            for method, stats in methods.items():
                print(
                    f"σ={sigma} {method}: test loss {stats['test_loss_mean']:.4f}±{stats['test_loss_std']:.4f} "
                    f"acc {stats['test_accuracy_mean']:.3f}±{stats['test_accuracy_std']:.3f} "
                    f"zero {stats['zero_step_frequency_mean']:.3f} time {stats['update_seconds_mean']:.3f}s"
                )


if __name__ == "__main__":
    main()
