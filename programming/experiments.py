import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from data import ImageData
from model import init_model, predict, Model
from metrics import accuracy
from sgd import SGDConfig, train_sgd, SGDHistory
from full_gd import GDConfig, train_full_gd, GDHistory


@dataclass
class RunResult:
    algo: str  # "sgd" or "gd"
    params: dict[str, float | int]
    model: Model
    hist: Any  # SGDHistory or GDHistory
    wall_time: float
    train_obj: float
    train_acc: Optional[float] = None
    val_acc: Optional[float] = None
    test_acc: Optional[float] = None
    pred_val: Optional[np.ndarray] = None
    pred_test: Optional[np.ndarray] = None


def _stack_train_val(data: ImageData) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([data.images_train, data.images_val])
    y = np.concatenate([data.labels_train, data.labels_val])
    return x, y


def pick_best(runs: list[RunResult], key: str = "val_acc") -> RunResult:
    if key not in ("val_acc", "test_acc", "train_acc", "train_obj"):
        raise ValueError(f"Unsupported key={key}")
    # maximize accuracy, minimize objective
    if key == "train_obj":
        return min(runs, key=lambda r: float(r.train_obj))
    return max(runs, key=lambda r: float(getattr(r, key) or -1.0))


def sweep_sgd_l2(
    data: ImageData,
    reg_values: list[float],
    *,
    baseline_lr: float,
    baseline_B: int,
    seed: int,
    max_epochs: int,
    eval_every: int,
    target_acc: Optional[float],
    target_patience: int = 1,
) -> list[RunResult]:
    runs: list[RunResult] = []

    for lam in reg_values:
        model = init_model(seed=seed)
        cfg = SGDConfig(
            learning_rate=baseline_lr,
            batch_size=baseline_B,
            l2_reg=lam,
            shuffle=True,
            seed=seed,
            max_epochs=max_epochs,
            target_acc=target_acc,
            target_patience=target_patience,
            eval_every=eval_every,
        )

        t0 = time.perf_counter()
        hist: SGDHistory = train_sgd(
            model,
            data.images_train, data.labels_train, cfg,
            val_images=data.images_val, val_labels=data.labels_val
        )
        t1 = time.perf_counter()

        if hist.val_acc is not None:
            val_acc = float(hist.val_acc[-1])
            pred_val = None
        else:
            pred_val = predict(model, data.images_val)
            val_acc = float(accuracy(pred_val, data.labels_val))

        runs.append(RunResult(
            algo="sgd",
            params={"l2": lam, "lr": baseline_lr, "B": baseline_B},
            model=model,
            hist=hist,
            wall_time=t1 - t0,
            train_obj=float(hist.train_loss[-1]),
            val_acc=val_acc,
            pred_val=pred_val,
        ))

    return runs


def sweep_sgd_lr(
    data: ImageData,
    learning_rates: list[float],
    *,
    l2_reg: float,
    baseline_B: int,
    seed: int,
    max_epochs: int,
    eval_every: int,
    target_acc: Optional[float],
    target_patience: int = 1,
) -> list[RunResult]:
    runs: list[RunResult] = []

    for lr in learning_rates:
        model = init_model(seed=seed)
        cfg = SGDConfig(
            learning_rate=lr,
            batch_size=baseline_B,
            l2_reg=l2_reg,
            shuffle=True,
            seed=seed,
            max_epochs=max_epochs,
            target_acc=target_acc,
            target_patience=target_patience,
            eval_every=eval_every,
        )

        t0 = time.perf_counter()
        hist: SGDHistory = train_sgd(
            model,
            data.images_train, data.labels_train, cfg,
            val_images=data.images_val, val_labels=data.labels_val
        )
        t1 = time.perf_counter()

        pred_train = predict(model, data.images_train)
        train_acc = float(accuracy(pred_train, data.labels_train))

        if hist.val_acc is not None:
            val_acc = float(hist.val_acc[-1])
            pred_val = None
        else:
            pred_val = predict(model, data.images_val)
            val_acc = float(accuracy(pred_val, data.labels_val))

        runs.append(RunResult(
            algo="sgd",
            params={"lr": lr, "l2": l2_reg, "B": baseline_B},
            model=model,
            hist=hist,
            wall_time=t1 - t0,
            train_obj=float(hist.train_loss[-1]),
            train_acc=train_acc,
            val_acc=val_acc,
            pred_val=pred_val,
        ))

    return runs


def sweep_sgd_batch(
    data: ImageData,
    batch_sizes: list[int],
    *,
    l2_reg: float,
    learning_rate: float,
    seed: int,
    max_epochs: int,
    eval_every: int,
    target_acc: Optional[float],
    target_patience: int = 1,
) -> list[RunResult]:
    runs: list[RunResult] = []

    for B in batch_sizes:
        model = init_model(seed=seed)
        cfg = SGDConfig(
            learning_rate=learning_rate,
            batch_size=B,
            l2_reg=l2_reg,
            shuffle=True,
            seed=seed,
            max_epochs=max_epochs,
            target_acc=target_acc,
            target_patience=target_patience,
            eval_every=eval_every,
        )

        t0 = time.perf_counter()
        hist: SGDHistory = train_sgd(
            model,
            data.images_train, data.labels_train, cfg,
            val_images=data.images_val, val_labels=data.labels_val
        )
        t1 = time.perf_counter()

        if hist.val_acc is not None:
            val_acc = float(hist.val_acc[-1])
            pred_val = None
        else:
            pred_val = predict(model, data.images_val)
            val_acc = float(accuracy(pred_val, data.labels_val))

        runs.append(RunResult(
            algo="sgd",
            params={"B": B, "lr": learning_rate, "l2": l2_reg},
            model=model,
            hist=hist,
            wall_time=t1 - t0,
            train_obj=float(hist.train_loss[-1]),
            val_acc=val_acc,
            pred_val=pred_val,
        ))

    return runs


def train_final_sgd(
    data: ImageData,
    *,
    l2_reg: float,
    learning_rate: float,
    batch_size: int,
    seed: int,
    max_epochs: int,
    eval_every: int,
    target_acc: Optional[float],
    target_patience: int = 1,
) -> RunResult:
    X_final, y_final = _stack_train_val(data)

    model = init_model(seed=seed)
    cfg = SGDConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        l2_reg=l2_reg,
        shuffle=True,
        seed=seed,
        max_epochs=max_epochs,
        target_acc=target_acc,
        target_patience=target_patience,
        eval_every=eval_every,
    )

    t0 = time.perf_counter()
    hist: SGDHistory = train_sgd(model, X_final, y_final, cfg)
    t1 = time.perf_counter()

    pred_train = predict(model, X_final)
    pred_test = predict(model, data.images_test)

    return RunResult(
        algo="sgd",
        params={"l2": l2_reg, "lr": learning_rate, "B": batch_size},
        model=model,
        hist=hist,
        wall_time=t1 - t0,
        train_obj=float(hist.train_loss[-1]),
        train_acc=float(accuracy(pred_train, y_final)),
        test_acc=float(accuracy(pred_test, data.labels_test)),
        pred_test=pred_test,
    )


def sweep_gd_lr(
    data: ImageData,
    learning_rates: list[float],
    *,
    l2_reg: float,
    seed: int,
    max_iters: int,
) -> list[RunResult]:
    runs: list[RunResult] = []

    for lr in learning_rates:
        model = init_model(seed=seed)
        cfg = GDConfig(
            learning_rate=lr,
            max_iters=max_iters,
            l2_reg=l2_reg,
        )

        t0 = time.perf_counter()
        hist: GDHistory = train_full_gd(
            model,
            data.images_train, data.labels_train, cfg,
            val_images=data.images_val, val_labels=data.labels_val
        )
        t1 = time.perf_counter()

        pred_train = predict(model, data.images_train)
        train_acc = float(accuracy(pred_train, data.labels_train))

        if hist.val_acc is not None:
            val_acc = float(hist.val_acc[-1])
            pred_val = None
        else:
            pred_val = predict(model, data.images_val)
            val_acc = float(accuracy(pred_val, data.labels_val))

        runs.append(RunResult(
            algo="gd",
            params={"lr": lr, "l2": l2_reg},
            model=model,
            hist=hist,
            wall_time=t1 - t0,
            train_obj=float(hist.train_loss[-1]),
            train_acc=train_acc,
            val_acc=val_acc,
            pred_val=pred_val,
        ))

    return runs


def train_final_gd(
    data: ImageData,
    *,
    l2_reg: float,
    learning_rate: float,
    seed: int,
    max_iters: int,
) -> RunResult:
    X_final, y_final = _stack_train_val(data)

    model = init_model(seed=seed)
    cfg = GDConfig(
        learning_rate=learning_rate,
        max_iters=max_iters,
        l2_reg=l2_reg,
    )

    t0 = time.perf_counter()
    hist: GDHistory = train_full_gd(model, X_final, y_final, cfg)
    t1 = time.perf_counter()

    pred_train = predict(model, X_final)
    pred_test = predict(model, data.images_test)

    return RunResult(
        algo="gd",
        params={"l2": l2_reg, "lr": learning_rate},
        model=model,
        hist=hist,
        wall_time=t1 - t0,
        train_obj=float(hist.train_loss[-1]),
        train_acc=float(accuracy(pred_train, y_final)),
        test_acc=float(accuracy(pred_test, data.labels_test)),
        pred_test=pred_test,
    )


def reference_f_star_fixed_gd(X_ref, y_ref, *, l2: float, lr: float, max_iters: int) -> float:
    model_ref = init_model(seed=12345)
    cfg_ref = GDConfig(learning_rate=lr, l2_reg=l2, max_iters=max_iters)
    hist_ref = train_full_gd(model_ref, X_ref, y_ref, cfg_ref)
    return float(min(hist_ref.train_loss))
