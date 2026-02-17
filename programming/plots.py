import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from experiments import RunResult


def plot_objective_curves(
    runs: list[RunResult],
    *,
    title: str,
    label_fn,
) -> None:
    plt.figure()
    for r in runs:
        y = np.asarray(r.hist.train_loss, dtype=float)
        x = np.arange(len(y))
        plt.plot(x, y, label=label_fn(r))
    plt.xlabel("Logged steps")
    plt.ylabel("Objective value")
    plt.title(title)
    plt.legend()


def plot_metric_vs_param(
    runs: list[RunResult],
    *,
    param_key: str,
    metric_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    logx: bool = True,
) -> None:
    xs = np.asarray([float(r.params[param_key]) for r in runs], dtype=float)
    ys = np.asarray([float(getattr(r, metric_key) or np.nan) for r in runs], dtype=float)

    plt.figure()
    if logx:
        xs_plot = xs.copy()
        xs_plot[xs_plot == 0.0] = 1e-18
        plt.semilogx(xs_plot, ys, marker="o")
    else:
        plt.plot(xs, ys, marker="o")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def plot_val_acc_vs_time(
    runs: list[RunResult],
    *,
    title: str,
    label_fn,
) -> None:
    plt.figure()
    for r in runs:
        hist = r.hist
        if hist.val_acc is None:
            continue
        plt.plot(hist.time_points, hist.val_acc, label=label_fn(r))
    plt.xlabel("Time (s)")
    plt.ylabel("Validation accuracy")
    plt.title(title)
    plt.legend()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, *, title: str) -> None:
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(title)


def plot_compare_objective_and_gap(
    sgd_train_loss: list[float],
    gd_train_loss: list[float],
    *,
    eval_every_sgd: int = 1,
    title_prefix: str = "SGD vs Full-GD",
    f_star: float | None = None,
) -> None:
    sgd = np.asarray(sgd_train_loss, float)
    gd = np.asarray(gd_train_loss, float)

    x_sgd = np.arange(len(sgd)) * eval_every_sgd
    x_gd = np.arange(len(gd))

    # objective
    plt.figure()
    plt.plot(x_sgd, sgd, label="SGD final")
    plt.plot(x_gd, gd, label="Full-GD final")
    plt.xlabel("Data passes")
    plt.ylabel("Objective value")
    plt.title(f"{title_prefix}: objective vs data passes")
    plt.legend()

    # gap
    if f_star is None:
        f_star = float(np.min(gd))
    sgd_gap = np.maximum(sgd - f_star, 0.0)
    gd_gap = np.maximum(gd - f_star, 0.0)

    plt.figure()
    plt.semilogy(x_sgd, sgd_gap + 1e-16, label="SGD final gap")
    plt.semilogy(x_gd, gd_gap + 1e-16, label="Full-GD final gap")
    plt.xlabel("Data passes")
    plt.ylabel(r"$f(x_k) - f^\star$")
    plt.title(f"{title_prefix}: optimality gap vs data passes (log)")
    plt.legend()
