import numpy as np
from sklearn.metrics import classification_report

from data import load_data
from experiments import (
    sweep_sgd_l2, sweep_sgd_lr, sweep_sgd_batch,
    sweep_gd_lr,
    pick_best,
    train_final_sgd, train_final_gd,
    reference_f_star_fixed_gd,
)
from plots import (
    plot_objective_curves, plot_metric_vs_param, plot_val_acc_vs_time,
    plot_confusion, plot_compare_objective_and_gap,
)


def main() -> None:
    # Data
    data = load_data(train_size=0.7, val_size=0.15, test_size=0.15, random_state=2)

    # Global experiment settings
    seed = 2

    max_epochs = 1000
    eval_every = 1
    target_acc = 0.98 # None => run for max_epochs

    # Grids
    reg_values = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    learning_rates = [0.05, 0.1, 0.2, 0.5, 1.0]
    batch_sizes = [8, 16, 32, 64, 128, 256]

    baseline_lr = 0.5
    baseline_B = 64

    # 1) Tune l2 (SGD)
    sgd_l2_runs = sweep_sgd_l2(
        data, reg_values,
        baseline_lr=baseline_lr,
        baseline_B=baseline_B,
        seed=seed,
        max_epochs=max_epochs,
        eval_every=eval_every,
        target_acc=target_acc,
    )
    best_l2_run = pick_best(sgd_l2_runs, key="val_acc")
    best_l2 = float(best_l2_run.params["l2"])

    plot_objective_curves(
        sgd_l2_runs,
        title=f"SGD: objective vs logged steps (varying l2, lr={baseline_lr:g}, B={baseline_B})",
        label_fn=lambda r: ("no reg" if r.params["l2"] == 0.0 else f"l2={r.params['l2']:g}"),
    )
    plot_metric_vs_param(
        sgd_l2_runs,
        param_key="l2",
        metric_key="val_acc",
        title=f"SGD: val_acc vs l2 (lr={baseline_lr:g}, B={baseline_B})",
        xlabel=r"$\lambda$",
        ylabel="Validation accuracy",
        logx=True,
    )

    # 2) Tune lr (SGD)
    sgd_lr_runs = sweep_sgd_lr(
        data, learning_rates,
        l2_reg=best_l2,
        baseline_B=baseline_B,
        seed=seed,
        max_epochs=max_epochs,
        eval_every=eval_every,
        target_acc=target_acc,
    )
    best_lr_run = pick_best(sgd_lr_runs, key="val_acc")
    best_lr = float(best_lr_run.params["lr"])

    plot_val_acc_vs_time(
        sgd_lr_runs,
        title=f"SGD: val_acc vs time (l2={best_l2:g}, B={baseline_B})",
        label_fn=lambda r: f"lr={r.params['lr']:g}",
    )

    # 3) Tune batch size (SGD)
    sgd_B_runs = sweep_sgd_batch(
        data, batch_sizes,
        l2_reg=best_l2,
        learning_rate=best_lr,
        seed=seed,
        max_epochs=max_epochs,
        eval_every=eval_every,
        target_acc=target_acc,
    )
    best_B_run = pick_best(sgd_B_runs, key="val_acc")
    best_B = int(best_B_run.params["B"])

    plot_metric_vs_param(
        sgd_B_runs,
        param_key="B",
        metric_key="val_acc",
        title=f"SGD: val_acc vs batch size (l2={best_l2:g}, lr={best_lr:g})",
        xlabel="Batch size B",
        ylabel="Validation accuracy",
        logx=True,
    )

    # 4) FINAL SGD (train+val) -> test
    final_sgd = train_final_sgd(
        data,
        l2_reg=best_l2,
        learning_rate=best_lr,
        batch_size=best_B,
        seed=seed,
        max_epochs=max_epochs,
        eval_every=eval_every,
        target_acc=None,
    )
    print("=== FINAL SGD ===")
    print(final_sgd.params)
    print("Train objective:", final_sgd.train_obj)
    print("Train acc:", final_sgd.train_acc)
    print("Test acc:", final_sgd.test_acc)
    print("Classification report (SGD final, test):")
    print(classification_report(data.labels_test, final_sgd.pred_test))
    print()

    # 5) Tune lr (GD) using same l2 (compare under same regularization)
    gd_learning_rates = [0.05, 0.1, 0.2, 0.5, 1.0]
    gd_iters = 1000

    gd_lr_runs = sweep_gd_lr(
        data, gd_learning_rates,
        l2_reg=best_l2,
        seed=seed,
        max_iters=gd_iters,
    )
    best_gd_run = pick_best(gd_lr_runs, key="val_acc")
    best_gd_lr = float(best_gd_run.params["lr"])

    plot_val_acc_vs_time(
        gd_lr_runs,
        title=f"Full-GD: val_acc vs time (l2={best_l2:g})",
        label_fn=lambda r: f"lr={r.params['lr']:g}",
    )

    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(best_lr_run.hist.time_points, best_lr_run.hist.val_acc, label=f"SGD best lr={best_lr:g}")
    plt.plot(best_gd_run.hist.time_points, best_gd_run.hist.val_acc, label=f"GD best lr={best_gd_lr:g}")
    plt.xlabel("Time (s)")
    plt.ylabel("Validation accuracy")
    plt.title("Validation accuracy vs time: SGD(best) vs Full-GD(best)")
    plt.legend()

    # 6) FINAL GD (train+val) -> test
    final_gd = train_final_gd(
        data,
        l2_reg=best_l2,
        learning_rate=best_gd_lr,
        seed=seed,
        max_iters=gd_iters,
    )
    print("=== FINAL Full-GD ===")
    print(final_gd.params)
    print("Train objective:", final_gd.train_obj)
    print("Train acc:", final_gd.train_acc)
    print("Test acc:", final_gd.test_acc)
    print("Classification report (Full-GD final, test):")
    print(classification_report(data.labels_test, final_gd.pred_test))
    print()

    # Confusion matrices
    plot_confusion(data.labels_test, final_sgd.pred_test, title="Confusion matrix (SGD final, test)")
    plot_confusion(data.labels_test, final_gd.pred_test, title="Confusion matrix (Full-GD final, test)")

    X_ref = np.vstack([data.images_train, data.images_val])
    y_ref = np.concatenate([data.labels_train, data.labels_val])

    f_star_ref = reference_f_star_fixed_gd(
        X_ref, y_ref,
        l2=best_l2,
        lr=0.05,
        max_iters=50000
    )
    print("Reference f* (GD fixed-step):", f_star_ref)

    # Final comparison: objective + gap vs data passes
    plot_compare_objective_and_gap(
        final_sgd.hist.train_loss,
        final_gd.hist.train_loss,
        eval_every_sgd=eval_every,
        title_prefix="Final runs (train+val)",
        f_star=f_star_ref,
    )

    plt.show()


if __name__ == "__main__":
    main()