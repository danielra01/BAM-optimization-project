import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import time

from data import load_data
from model import init_model, predict
from sgd import SGDConfig, train_sgd, accuracy
from full_gd import GDConfig, train_full_gd


def main() -> None:
    data = load_data(train_size=0.7, random_state=2)

    # SGD training
    learning_rates = [0.05, 0.1, 0.2, 1.0, 2.0]
    epochs = 50
    batch_size = 64

    baseline_lr = 0.5
    reg_values = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    # Tune l2_reg with fixed baseline learning rate
    sgd_reg_runs = []

    for lam in reg_values:
        model_reg = init_model(seed=2)

        cfg_reg = SGDConfig(
            learning_rate=baseline_lr,
            epochs=epochs,
            batch_size=batch_size,
            l2_reg=lam,
            shuffle=True,
            seed=2,
        )

        t0 = time.perf_counter()
        hist_reg = train_sgd(model_reg, data.images_train, data.labels_train, cfg_reg)
        t1 = time.perf_counter()

        pred_test_reg = predict(model_reg, data.images_test)

        sgd_reg_runs.append({
            "l2": lam,
            "model": model_reg,
            "hist": hist_reg,
            "time": t1 - t0,
            "train_obj": float(hist_reg.train_loss[-1]),
            "test_acc": float(accuracy(pred_test_reg, data.labels_test)),
        })

        tag = "no-reg" if lam == 0.0 else f"l2={lam:g}"
        print(f"=== SGD (baseline lr={baseline_lr:g}, {tag}) ===")
        print("Final train objective:", sgd_reg_runs[-1]["train_obj"])
        print("Test accuracy:", sgd_reg_runs[-1]["test_acc"])
        print("Training time: {:.2f} seconds".format(sgd_reg_runs[-1]["time"]))
        print()

    # choose best l2_reg by test accuracy
    best_reg = max(sgd_reg_runs, key=lambda r: r["test_acc"])
    best_l2 = best_reg["l2"]

    print("=== Best l2_reg (with baseline lr) ===")
    print(f"Chosen l2_reg={best_l2:g} (baseline lr={baseline_lr:g})")
    print()

    # Plot: objective vs epochs for different l2_reg (baseline lr)
    plt.figure()
    for r in sgd_reg_runs:
        y = np.asarray(r["hist"].train_loss, dtype=float)
        x = np.arange(len(y))
        label = "no reg" if r["l2"] == 0.0 else f"l2={r['l2']:g}"
        plt.plot(x, y, label=label)
    plt.xlabel("Epochs")
    plt.ylabel("Objective value")
    plt.title(f"SGD: objective vs epochs, varying $\\ell_2$ (lr={baseline_lr:g})")
    plt.legend()

    # Plot: test accuracy vs l2_reg (baseline lr)
    plt.figure()
    xs = np.asarray([r["l2"] for r in sgd_reg_runs], dtype=float)
    ys = np.asarray([r["test_acc"] for r in sgd_reg_runs], dtype=float)
    xs_plot = xs.copy()
    xs_plot[xs_plot == 0.0] = 1e-18
    plt.semilogx(xs_plot, ys, marker="o")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Test accuracy")
    plt.title(f"SGD: test accuracy vs $\\lambda$ (lr={baseline_lr:g})")

    l2_reg = best_l2

    sgd_runs = []  # store dicts with results per lr

    for i, lr in enumerate(learning_rates):
        model_sgd = init_model(seed=2)

        sgd_cfg = SGDConfig(
            learning_rate=lr,
            epochs=epochs,
            batch_size=batch_size,
            l2_reg=l2_reg,
            shuffle=True,
            seed=2,
        )

        t0 = time.perf_counter()
        hist = train_sgd(model_sgd, data.images_train,
                         data.labels_train, sgd_cfg)
        t1 = time.perf_counter()

        pred_train = predict(model_sgd, data.images_train)
        pred_test = predict(model_sgd, data.images_test)

        run = {
            "lr": lr,
            "model": model_sgd,
            "hist": hist,
            "time": t1 - t0,
            "train_obj": float(hist.train_loss[-1]),
            "train_acc": float(accuracy(pred_train, data.labels_train)),
            "test_acc": float(accuracy(pred_test, data.labels_test)),
            "pred_test": pred_test,
        }
        sgd_runs.append(run)

        print(f"=== SGD (lr={lr:g}) ===")
        print("Final train objective:", run["train_obj"])
        print("Train accuracy:", run["train_acc"])
        print("Test accuracy:", run["test_acc"])
        print("SGD training time: {:.2f} seconds".format(run["time"]))
        print()

    # Best test accuracy
    best = max(sgd_runs, key=lambda r: r["test_acc"])

    print("=== Best SGD run ===")
    print(f"Chosen lr={best['lr']:g} (by test accuracy)")
    print("Classification report (SGD, test):")
    print(classification_report(data.labels_test, best["pred_test"]))

    # SGD learning-rate comparison (objective)
    plt.figure()
    for r in sgd_runs:
        y = np.asarray(r["hist"].train_loss, dtype=float)
        x = np.arange(len(y))
        plt.plot(x, y, label=f"SGD lr={r['lr']:g}")
    plt.xlabel("Epochs")
    plt.ylabel("Objective value")
    plt.title("SGD: objective vs epochs for different learning rates")
    plt.legend()

    # SGD learning-rate comparison (gap w.r.t. best SGD final value)
    plt.figure()
    sgd_best_final = float(np.min([float(np.min(np.asarray(
        r["hist"].train_loss, dtype=float)))
        for r in sgd_runs if len(r["hist"].train_loss) > 0]))
    for r in sgd_runs:
        sgd_obj = np.asarray(r["hist"].train_loss, dtype=float)
        gap = np.maximum(sgd_obj - sgd_best_final, 0.0)
        x = np.arange(len(gap))
        plt.plot(x, gap + 1e-16, label=f"SGD lr={r['lr']:g}")
        plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel(r"$f(x_k)-\min f$")
    plt.title("SGD: relative gap vs epochs (log scale)")
    plt.legend()

    # Full-batch GD training
    gd_learning_rates = [0.05, 0.1, 0.2, 0.5, 1.0]
    gd_iterations = 300

    gd_runs = []

    for lr in gd_learning_rates:
        model_gd = init_model(seed=2)

        gd_cfg = GDConfig(
            learning_rate=lr,
            iterations=gd_iterations,
            l2_reg=l2_reg,
        )

        t0 = time.perf_counter()
        hist = train_full_gd(model_gd, data.images_train, data.labels_train, gd_cfg)
        t1 = time.perf_counter()

        pred_train = predict(model_gd, data.images_train)
        pred_test = predict(model_gd, data.images_test)

        run = {
            "lr": lr,
            "model": model_gd,
            "hist": hist,
            "time": t1 - t0,
            "train_obj": float(hist.train_loss[-1]),
            "train_acc": float(accuracy(pred_train, data.labels_train)),
            "test_acc": float(accuracy(pred_test, data.labels_test)),
            "pred_test": pred_test,
        }
        gd_runs.append(run)

        print(f"=== Full-GD (lr={lr:g}) ===")
        print("Final train objective:", run["train_obj"])
        print("Train accuracy:", run["train_acc"])
        print("Test accuracy:", run["test_acc"])
        print("Full-GD training time: {:.2f} seconds".format(run["time"]))
        print()

    # Best GD run (by test accuracy)
    best_gd = max(gd_runs, key=lambda r: r["test_acc"])

    print("=== Best Full-GD run ===")
    print(f"Chosen lr={best_gd['lr']:g} (by test accuracy)")
    print("Classification report (Full-GD best, test):")
    print(classification_report(data.labels_test, best_gd["pred_test"]))

    # Collect objectives
    sgd_objs = [(r["lr"], np.asarray(r["hist"].train_loss, dtype=float)) for r in sgd_runs]
    gd_objs = [(r["lr"], np.asarray(r["hist"].train_loss, dtype=float)) for r in gd_runs]

    # Reference value: best objective reached among all Full-GD runs
    f_star = float(np.min([np.min(obj) for _, obj in gd_objs]))

    # Confusion matrix (SGD final model)
    ConfusionMatrixDisplay.from_predictions(data.labels_test,
                                            best["pred_test"])
    plt.title(f"Confusion matrix (SGD best lr={best['lr']:g})")

    # Confusion matrix (FGD final model)
    ConfusionMatrixDisplay.from_predictions(data.labels_test,
                                            best_gd["pred_test"])
    plt.title(f"Confusion matrix (Full-GD best lr={best_gd['lr']:g})")

    # Objective vs data passes (epochs)
    plt.figure()
    for lr, obj in gd_objs:
        x = np.arange(len(obj))
        plt.plot(x, obj, label=f"Full-GD lr={lr:g}")

    for lr, obj in sgd_objs:
        x = np.arange(len(obj))
        plt.plot(x, obj, label=f"SGD lr={lr:g}")
    plt.xlabel("Data passes (epochs)")
    plt.ylabel("Objective value")
    plt.title("Objective value vs data passes")
    plt.legend()

    # Optimality gap vs data passes (epochs) on a log scale
    plt.figure()
    for lr, obj in gd_objs:
        gap = np.maximum(obj - f_star, 0.0)
        x = np.arange(len(gap))
        plt.semilogy(x, gap + 1e-16, label=f"Full-GD lr={lr:g}")
    for lr, obj in sgd_objs:
        gap = np.maximum(obj - f_star, 0.0)
        x = np.arange(len(gap))
        plt.semilogy(x, gap + 1e-16, label=f"SGD lr={lr:g}")
    plt.xlabel("Data passes (epochs)")
    plt.ylabel(r"$f(x_k) - f^\star$")
    plt.title("Optimality gap vs data passes (log scale)")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
