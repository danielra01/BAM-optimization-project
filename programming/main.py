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

    # Keep regularization consistent to compare
    l2_reg = 1e-4

    # SGD training
    learning_rates = [0.05, 0.1, 0.2, 0.5, 1.0]
    epochs = 50
    batch_size = 64

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
    model_gd = init_model(seed=2)
    gd_cfg = GDConfig(
        learning_rate=0.5,
        iterations=300,
        l2_reg=l2_reg,
    )
    gd_t0 = time.perf_counter()
    gd_hist = train_full_gd(model_gd, data.images_train, data.labels_train,
                            gd_cfg)
    gd_t1 = time.perf_counter()

    pred_train_gd = predict(model_gd, data.images_train)
    pred_test_gd = predict(model_gd, data.images_test)

    print("=== Full-batch GD (reference) ===")
    print("Final train objective:", gd_hist.train_loss[-1])
    print("Train accuracy:", accuracy(pred_train_gd, data.labels_train))
    print("Test accuracy:", accuracy(pred_test_gd, data.labels_test))
    print()
    print("Classification report (Full-batch GD, test):")
    print(classification_report(data.labels_test, pred_test_gd))
    print("Full-batch GD training time: {:.2f} seconds".format(gd_t1 - gd_t0))

    # objective per iteration
    gd_obj = np.asarray(gd_hist.train_loss, dtype=float)
    # reference best value
    f_star = float(np.min(gd_obj))
    gd_gap = np.maximum(gd_obj - f_star, 0.0)

    # Confusion matrix (SGD final model)
    ConfusionMatrixDisplay.from_predictions(data.labels_test,
                                            best["pred_test"])
    plt.title(f"Confusion matrix (SGD best lr={best['lr']:g})")

    # Confusion matrix (FGD final model)
    ConfusionMatrixDisplay.from_predictions(data.labels_test, pred_test_gd)
    plt.title("Confusion matrix (FGD final model)")

    # Objective vs data passes (epochs)
    plt.figure()
    x_gd = np.arange(len(gd_obj))
    plt.plot(x_gd, gd_obj, label="Full-GD (reference)")
    for r in sgd_runs:
        sgd_obj = np.asarray(r["hist"].train_loss, dtype=float)
        x = np.arange(len(sgd_obj))
        plt.plot(x, sgd_obj, label=f"SGD lr={r['lr']:g}")
    plt.xlabel("Data passes (epochs)")
    plt.ylabel("Objective value")
    plt.title("Objective value vs data passes")
    plt.legend()

    # Optimality gap vs data passes (epochs) on a log scale
    plt.figure()
    plt.semilogy(x_gd, gd_gap + 1e-16, label="Full-GD gap (reference)")
    for r in sgd_runs:
        sgd_obj = np.asarray(r["hist"].train_loss, dtype=float)
        sgd_gap = np.maximum(sgd_obj - f_star, 0.0)
        x = np.arange(len(sgd_gap))
        plt.semilogy(x, sgd_gap + 1e-16, label=f"SGD lr={r['lr']:g}")
    plt.xlabel("Data passes (epochs)")
    plt.ylabel(r"$f(x_k) - f(x^\star)$")
    plt.title("Optimality gap vs data passes (log scale)")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
