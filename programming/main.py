import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from data import load_data
from model import init_model, predict
from sgd import SGDConfig, train_sgd, accuracy
from full_gd import GDConfig, train_full_gd


def main() -> None:
    data = load_data(train_size=0.7, random_state=2)

    # Keep regularization consistent to compare
    l2_reg = 1e-4

    # SGD training
    model_sgd = init_model(seed=2)
    sgd_cfg = SGDConfig(
        learning_rate=0.5,
        epochs=50,
        batch_size=64,
        l2_reg=l2_reg,
        shuffle=True,
        seed=2,
    )
    sgd_hist = train_sgd(model_sgd, data.images_train, data.labels_train,
                         sgd_cfg)

    pred_train_sgd = predict(model_sgd, data.images_train)
    pred_test_sgd = predict(model_sgd, data.images_test)

    print("=== SGD ===")
    print("Final train objective:", sgd_hist.train_loss[-1])
    print("Train accuracy:", accuracy(pred_train_sgd, data.labels_train))
    print("Test accuracy:", accuracy(pred_test_sgd, data.labels_test))
    print()
    print("Classification report (SGD, test):")
    print(classification_report(data.labels_test, pred_test_sgd))

    # Full-batch GD training
    model_gd = init_model(seed=2)
    gd_cfg = GDConfig(
        learning_rate=0.5,
        iterations=300,
        l2_reg=l2_reg,
    )
    gd_hist = train_full_gd(model_gd, data.images_train, data.labels_train,
                            gd_cfg)

    pred_train_gd = predict(model_gd, data.images_train)
    pred_test_gd = predict(model_gd, data.images_test)

    print("=== Full-batch GD (reference) ===")
    print("Final train objective:", gd_hist.train_loss[-1])
    print("Train accuracy:", accuracy(pred_train_gd, data.labels_train))
    print("Test accuracy:", accuracy(pred_test_gd, data.labels_test))
    print()

    # Plots
    sgd_obj = np.asarray(sgd_hist.train_loss,
                         dtype=float)  # objective per epoch
    gd_obj = np.asarray(gd_hist.train_loss,
                        dtype=float)   # objective per iteration

    f_star = float(np.min(gd_obj))  # "Optimal" reference

    sgd_gap = np.maximum(sgd_obj - f_star, 0.0)
    gd_gap = np.maximum(gd_obj - f_star, 0.0)

    # Confusion matrix (SGD final model)
    ConfusionMatrixDisplay.from_predictions(data.labels_test, pred_test_sgd)
    plt.title("Confusion matrix (SGD final model)")

    # Confusion matrix (FGD final model)
    ConfusionMatrixDisplay.from_predictions(data.labels_test, pred_test_gd)
    plt.title("Confusion matrix (FGD final model)")

    # Objective vs data passes (epochs)
    x_sgd = np.arange(1, len(sgd_obj) + 1)
    x_gd = np.arange(1, len(gd_obj) + 1)
    plt.figure()
    plt.plot(x_sgd, sgd_obj, label="SGD (per pass)")
    plt.plot(x_gd, gd_obj, label="Full-GD (per pass)")
    plt.xlabel("Data passes (epochs)")
    plt.ylabel("Objective value")
    plt.title("Objective value vs data passes (epochs)")
    plt.legend()


    # Optimality gap vs data passes (epochs) on a log scale
    x_sgd = np.arange(1, len(sgd_gap) + 1)
    x_gd = np.arange(1, len(gd_gap) + 1)
    plt.figure()
    plt.semilogy(x_sgd, sgd_gap + 1e-16, label="SGD gap (per pass)")
    plt.semilogy(x_gd, gd_gap + 1e-16, label="Full-GD gap (per pass)")
    plt.xlabel("Data passes (epochs)")
    plt.ylabel(r"$f(x_k) - f(x^\star)$")
    plt.title("Optimality gap vs data passes (epochs)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
