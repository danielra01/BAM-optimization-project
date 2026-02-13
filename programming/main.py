from data import load_data
from model import init_model, scores, loss, predict
from sgd import SGDConfig, train_sgd, accuracy


def main():
    ImageData = load_data(train_size=0.7, random_state=0)
    model = init_model(seed=1)          # random weights biases zero
    S = scores(model, ImageData.images_train[:20])
    L = loss(S, ImageData.labels_train[:20])
    P = predict(model, ImageData.images_test[:20])

    print("scores shape:", S.shape)
    print("loss:", L)
    print("predictions:", P)
    print("true labels:", ImageData.labels_test[:20])

    cfg = SGDConfig(
        learning_rate=0.5,
        epochs=50,
        batch_size=64,
        l2_reg=1e-4,
        shuffle=True,
        seed=0,
    )

    history = train_sgd(model, ImageData.images_train,
                        ImageData.labels_train, cfg)

    pred_train = predict(model, ImageData.images_train)
    pred_test = predict(model, ImageData.images_test)

    print("Final train loss:", history.train_loss[-1])
    print("Train accuracy:", accuracy(pred_train, ImageData.labels_train))
    print("Test accuracy:", accuracy(pred_test, ImageData.labels_test))
    print("First 20 test predictions:", pred_test[:20])
    print("First 20 test labels:", ImageData.labels_test[:20])


if __name__ == "__main__":
    main()
