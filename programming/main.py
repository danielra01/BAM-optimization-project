from data import load_data
from model import init_model, scores, loss, predict


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


if __name__ == "__main__":
    main()
