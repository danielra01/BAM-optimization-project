from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from data import load_data
from model import init_model, predict

ImageData = load_data(train_size=0.7, random_state=200)
model = init_model(seed=100)          # random weights biases zero
P = predict(model, ImageData.images_test[:])

# I found those two functions in the documentation of sklearn, they seem
# to be very Useful for evaluating the performance of our model
print(classification_report(ImageData.labels_test[:], P))
ConfusionMatrixDisplay.from_predictions(ImageData.labels_test[:], P)
plt.show()
