from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
print(digits.data.shape)

plt.matshow(digits.images[0], cmap="gray")
plt.show()
