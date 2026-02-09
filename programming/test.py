from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
print(digits.data.shape)

print(digits.images[0])
plt.matshow(digits.images[0], cmap="gray")
plt.show()
