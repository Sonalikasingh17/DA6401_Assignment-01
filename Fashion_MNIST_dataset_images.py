"""Download the fashion-MNIST dataset and plot 1 sample image for each class."""

from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Plot one sample per class
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(X_train[y_train == i][0], cmap='gray')
    ax.set_title(class_names[i])
    ax.axis("off")
plt.show()

# Save the first 10 images
for i in range(10):
    plt.imsave(f"sample_{i}.png", X_train[y_train == i][0], cmap='gray')
    

