"""Download the fashion-MNIST dataset and plotting 1 sample image for each class."""

from keras.datasets import fashion_mnist
from wandb.keras import wandbCallback
import wandb
# import matplotlib.pyplot as plt


wandb.init(project="fashion-mnist",id = "Question 01",entity="wandb")

# Define the class names
class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# # Plot one sample per class
# fig, axes = plt.subplots(2, 5, figsize=(10, 5))


# for i in range(10):
#     ax = axes[i // 5, i % 5]
#     ax.imshow(X_train[y_train == i][0], cmap='gray')
#     ax.set_title(class_names[i])
#     ax.axis("off")
# plt.show()

# # Save the first 10 images
# for i in range(10):
#     plt.imsave(f"sample_{i}.png", X_train[y_train == i][0], cmap='gray')
    
trainX=trainX / 255.0
testX=testX / 255.0

def log_images():
	set_images=[]
	set_labels=[]
	count=0
	for d in range(len(trainy)):
		if trainy[d]==count:
				set_images.append(trainX[d])
				set_labels.append(class_names[trainy[d]])
				count=count+1
		else:
				pass
		if count==10:
			break

	wandb.log({"Plot": [wandb.Image(img, caption=caption) for img, caption in zip(set_images, set_labels)]})
log_images()
