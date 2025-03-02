# Download the fashion-MNIST dataset and plotting 1 sample image for each class.

from keras.datasets import fashion_mnist
import wandb

wandb.init(project="Fashion-MNIST-Images",id="Question-1")

class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train / 255.0
X_test = X_test/ 255.0

def fashion_MNIST():
	set_images=[]
	set_labels=[]
	count=0
	for i in range(len(y_train)):
		if y_train[i]==count:
				set_images.append(X_train[i])
				set_labels.append(class_names[y_train[i]])
				count=count+1
		else:
				pass
		if count==10:
			break

	wandb.log({"Plot": [wandb.Image(img, caption=caption) for img, caption in zip(set_images, set_labels)]})
fashion_MNIST()


