import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


from train import NeuralNetwork


# Load the datasets  
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define dataset sizes
total_train_samples = train_labels.shape[0]
train_data = int(0.9 * total_train_samples)  # 90% for training
validation_size = total_train_samples - train_data  # 10% for validation
test_size = test_labels.shape[0]

shuffle_indices  = np.random.choice(train_labels.shape[0], total_train_samples, replace=False)
shuffle_indices_2 = np.random.choice(test_labels.shape[0], test_size, replace=False)

total_train_images = train_images[shuffle_indices, :]
total_train_labels = train_labels[shuffle_indices]

X_train = total_train_images[:train_data,:]
Y_train = total_train_labels[:train_data]

X_valid = total_train_images[train_data:, :]
Y_valid = total_train_labels[train_data:]    

X_test = test_images[shuffle_indices_2, :]
Y_test = test_labels[shuffle_indices_2]

sweep_config = {
  "name": "Bayes_Hyperparam_Tuning",
  "method": "bayes",
  "metric":{
  "name": "validation_accuracy",
  "goal": "maximize"
  },
  "parameters": {
        "max_epochs": {
            "values": [5, 10]
        },

        "initializer": {
            "values": ["RANDOM", "XAVIER"]
        },

        "layers": {
            "values": [2, 3, 4]
        },
        
        
        "hidden_neurons": {
            "values": [32, 64, 128]
        },
        
        "activation_func": {
            "values": [ 'TANH',  'SIGMOID', 'RELU']
        },
        
        "learning_rate": {
            "values": [0.001, 0.0001]
        },
        
        
        "weight_decay": {
            "values": [0, 0.0005,0.5]
        },
        
        "optimizer": {
            "values": ["SGD", "MGD", "NGD", "RMSPROP", "ADAM"]
        },
                    
        "batch_size": {
            "values": [16, 32, 64]
        }
        
        
    }
}


def train_and_test():    
    config_best = dict(
            max_epochs=5,
            layers=3,
            hidden_neurons=128,
            weight_decay=0.0005,
            learning_rate=1e-4,
            optimizer="ADAM",
            batch_size=64,
            activation="TANH",
            initializer="XAVIER",
            loss_function="CROSS",
        )

   
    wandb.init(project='Fashion_MNIST_Images', entity='singhsonalika5-indian-institute-of-technology-madras', config = config_best)
    

    wandb.run.name = "hl_" + str(wandb.config.layers) + "_hn_" + str(wandb.config.hidden_neurons) + "_opt_" + wandb.config.optimizer + "_act_" + wandb.config.activation + "_lr_" + str(wandb.config.learning_rate) + "_bs_"+str(wandb.config.batch_size) + "_init_" + wandb.config.initializer + "_ep_"+ str(wandb.config.max_epochs)+ "_l2_" + str(wandb.config.weight_decay) 
    CONFIG = wandb.config

  
    NN = NeuralNetwork(
        layers=CONFIG.layers,                hidden_neurons=CONFIG.hidden_neurons,
        train_images=train_images,           train_labels=train_labels,
        val_images=X_valid,                  val_labels=Y_valid,
        test_images=X_test,                  test_labels=Y_test,
        num_test=test_size,                  num_train=train_data,            
        optimizer=CONFIG.optimizer,          batch_size=CONFIG.batch_size,
        weight_decay=CONFIG.weight_decay,    learning_rate=CONFIG.learning_rate,
        max_epochs=CONFIG.max_epochs,        activation=CONFIG.activation,  
        initializer=CONFIG.initializer,      loss_function=CONFIG.loss_function,
        num_val=validation_size
    )

    training_loss, training_accuracy, validation_accuracy, predicted_labels = NN.optimizer(
        NN.max_epochs, NN.num_train, NN.batch_size, NN.learning_rate)
    


    Y_pred_test =  NN.predict(NN.X_test, NN.N_test)
    train_accuracy, Y_true_train, Y_pred_train = NN.accuracy(NN.Y_train, Y_pred_train, NN.N_train)
    test_accuracy, Y_true_test, Y_pred_test = NN.accuracy(NN.Y_test, Y_pred_test,NN.N_test)
    train_pred = (train_accuracy, Y_true_train, Y_pred_train)
    test_pred = (test_accuracy, Y_true_test, Y_pred_test)

    return train_pred, test_pred

    
if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    Results = {}
    Results["train_pred_best"], Results["test_pred_best"] = train_and_test()
    
    data = [[label, val] for (label, val) in zip(["test_pred_best"],[Results['test_pred_best'][0]])] 
    table = wandb.Table(data=data, columns = ["Configuration", "Test accuracy"])
   
    
    
    wandb.init(project='Fashion_MNIST_Images', entity='singhsonalika5-indian-institute-of-technology-madras', name="Train_Confusion_Matrix")
    wandb.sklearn.plot_confusion_matrix(Results["train_pred_best"][1], Results["train_pred_best"][2], labels =[0,1,2,3,4,5,6,7,8,9])    
  
    
    wandb.init( project='Fashion_MNIST_Images', entity='singhsonalika5-indian-institute-of-technology-madras', name="Test_Confusion_Matrix") 
    wandb.sklearn.plot_confusion_matrix(Results["test_pred_best"][1], Results["test_pred_best"][2], labels =[0,1,2,3,4,5,6,7,8,9])

    wandb.log({"my_bar_chart_id" : wandb.plot.bar(table, "Configuration", "Test accuracy",title="Best Configuration Test Accuracy for Fashion MNIST Classification")})
    wandb.finish()
    