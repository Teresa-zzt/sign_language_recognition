import random
import warnings
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import os
from tqdm import tqdm
from PIL import Image  # Import the Image module from PIL

warnings.filterwarnings("ignore")
# set seed for reproducibility
seed = 3407
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

class MnasNetA1(nn.Module):
    def __init__(self, numClasses):
        # Inputs : numClasses - number of classes in the dataset
        super(MnasNetA1, self).__init__()
        self.model = torchvision.models.mnasnet1_0(pretrained=True)  # Load the MnasNet model
        num_ftrs = self.model.classifier[-1].in_features  # Get the number of input features to the final classifier
        self.model.classifier[-1] = nn.Linear(num_ftrs, numClasses)  # Replace the final classifier layer

    # Define the forward pass
    def forward(self, x):
        return self.model(x)
    
class MobileNetV2(nn.Module):
    def __init__(self, numClasses):
        # Inputs : numClasses - number of classes in the dataset
        super(MobileNetV2, self).__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True) # Load the MobileNet model
        num_ftrs = self.model.classifier[1].in_features # Get the number of input features to the final classifier
        self.model.classifier[1] = nn.Linear(num_ftrs, numClasses) # Replace the final classifier layer

    # Define the forward pass
    def forward(self, x):
        return self.model(x)
    

class CustomModel(nn.Module):
    def __init__(self, numClasses):
        # Inputs : numClasses - number of classes in the dataset
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7,
                             1000)  # Adjust the dimension according to your input image size
        self.fc2 = nn.Linear(1000, numClasses)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class Trainer:
# Create a trainer class for model training
# Define the transformations for the images
    transforms = [
        # transform for the Mnasenet  model
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        # transform for the mobilenetv2 modol, Param 4.38M
       transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),

        # transform for the custom model
        transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])       
    ]
# Define the models for training
    models = {
        'MnasNet': MnasNetA1,
        'MobileNet': MobileNetV2,
         'Custom': CustomModel
     }
    modelNames = ['MnasNet', 'MobileNet', 'Custom']
# Define the labels for the dataset
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S','T',
            'U', 'V', 'W', 'X', 'Y', 
            '9', '0', '7', '6', '1', '8', '4', '3', '2', '5']

    # Define the initialization method
    def __init__(self,model_name, root_dir='./data_images', signals=None, batch_size = 64, lr=0.0001, num_epochs=3, testSize=0.2):
        # Inputs : model_name = name of the model to train
        #          root_dir = directory containing the dataset
        #          signals = signal object to communicate with the GUI
        #          batch_size = batch size for training
        #          lr = learning rate for training
        #          num_epochs = number of epochs for training
        #          testSize = size of the test set

        # Check if the model name is in the models dictionary
        assert model_name in self.models, f'Model {model_name} not found.'
        self.modelName = model_name
        self.rootDir = root_dir
        self.batchSize = batch_size
        self.signals = signals
        self.lr = lr
        self.numEpochs = num_epochs
        self.lossHistory = []
        self.accHistory = []

        # filename prefix for saving the model
        self.savePrefix = f'{model_name}_bs{batch_size}_epochs{num_epochs}'

        # Add a stop flag to stop the training process
        self.stopRequested = False  # Stop flag

        self.transform = self.transforms[self.modelNames.index(model_name)]

        # Create the training and testing datasets
        dataset = ImageFolder(root=self.rootDir, transform=self.transform)
        trainSize = int((1 - testSize) * len(dataset))
        testSize = len(dataset) - trainSize
        self.train_dataset, self.test_dataset = random_split(dataset, [trainSize, testSize])

    # Create data loaders
        self.trainLoader = DataLoader(self.train_dataset, batch_size=self.batchSize, shuffle=True, num_workers=4)
        self.testLoader = DataLoader(self.test_dataset, batch_size=self.batchSize, shuffle=False, num_workers=4)

    # Initialize the model
        self.model = self.models[model_name](numClasses=34)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    # Define device (CPU or CUDA)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to the selected device
        self.model.to(self.device)

    def train(self):
        for epoch in range(self.numEpochs):
            self.model.train()
            runningLoss = 0.0
            if self.stopRequested:
                print("Training stopped")
                break


            for images, labels in tqdm(self.trainLoader,leave=not self.stopRequested):
                # Check if the stop flag is set
                if self.stopRequested:
                    print("Second if")
                    break
                else:
                    # Move data to device
                    images, labels = images.to(self.device), labels.to(self.device)               
                    self.optimizer.zero_grad()
                    # Forward pass
                    outputs = self.model(images)
                    # Calculate loss
                    loss=self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    runningLoss += loss.item()    
         
            loss = runningLoss / len(self.trainLoader)
            acc=self.eval()

            
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
            self.lossHistory.append(loss)
            self.accHistory.append(acc)
            #plot the loss and accuracy
            self.ploting()
            # update graph after every epoch
            if not self.stopRequested:
                self.signals.showGraphSignal.emit()
            # if no stop requested, save the model
        if not self.stopRequested:
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            # Save the model
            torch.save(self.model.state_dict(),f'./checkpoints/{self.savePrefix}.pt')
            print(f"Model saved as {self.savePrefix}.pt")
        print("end of train function")
            
    def ploting(self):
        if not os.path.exists('./plots'):
            os.makedirs('./plots')
             # Save loss history in a 1x2 figure
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Validation accuracy')
        plt.plot(self.accHistory)
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')

        plt.subplot(1, 2, 2)
        plt.title('Training loss')
        plt.plot(self.lossHistory)
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(f'./plots/{self.savePrefix}.png')
        print(f"Loss/Accuracy history saved as {self.savePrefix}.png")   

           
    def eval(self):
         # Testing loop
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(self.testLoader, leave=not self.stopRequested):
                #Check if the stop flag is set
                if self.stopRequested:
                    print("Evaluation stopped.")
                    break
                else:
                # Move data to device
                    images, labels = images.to(self.device), labels.to(self.device)
                        
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        if total == 0:
            print("No test data")
            return True
        else:
            acc=(correct / total) * 100
            print(f"Accuracy on test set: {(correct / total) * 100}%")
            return acc
    
    # Inference function for predict the result from the model
    def inference(self, modelName,image) -> int:
        # Inputs : modelName = name of the model to load

        # Load the current model
        self.model.load_state_dict(
            torch.load(f'./checkpoints/{modelName}')
        )
        # Set the model to evaluation mode
        self.model.eval()
        # Load the image
        image = self.transform(image).unsqueeze(0).to(self.device)
        # Make a prediction
        with torch.no_grad():
            output = self.model(image)
            # Get the class probabilities
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            sorted_prob, indices = torch.sort(probabilities, descending=True)
            # Print the top 5 possible classes
            top_k = 5 
            # top_k = len(probabilities)
            # need to save all the printed string inside a list, than return the list
            list = []
            for i in range(top_k):
                list.append(f'{self.labels [indices[i].item()]}: {sorted_prob[i].item() * 100:.2f}% confidence')
            return list
        
    def request_stop(self):
        self.stopRequested = True

# uncomment to run this file
if __name__ == '__main__':
    train = Trainer('MnasNet', './data_images')
    # train.train()
    image = Image.open('test_capture_image/capture_2.png').convert('RGB')
    model_path='MnasNet_bs16_epochs5.pt'
    list=train.inference(model_path,image)
    print(list)

