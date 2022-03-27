
import os
import seaborn as sns
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from os.path import join
from torch.utils.data import Dataset
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, classification_report



BASE = "./Dataset"
# BASE = "gdrive/MyDrive/Dataset"
cloth_folder = join(BASE, "Cloth Mask")
n95_folder = join(BASE, "N95")
n95valve_folder = join(BASE, "N95 mask with valve")
surgical_folder = join(BASE, "Surgical Mask")
without_mask_folder = join(BASE, "No Mask")
root_folder = ("./")

# predefined classes
classes = {
    "cloth": 0,
    "n95": 1,
    "n95valve": 2,
    "surgical": 3,
    "without_mask": 4
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


class FaceMaskDataset(Dataset):
    dataset = []
    conversion = None

    def __init__(self, images, indexes, conversion=transforms.ToTensor()):
        self.conversion = conversion
        self.dataset = images[int(indexes[0]):int(indexes[1])]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]['image']
        if self.conversion is not None:
            image = self.conversion(image)
        return image, self.dataset[index]['target']



class FaceMaskClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}".format(epoch + 1, result['train_loss']))


class CNN(FaceMaskClassificationBase):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(246016, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        x = self.conv_layer(x)  # convolution layer
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layer(x)  # fully connected layer
        return x



def fit(epochs, lr, model, train_loader, optimizer_func):
    results = []
    optimizer = optimizer_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
            loss = model.training_step(batch)
            # update-training-loss
            train_losses.append(loss)
            # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
            loss.backward()
            # perform-a-single-optimization-step (parameter-update)
            optimizer.step()
            # clear-the-gradients-of-all-optimized-variables
            optimizer.zero_grad()

        result = {'train_loss': torch.stack(train_losses).mean().item()}
        model.epoch_end(epoch, result)
        results.append(result)

    return results


# Load images from the dataset
def load_images():
    dat = []

    def helper(cls, folder):
        for filename in os.listdir(folder):
            try:
                sample = {}
                img = Image.open(join(folder, filename)).convert('RGB')
                sample['image'] = img
                sample['target'] = classes[cls]
                dat.append(sample)
            except:
                continue

    helper("cloth", cloth_folder)
    helper("n95",n95_folder)
    helper("n95valve",n95valve_folder)
    helper("surgical",surgical_folder)
    helper("without_mask",without_mask_folder)

    return dat

# Plot the losses in each epoch
def plot_losses(results):
    train_losses = [result.get('train_loss') for result in results]
    plt.plot(train_losses, '-bx')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training Loss'])
    plt.title('Loss vs. No. of epochs');


# show confusion matrix
def show_confusion_matrix(y_true, y_preds):
    matrix = confusion_matrix(y_true, y_preds)
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(matrix, fmt='', annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix with labels!!');
    ax.set_xlabel('Predicted Mask Type')
    ax.set_ylabel('Actual Mask Type')
    ax.xaxis.set_ticklabels([i for i in classes.keys()])
    ax.yaxis.set_ticklabels([i for i in classes.keys()])
    plt.show()

if __name__ == "__main__":


    # LOADING AND SPLITTING DATASET
    train_split_percentage = 0.75
    test_split_percentage = 0.25

    # Load images into memory
    images = load_images()
    random.shuffle(images)
    size_of_the_dataset = len(images)
    print("size_of_the_dataset", size_of_the_dataset)

    num_of_classes = len(classes.keys())

    train_indexes = [0, train_split_percentage * size_of_the_dataset]
    test_indexes = [train_split_percentage * size_of_the_dataset, size_of_the_dataset]

    print("Effective train split = {} to {}".format(train_indexes[0], train_indexes[1]) )
    print("Effective test split = {} to {}".format(test_indexes[0],test_indexes[1]))


    transform = transforms.Compose(
        [transforms.Resize((250, 250)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    print("Loading training and validation set")
    train_dataset = FaceMaskDataset(images, train_indexes, transform)
    print("Train Dataset length ", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=25, shuffle=True, num_workers=0)


    # COMPILE AND TEST MODEL
    print("Training our Model")
    model = CNN()

    results = fit(2, 0.01, model, train_loader, torch.optim.ASGD)

    # Save model
    print("Saving model to ", join(BASE, "model.pth"))
    torch.save(model.state_dict(), join(BASE, "model.pth"))

    # Plotting loss vs epoch chart
    plot_losses(results)


    # Testing phase
    test_dataset = FaceMaskDataset(images, test_indexes, conversion=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=25, shuffle=True, num_workers=0)

    y_true = torch.tensor([])
    y_preds = torch.tensor([])

    # Test the model
    model.eval()  # turn off layers like Dropout layers, BatchNorm Layers
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            y_true = torch.cat((y_true, labels))

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_preds = torch.cat((y_preds, predicted))

        print('Test Accuracy of the model: {} %'.format(100 * correct / total))


    # Display Confusion matrix and other important parameters
    show_confusion_matrix(y_true, y_preds)

    precision, recall, fscore, support = score(y_true, y_preds)
    print(classification_report(y_true, y_preds))
