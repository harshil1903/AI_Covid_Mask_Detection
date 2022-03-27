
import os
import seaborn as sns
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from os.path import isfile, join
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


# Load images from the dataset
def load_images():
    dat = []

    def helper(folder):
        for filename in os.listdir(folder):
            try:
                sample = {}
                img = Image.open(join(folder, filename)).convert('RGB')
                sample['image'] = img
                sample['target'] = classes['cloth']
                dat.append(sample)
            except:
                continue

    helper(cloth_folder)
    helper(n95_folder)
    helper(n95valve_folder)
    helper(surgical_folder)
    helper(without_mask_folder)

    return dat


class FaceMaskClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'validation_loss': loss.detach(), 'validation_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['validation_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['validation_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'validation_loss': epoch_loss.item(), 'validation_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}"
              .format(epoch + 1, result['train_loss'], result['validation_loss'], result['validation_acc']))


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
            nn.Linear(512, num_of_classes),
        )

    def forward(self, x):
        x = self.conv_layer(x)  # convolution layer
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layer(x)  # fully connected layer
        return x


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    acc = torch.sum(preds == labels).item() / len(preds)
    return torch.tensor(acc)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, optimizer_func):
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

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        results.append(result)

    return results



# LOADING AND SPLITTING DATASET

train_split_percentage = 0.75
validation_split_percentage = 0.15
test_split_percentage = 0.1

# load images into memory
images = load_images()
random.shuffle(images)
size_of_the_dataset = len(images)
print("size_of_the_dataset", size_of_the_dataset)

num_of_classes = len(classes.keys())

train_indexes = [0, train_split_percentage * size_of_the_dataset]
validation_indexes = [train_split_percentage * size_of_the_dataset, (train_split_percentage + validation_split_percentage) * size_of_the_dataset]
test_indexes = [(train_split_percentage + validation_split_percentage) * size_of_the_dataset, size_of_the_dataset]

print(f"Effective train split = {train_indexes[0]} to {train_indexes[1]}")
print(f"Effective val split = {validation_indexes[0]} to {validation_indexes[1]}")
print(f"Effective test split = {test_indexes[0]} to {test_indexes[1]}")


transform = transforms.Compose(
    [transforms.Resize((250, 250)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


print("Loading training and validation set")
train_dataset = FaceMaskDataset(images, train_indexes, transform)
validation_dataset = FaceMaskDataset(images, validation_indexes, transform)

print("Train Dataset length ", len(train_dataset))
print("Validation Dataset length ", len(validation_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=25, shuffle=True, num_workers=0)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=25, shuffle=False,num_workers=0)


#COMPILE AND TEST MODEL
model = CNN()
model = model.to(device)

results = fit(2, 0.01, model, train_loader, validation_loader, torch.optim.ASGD)

#save model
model_path = os.path.join(root_folder, 'model/AI.pth')
torch.save(model.state_dict(), os.path.join(model_path, "model.pth"))

import matplotlib.pyplot as plt


# Plot the history of accuracies
def plot_accuracies(results):
    accuracies = [result['validation_acc'] for result in results]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');

plot_accuracies(results)


# Plot the losses in each epoch
def plot_losses(results):
    train_losses = [result.get('train_loss') for result in results]
    val_losses = [result['validation_loss'] for result in results]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');


plot_losses(results)

#Testing phase
test_dataset = FaceMaskDataset(images, test_indexes, conversion=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=25, shuffle=True, num_workers=0)


y_true = torch.tensor([])
y_true = y_true.to(device)
y_preds = torch.tensor([])
y_preds = y_preds.to(device)

# test-the-model
model.eval()  # turn off layers like Dropout layers, BatchNorm Layers
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        y_true = torch.cat((y_true, labels))

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_preds = torch.cat((y_preds, predicted))

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))

y_true = y_true.to('cpu')
y_preds = y_preds.to('cpu')



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


show_confusion_matrix(y_true, y_preds)


precision, recall, fscore, support = score(y_true, y_preds)
print(classification_report(y_true, y_preds))


