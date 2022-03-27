
# from google.colab import drive
# drive.mount('/content/gdrive')


import pandas as pd
import os
from os.path import isfile, join
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms

BASE = "./Dataset"
#BASE = "gdrive/MyDrive/Dataset"
cloth_folder = os.path.join(BASE, "Cloth Mask")
n95_folder = os.path.join(BASE, "N95")
n95valve_folder = os.path.join(BASE, "N95 mask with valve")
surgical_folder = os.path.join(BASE, "Surgical Mask")
without_mask_folder = os.path.join(BASE, "No Mask")
root_folder = (".")

# create directory
def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


model_path = os.path.join(root_folder, 'model')
make_dir(model_path)

# predefined classes
classes = {
    "cloth": 0,
    "n95": 1,
    "n95valve": 2,
    "surgical": 3,
    "without_mask": 4
}


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



def load_images():
    dat = []
    cnt = 0
    for filename in os.listdir(cloth_folder):
        sample = {}
        img = Image.open(os.path.join(cloth_folder, filename)).convert('RGB')
        sample['image'] = img
        sample['target'] = classes['cloth']
        dat.append(sample)
        cnt += 1
        if cnt % 50 == 0:
            print(cnt)

    for filename in os.listdir(n95_folder):
        sample = {}
        img = Image.open(os.path.join(n95_folder, filename)).convert('RGB')
        sample['image'] = img
        sample['target'] = classes['n95']
        dat.append(sample)
        cnt += 1
        if cnt % 50 == 0:
            print(cnt)

    for filename in os.listdir(n95valve_folder):
        try:
            sample = {}
            img = Image.open(os.path.join(n95valve_folder, filename)).convert('RGB')
            sample['image'] = img
            sample['target'] = classes['n95valve']
            dat.append(sample)
            cnt += 1
            if cnt % 50 == 0:
                print(cnt)
        except:
            continue

    for filename in os.listdir(surgical_folder):
        try:
            sample = {}
            img = Image.open(os.path.join(surgical_folder, filename)).convert('RGB')
            sample['image'] = img
            sample['target'] = classes['surgical']
            dat.append(sample)
            cnt += 1
            if cnt % 50 == 0:
                print(cnt)
        except:
            continue

    for filename in os.listdir(without_mask_folder):
        try:
            sample = {}
            img = Image.open(os.path.join(without_mask_folder, filename)).convert('RGB')
            sample['image'] = img
            sample['target'] = classes['without_mask']
            dat.append(sample)
            cnt += 1
            if cnt % 50 == 0:
                print(cnt)
        except:
            continue

    return dat


# %%


train_split_percentage = 0.75
val_split_percentage = 0.15
test_split_percentage = 0.1

# load images into memory
images = load_images()
random.shuffle(images)
size_of_the_dataset = len(images)
print("size_of_the_dataset", size_of_the_dataset)

batch_size = 25
num_of_classes = len(classes.keys())


train_indexes = [0, train_split_percentage * size_of_the_dataset]
val_indexes = [train_split_percentage * size_of_the_dataset,
               (train_split_percentage + val_split_percentage) * size_of_the_dataset]
test_indexes = [(train_split_percentage + val_split_percentage) * size_of_the_dataset, size_of_the_dataset]

print(f"Effective train split = {train_indexes[0]} to {train_indexes[1]}")
print(f"Effective val split = {val_indexes[0]} to {val_indexes[1]}")
print(f"Effective test split = {test_indexes[0]} to {test_indexes[1]}")



transform = transforms.Compose(
    [transforms.Resize((250, 250)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])


# 0 for train, 1 for validation, 2 for test
print("Loading training set")
train_dataset = FaceMaskDataset(images, train_indexes, conversion=transform)

print("Loading validation set")
val_dataset = FaceMaskDataset(images, val_indexes, conversion=transform)

print("Train Dataset length ", len(train_dataset))
print("Validation Dataset length ", len(val_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


import torch.nn as nn
import torch.nn.functional as F


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
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch + 1, result['train_loss'], result['val_loss'], result['val_acc']))



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
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x



def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            images, labels = batch
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
        history.append(result)

    return history




model = CNN()
model = model.to(device)


torch.cuda.empty_cache()


history = fit(25, 0.0001, model, train_loader, val_loader, torch.optim.Adam)

# torch.save(model, model_path)


import matplotlib.pyplot as plt


def plot_accuracies(history):
    """ Plot the history of accuracies"""
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');


plot_accuracies(history)



def plot_losses(history):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');


plot_losses(history)


test_dataset = FaceMaskDataset(images, test_indexes, conversion=transform)
print("Loading test set")
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


y_true = torch.tensor([])
y_true = y_true.to(device)
y_preds = torch.tensor([])
y_preds = y_preds.to(device)

# test-the-model
model.eval()  # it-disables-dropout
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


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


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


from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(y_true, y_preds)
print(classification_report(y_true, y_preds))


def label_to_classname(label):
    for classname in classes.keys():
        if classes[classname] == label:
            return classname
    return 'NULL'



# Put new images at the 'test' directory to classify them
new_images_path = os.path.join(root_folder, "./test")

# get new images
new_images = os.listdir(new_images_path)

with torch.no_grad():
    for image in new_images:
        file_name = image
        image = transform(Image.open(os.path.join(new_images_path, image)).convert('RGB'))
        # image = image.unsqueeze(1)
        image = image.unsqueeze(0)
        image = image.to(device)
        labels = model(image)
        _, predicted = torch.max(labels.data, 1)
        print(f'{file_name} file is {label_to_classname(predicted[0])}')
