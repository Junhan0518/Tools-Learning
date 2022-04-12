import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder,ImageFolder
import torch.nn.functional as F


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(0)

# root_dir = 'casting_data/'
# train_path = root_dir + 'train/'
# train_defective = root_dir + 'train/def_front/'
# train_ok = root_dir + 'train/ok_front/'

# test_path = root_dir + 'test/'
# test_defective = root_dir + 'test/def_front/'
# test_ok = root_dir + 'test/ok_front/'

# fig, axes = plt.subplots(1, 2, figsize=(8,4))
# axes[0].imshow(plt.imread(train_defective+os.listdir(train_defective)[0]))
# axes[1].imshow(plt.imread(train_ok+os.listdir(train_ok)[0]))
# axes[0].set_title('Defective')
# axes[1].set_title('OK')
# plt.show()


# x_train = np.array([len(os.listdir(train_defective)),len(os.listdir(train_ok))])
# x_test = np.array([len(os.listdir(test_defective)),len(os.listdir(test_ok))])
# label = ['Defective','Ok']
  
# fig, axes = plt.subplots(1, 2, figsize=(8,4))
# axes[0].pie(x_train, labels=label, autopct='%1.1f%%',shadow=True, startangle=90)
# axes[1].pie(x_test, labels=label, autopct='%1.1f%%',shadow=True, startangle=90)
# axes[0].set_title('Train')
# axes[1].set_title('Test')
# plt.show()

# print(' Defective Training Images \t: ' + str(len(os.listdir(train_defective))))
# print(' Ok Training Images \t\t: ' + str(len(os.listdir(train_ok))))
# print()
# print(' Defective Testing Images \t: ' + str(len(os.listdir(test_defective))))
# print(' Ok Testing Images \t\t: ' + str(len(os.listdir(test_ok))))

# train_tfm = transforms.Compose([
#     # Resize the image into a fixed shape (height = width = 128)
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
# ])

# test_tfm = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
# ])


# batch_size = 64

# # Construct datasets.
# train_set = ImageFolder(train_path, transform=train_tfm)
# test_set = ImageFolder(test_path, transform=test_tfm)

# n = len(train_set)
# n_test = int(0.1 * n)
# train_set, dev_set = torch.utils.data.random_split(train_set, [n - n_test, n_test])

# # Construct data loaders.
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
# dev_loader = DataLoader(dev_set, batch_size = batch_size, shuffle = False, num_workers=0)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

class Origin(nn.Module):
    def __init__(self, dropout):
        super(Origin, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(3, 3, 0),

            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(3, 3, 0),

            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(3, 3, 0),  
        )
        self.fc_layers = nn.Sequential(
            nn.BatchNorm1d(8192),
            nn.Linear(8192, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(p= dropout),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)
        
        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x


def train(lr = 0.001, weight_decay = 1e-5, drop = 0.3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize a model, and put it on the device specified.
    model = Origin(drop).to(device)
    model.device = device

    # For the classification task, we use cross-entropy as the measurement of performance.
    # criterion = AdMSoftmaxLoss(11, 11, s=30.0, m=0.4).to(device)
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # The number of training epochs.
    n_epochs = 5

    # Whether to do semi-supervised learning.
    do_semi = False

    best_acc = 0

    total_accs = []
    total_loss = []
    total_dev_acc = []
    total_dev_loss = []
    for epoch in range(n_epochs):
        model.train()

        train_loss = []
        train_accs = []

        for batch in (train_loader):
            imgs, labels = batch
            print(imgs.size())

            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()

            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()

            acc = (logits.argmax(dim = -1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        total_accs.append(train_acc)
        total_loss.append(train_loss)
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        model.eval()

        # These are used to record information in validation.
        dev_loss = []
        dev_accs = []
        # Iterate the validation set by batches.
        for batch in (dev_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                  logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            dev_loss.append(loss.item())
            dev_accs.append(acc)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        dev_loss = sum(dev_loss) / len(dev_loss)
        dev_acc = sum(dev_accs) / len(dev_accs)
        total_dev_acc.append(dev_acc)
        total_dev_loss.append(dev_loss)
        print(f"[ dev | {epoch + 1:03d}/{n_epochs:03d} ] loss = {dev_loss:.5f}, acc = {dev_acc:.5f}")

 
        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(model, './model.pb')
#             print('saving model with acc {:.3f}'.format(best_acc))
    

    model = torch.load('model.pb')
    model = model.to(device)
    model.eval()
    test_loss = []
    test_accs = []
    y_pred = []
    y_label = []
    for batch in (test_loader):
        imgs, labels = batch

        logits = model(imgs.to(device))    
        loss = criterion(logits, labels.to(device))

        optimizer.zero_grad()

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        optimizer.step()
        
        pred = logits.argmax(dim = -1)
        acc = (pred == labels.to(device)).float().mean()
        y_pred.extend(pred)
        y_label.extend(labels)
        test_loss.append(loss.item())
        test_accs.append(acc)
    test_loss = sum(test_loss) / len(test_loss)
    test_acc = sum(test_accs) / len(test_accs)
    return test_acc, model
 
if __name__ =='__main__':

	 test_acc, model = train()
	 print(test_acc)