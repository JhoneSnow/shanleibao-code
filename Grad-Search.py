import torch
import numpy as np
import skorch
import sklearn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import generate_model
from setting import parse_opts
from torchvision.transforms import transforms
from dataset_make import LiverDataset,LiverDataset_val
import matplotlib.pyplot as plt
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from torch import optim
from torch import nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import torch.nn.functional as F


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class BrainNet(nn.Module):

    def __init__(self):
        super(BrainNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.conv2 = nn.Conv3d(
            64,
            16,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False

        )
        self.bn2 = nn.BatchNorm3d(16)
        self.GAP = nn.AdaptiveAvgPool3d((6, 6, 6))
        self.drop = nn.Dropout(0.5)
        self.Dense1 = nn.Linear(3456, 64)
        self.Dense2 = nn.Linear(64, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = F.relu(self.Dense1(x))
        x = self.drop(x)
        x = self.Dense2(x)
        return x

class InfantBrainNet(nn.Module):
    def __init__(self):
        super(InfantBrainNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            30,
            kernel_size=(6, 6, 6),
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.batch_nom = nn.BatchNorm3d(30)
        self.maxpooling = nn.MaxPool3d(4)
        self.conv2 = nn.Conv3d(30,
                               30,
                               kernel_size=(3, 3, 3),
                               stride=(2, 2, 2),
                               padding=(3, 3, 3),
                               bias=False
                               )
        self.maxpooling2 = nn.MaxPool3d(3)
        self.conv3 = nn.Conv3d(30,
                               30,
                               kernel_size=(3, 3, 3),
                               stride=(2, 2, 2),
                               padding=(3, 3,3 ),
                               bias=False

                               )
        self.maxpooling3 = nn.MaxPool3d(2)

        self.conv4 = nn.Conv3d(30,
                               30,
                               kernel_size=(2, 2, 2),
                               stride=(2, 2, 2),
                               padding=(3, 3, 3),
                               bias=False
                               )
        self.GlobalMaxPooling = nn.AdaptiveMaxPool3d((2, 2, 2))
        self.Dense1 = nn.Linear(4320, 32)
        self.Dense2 = nn.Linear(32, 2)
        self.Dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_nom(x)
        x = self.relu1(x)
        x = self.maxpooling(x)
        x = self.conv2(x)
        x = self.batch_nom(x)
        x = self.relu1(x)
        x = self.maxpooling2(x)
        x = self.conv3(x)
        x = self.batch_nom(x)
        x = self.relu1(x)
        x = self.maxpooling2(x)
        x = self.conv4(x)
        x = self.batch_nom(x)
        x = self.relu1(x)
        x = self.maxpooling3(x)
        x = self.GlobalMaxPooling(x)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x)
        x = F.relu(self.Dense1(x))
        x = self.Dropout(x)
        x = F.softmax(self.Dense2(x))





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
liver_dataset = LiverDataset("/home/sinclair/Documents/nii/target_data/target_adc/norm", transform=None, target_transform=None)
test_dataset = LiverDataset_val("/home/sinclair/Documents/nii/target_data/target_adc/norm", transform=None, target_transform=None)
#dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#print(dataloaders)
def Grad_search():
    torch.manual_seed(0)
    #model, parameters = generate_model(sets)
    model = InfantBrainNet()
    #model = model.module
    net = NeuralNetClassifier(model,
                              max_epochs= 50,
                              lr= 0.000027,
                              iterator_train__num_workers=4,
                              iterator_valid__num_workers=4,
                              batch_size=18,
                              optimizer=optim.Adam,
                              criterion=nn.CrossEntropyLoss,
                              device=device,
                              #optimizer__momentum=0.9,
                              optimizer__weight_decay=1e-1

                              #train_split=False,
                              )
    y = np.array([y for x, y in iter(liver_dataset)])
    net.fit(liver_dataset, y)
    val_loss=[]
    train_loss=[]
    for i in range(50):
        val_loss.append(net.history[i]['valid_loss'])
        train_loss.append(net.history[i]['train_loss'])

    plt.figure(figsize=(10, 8))
    plt.semilogy(train_loss, label='Train loss')
    plt.semilogy(val_loss, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.show()
    y_pred = net.predict(test_dataset)
    y_test = np.array([y for x, y in iter(test_dataset)])
    accuracy_score(y_test, y_pred)
    plot_confusion_matrix(net, test_dataset, y_test.reshape(-1, 1))
    plt.show()
'''
    params = {
        'max_epochs':[20, 50, 70, 90, 100],
        'lr':[0.001, 0.002, 0.005, 0.01, 0.05, 0.1],
        'batch_size':[2, 5, 10, 16],
        'opimizer__momentum':[0.2, 0.5, 0.9],

    }
    gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy', verbose=2)
    y = np.array([y for x, y in iter(liver_dataset)])
    gs.fit(liver_dataset, y)
'''
 __name__ == '__main__':
    # settting
    sets = parse_opts()
    sets.target_type = "normal"
    sets.phase = 'train'
    Grad_search()

