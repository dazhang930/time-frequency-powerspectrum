import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from sklearn import metrics
import cv2
import numpy as np
import matplotlib.pyplot as plt
from accelerate import Accelerator
import torchvision
from torchvision import datasets
from sklearn.model_selection import LeaveOneOut

import os
import gc
import sys
import glob
import math
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from SpectroDataset import SpectroDataset
from Model1D import Model

import warnings
warnings.filterwarnings('ignore')
from accelerate import Accelerator

from functools import partial
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import cmocean
import cmocean.cm as cmo


from colorama import Fore, Back, Style
r_ = Fore.RED
b_ = Fore.BLUE
c_ = Fore.CYAN
g_ = Fore.GREEN
y_ = Fore.YELLOW
m_ = Fore.MAGENTA
sr_ = Style.RESET_ALL

config = {'lr':1e-3,
          'wd':1e-2,
          'bs':4,
          # 'img_size':128,
          'epochs':50,
          'seed':1000}


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(seed=config['seed'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(0.25),
            # torchvision.transforms.RandomCrop(32, padding=(21, 18)),
            # transforms.RandomVerticalFlip(0.25),
            # transforms.ColorJitter(brightness=(0.2,0.7),contrast=1,saturation=0.9,hue=0.4),
            # transforms.ColorJitter(brightness=(0.1,0.6), contrast=1,saturation=0.5, hue=0.4)
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),           
            # transforms.Resize((64, 64))
            # transforms.RandomResizedCrop((27,50))
        ])

valid_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


annots = loadmat('ls_animal_73.mat')
ls_animal = annots['ls_animal']
# ls_animal = ls_animal[:,:13,:]
annots = loadmat('ls_person_73.mat')
ls_person = annots['ls_person']
# ls_person = ls_person[:,:13,:]
annots = loadmat('ls_tool_73.mat')
ls_tool = annots['ls_tool']

ls_animal = ls_animal.reshape(ls_animal.shape[0], ls_animal.shape[1]*ls_animal.shape[2])
ls_person = ls_person.reshape(ls_person.shape[0], ls_person.shape[1]*ls_person.shape[2])
ls_tool = ls_tool.reshape(ls_tool.shape[0], ls_tool.shape[1]*ls_tool.shape[2])

print(ls_animal.shape)
print(ls_person.shape)
print(ls_tool.shape)


x = np.concatenate([ls_animal, ls_person, ls_tool])
y = np.concatenate([ [0]*ls_animal.shape[0], [1]*ls_person.shape[0], [2]*ls_tool.shape[0] ])



# paths = np.concatenate([ls_animal, ls_tool])
paths = np.concatenate([ls_person, ls_tool])

# x = np.concatenate([ls_person, paths])
x = np.concatenate([ls_animal, paths])
# y = np.concatenate([ [1]*ls_person.shape[0], [0]*paths.shape[0] ])

y = np.concatenate([ [1]*ls_animal.shape[0], [0]*paths.shape[0] ])


# random.shuffle(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)



train_dataset = SpectroDataset(x_train, y_train, transform=train_transform)
# train_dataset = SpectroDataset(train_paths,augmentations=get_val_transforms())
train_dl = DataLoader(train_dataset,batch_size=config['bs'],shuffle=True)

dataiter = iter(train_dl)
sample, labels = next(dataiter)

# cmap = cmo.curl
img = torchvision.utils.make_grid(sample).permute(1,2,0).numpy()
norm = plt.Normalize(vmin=img.min(), vmax=img.max())


img = norm(img)
plt.figure(figsize=(15,15))
plt.imsave("show.png",img)



train_dataset = SpectroDataset(x_train, y_train, transform=train_transform)
train_dl = DataLoader(train_dataset,batch_size=4,shuffle=True)
    
#valid
valid_dataset = SpectroDataset(x_test, y_test, transform=valid_transform)
valid_dl = DataLoader(valid_dataset,batch_size=4,shuffle=False)



trainingEpoch_acc = []
validationEpoch_acc = []



out = []
from sklearn.metrics import confusion_matrix
import pandas as pd
def inference (model, val_dl):
  correct_prediction = 0
  total_prediction = 0
  # all_y_pred = []
  # all_y_true = []
  y_pred_list = []
  y_true_list = []

  # Disable gradient updates
  with torch.no_grad():
    for data in val_dl:
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data[0].to(device), data[1].to(device)

      # Normalize the inputs
      # inputs_m, inputs_s = inputs.mean(), inputs.std()
      # inputs = (inputs - inputs_m) / inputs_s

      # Get predictions
      outputs = model(inputs)

      # Get the predicted class with the highest score
      _, prediction = torch.max(outputs,1)
      y_pred_list.extend(prediction.cpu())
      y_true_list.extend(labels.cpu())

      print(outputs)
      # out.extend(torch.max(outputs).cpu())
      print(prediction, labels)
      # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
    
  acc = correct_prediction/total_prediction
  print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
  return out,y_pred_list,y_true_list


trainingEpoch_acc = []
validationEpoch_acc = []
result = []


def training(model, train_dl,  num_epochs):
  # Loss Function, Optimizer and Scheduler
  criterion = nn.CrossEntropyLoss()
  # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-3, momentum=0.9)
  # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, verbose=True)

  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')
  


  # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

  # # Decay LR by a factor of 0.1 every 7 epochs
  # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.001)
  
  
  def evaluate(model, valid_loader):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
      for data in valid_loader:
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # Normalize the inputs
        # inputs_m, inputs_s = inputs.mean(), inputs.std()
        # inputs = (inputs - inputs_m) / inputs_s

        # Get predictions
        outputs = model(inputs)

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]
      
    acc = correct_prediction/total_prediction
    return acc
    # print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')


  # Repeat for each epoch

  best_acc = -1.0


  for epoch in range(num_epochs):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # Repeat for each batch in the training set
    for i, data in enumerate(train_dl):
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)

        # Normalize the inputs
        # inputs_m, inputs_s = inputs.mean(), inputs.std()
        # inputs = (inputs - inputs_m) / inputs_s

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]

        #if i % 10 == 0:    # print every 10 mini-batches
        #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
    
    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    # test_acc = evaluate(model, test_dl)

    

    # if test_acc > best_acc:
    #   print(f"{g_}Accuracy Increased from {best_acc} to {test_acc}{sr_}")

    #   best_acc = test_acc
    #   torch.save(model.state_dict(),'./imagenet_vq_vae_model.bin')
    torch.save(model.state_dict(),'./imagenet_vq_vae_model.bin')

    trainingEpoch_acc.append(np.array(acc).mean())
    # validationEpoch_acc.append(np.array(test_acc).mean())
    # print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}, Valid Accuracy: {test_acc:.2f}')
    print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
    
  # test_acc = evaluate(model, test_dl)
  # result.append(test_acc)
    

  print('Finished Training')
  
myModel = Model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda/

next(myModel.parameters()).device


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_para = count_parameters(myModel)
print("Model Number of Parameters: ", num_para)



num_epochs=1  # Just for demo, adjust this higher.
training(myModel, train_dl, num_epochs)



cv = LeaveOneOut()
count = 1
# enumerate splits
y_true, y_pred = list(), list()
for train_ix, test_ix in cv.split(x):
  # split data
  print("This is for the: %s item"%count)
  count += 1
  x_train_valid, x_test = x[train_ix, :], x[test_ix, :]
  y_train_valid, y_test = y[train_ix], y[test_ix]
  # fit model
  #  print(y_test)
  # x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, shuffle=True, test_size=0.2)

  train_dataset = SpectroDataset(x_train_valid, y_train_valid, transform=train_transform)
  train_dl = DataLoader(train_dataset, batch_size=4,shuffle=True)

  test_dataset = SpectroDataset(x_test, y_test, transform=valid_transform)
  test_dl = DataLoader(test_dataset, batch_size=4,shuffle=False)


  myModel = Model()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  myModel = myModel.to(device)
  # Check that it is on Cuda
  next(myModel.parameters()).device
  num_epochs=100  # Just for demo, adjust this higher.
  training(myModel, train_dl, num_epochs)

  myModel = Model()
  myModel.load_state_dict(torch.load('./imagenet_vq_vae_model.bin'))
  if torch.cuda.is_available():
    myModel.cuda()

  outputs, pred, ground_truth= inference(myModel, test_dl)
  y_true.append(ground_truth)
  y_pred.append(pred)


y_pred_list = [i[0].cpu().numpy().item() for i in y_pred]
y_true_list = [i[0].cpu().numpy().item() for i in y_true]
print(y_pred_list, y_true_list)

y_true = y_true_list
y_pred = y_pred_list

fpr_cnn, tpr_cnn, _ = metrics.roc_curve(y_true_list, y_pred_list)
roc_auc_cnn = metrics.roc_auc_score(y_true_list, y_pred_list)

print("AUC_ROC Score: ",roc_auc_cnn)

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, average_precision_score

precision1, recall1, thresholds = precision_recall_curve(y_true_list, y_pred_list)
# print(len(precision1), len(recall1))

prauc = round(auc(recall1, precision1),6)
print("PR Score: ",prauc)

# plt.figure(0).clf()
# plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Random Classifier')
# plt.plot(fpr_cnn,tpr_cnn,color='red',label="CNN Animal, (AUC = %0.3f)"%(roc_auc_cnn))
# plt.legend(loc=0)

num_para = count_parameters(myModel)
print("Model Number of Parameters: ", num_para)