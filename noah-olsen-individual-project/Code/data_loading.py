import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F


torch.backends.cudnn.deterministic = True
torch.manual_seed(999)

data_dir = "/home/ubuntu/final_project/Data"

input_size = 64*64*3
num_classes = 29
num_epochs = 5
batch_size = 16
learning_rate = 0.00001

classes = ('A', 'B', 'C', 'D', 'del','E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'nothing','O', 'P', 'Q', 'R', 'S', 'space','T', 'U', 'V', 'W', 'X', 'Y', 'Z')

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'asl_alphabet_train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'asl_alphabet_test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['asl_alphabet_train', 'asl_alphabet_test']}


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True)
              for x in ['asl_alphabet_train', 'asl_alphabet_test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['asl_alphabet_train', 'asl_alphabet_test']}

class_names = image_datasets['asl_alphabet_train'].classes
print(class_names)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ",device)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(dataloaders['asl_alphabet_train'])
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
plt.show()

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


#The network model goes here
