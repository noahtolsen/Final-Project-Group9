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
learning_rate = 0.0001

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(3 , 32, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.fc1 = torch.nn.Linear(128 * 28 * 28, 128, bias= True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(128, 29, bias = True)
        self.softmax = nn.LogSoftmax(dim=1)



    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = F.relu(out)
        out = F.relu(self.conv2_bn(self.conv2(out)))
        out = self.pool(out)

        out = F.relu(self.conv3_bn(self.conv3(out)))
        out = F.relu(self.conv4_bn(self.conv4(out)))
        out = self.pool(out)

        out = self.relu(self.conv5_bn(self.conv5(out)))
        out = self.relu(self.conv6_bn(self.conv6(out)))
        out = self.pool(out)

        #print(out.shape)

        out = out.view(-1, 128 * 28 * 28)

        out = self.dropout(F.relu(self.fc1(out)))
        out = self.softmax(self.out(out))

        return (out)


#--------------------------------------------------------------------------------------------

net = Net()
net.cuda()
nn.DataParallel(net)
#-------------------------------------------------------------------------------
criterion = nn.NLLLoss();
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate);

j = 0
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epochs):
    for i, data in enumerate(dataloaders['asl_alphabet_train']):
        images, labels = data
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        # iteration_list.append(i)

        loss.backward()
        optimizer.step()

        j += 1
        if j % 32 == 0:
            correct = 0
            total = 0

            for images, labels in dataloaders['asl_alphabet_test']:
                images = Variable(images).cuda()
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                # total = len(labels)
                total += labels.size(0)
                correct += (predicted.cpu() == labels).sum()
            accuracy = 100*correct/float(total)

            loss_list.append(loss.data)
            iteration_list.append(j)
            accuracy_list.append(accuracy)

            if j % 64 == 0:
                print('Epoch {}/{} Iteration: {} Loss: {} Accuracy: {}%'.format(epoch + 1,num_epochs,  j, loss.data, accuracy))


        # print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
        #           % (epoch + 1, num_epochs, i + 1, len(image_datasets['asl_alphabet_train']) // batch_size, loss.item()))


plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("Loss vs Number of iteration")
plt.show()
plt.savefig('our_model_loss.png')


plt.plot(iteration_list,accuracy_list , color="red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of iteration")
plt.show()
plt.savefig('our_model_accuracy.png')
torch.save(net.state_dict(), 'model.pkl')

# correct = 0
# total = 0
#
# for images, labels in dataloaders['asl_alphabet_test']:
#         images = Variable(images).cuda()
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted.cpu() == labels).sum()
#
# print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
