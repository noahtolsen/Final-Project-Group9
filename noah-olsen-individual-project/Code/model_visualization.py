import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

#Data loading goes here


#Model goes here


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

plt.plot(iteration_list,accuracy_list , color="red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of iteration")
plt.show()

torch.save(net.state_dict(), 'model.pkl')
