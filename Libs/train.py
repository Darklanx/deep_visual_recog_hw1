'''
reference: https://www.kaggle.com/deepbear/pytorch-car-classifier-90-accuracy?select=names.csv
'''

import time
import torch
import traceback


def train_model(model,
                train_loader,
                test_loader,
                criterion,
                optimizer,
                scheduler,
                n_epochs=5):

    losses = []
    accuracies = []
    test_accuracies = []
    # set the model to train mode initially
    model.train()
    for epoch in range(n_epochs):
        print("Training epoch {} ...".format(epoch + 1))
        since = time.time()
        running_loss = 0.0
        running_correct = 0.0
        for i, data in enumerate(train_loader, 0):

            # get the inputs and assign them to cuda
            inputs, labels = data
            # print(i)
            # print(inputs.shape)
            #inputs = inputs.to(device).half() # uncomment for half precision model
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            predicted = torch.argmax(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate the loss/acc later
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

        epoch_duration = time.time() - since
        len_dataset = (len(train_loader) * train_loader.batch_size)
        epoch_loss = running_loss / len_dataset
        epoch_acc = running_correct / len_dataset
        print("Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" %
              (epoch + 1, epoch_duration, epoch_loss, epoch_acc))

        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        # switch the model to eval mode to evaluate on test data
        '''
        model.eval()
        test_acc = eval_model(model, test_loader)
        test_accuracies.append(test_acc)
        '''

        # re-set the model to train mode after validating
        model.train()
        scheduler.step()
        since = time.time()
    print('Finished Training')

    return model, losses, accuracies, test_accuracies


def eval_model(model, test_loader):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            #images = images.to(device).half() # uncomment for half precision model
            images = images.to(model.device)
            labels = labels.to(model.device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (test_acc))

    return test_acc