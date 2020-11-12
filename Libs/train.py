'''
reference: https://www.kaggle.com/deepbear/pytorch-car-classifier-90-accuracy?select=names.csv
'''

import time
import torch
import traceback
import os


def train_model(model,
                train_loader,
                test_loader,
                criterion,
                optimizer,
                scheduler,
                start_epoch,
                end_epoch=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    losses = []
    accuracies = []
    test_accuracies = []
    # set the model to train mode initially
    model.train()
    for epoch in range(start_epoch, end_epoch):
        model.train()
        print("Training epoch {} with lr {}...".format(
            epoch + 1, optimizer.param_groups[0]['lr']))
        since = time.time()
        running_loss = 0.0
        running_correct = 0.0
        for i, data in enumerate(train_loader, 0):
            if i % 30 == 0:
                print(i)

            # get the inputs and assign them to cuda
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # print("\nOutside: input size", inputs.size(), "output size",
            #   outputs.size())
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
        model.eval()
        test_acc = eval_model(model, test_loader)
        test_accuracies.append(test_acc)

        # re-set the model to train mode after validating

        model.train()

        if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step(test_acc)
        else:
            scheduler.step()
        save = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler,
            "test_acc": test_acc
        }
        since = time.time()

        torch.save(save,
                   '{}.pth'.format(os.path.join("./model/", str(epoch + 1))))
    print('Finished Training')

    return model, losses, accuracies, test_accuracies


def eval_model(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print(correct)
    # print(total)
    # print(correct / total)
    test_acc = 100.0 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (test_acc))

    return test_acc