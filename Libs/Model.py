import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import skimage.io as io


class Net(nn.Module):

    def __init__(self, n_class=196):
        super(Net, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # print(self.model)
        ## Replace the last fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, n_class)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self.model(x)
            rv = torch.argmax(output, 1)
        self.train()
        return rv
