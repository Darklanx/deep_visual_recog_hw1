import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import skimage.io as io
from torch.nn import Conv2d


class Net(nn.Module):

    def __init__(self, n_class=196, use_att=False):
        super(Net, self).__init__()
        self.model = models.resnet152(pretrained=True)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        if use_att == True:
            pass
        else:
            modification = nn.Sequential(
                nn.Linear(self.model.fc.in_features, n_class))
        ## Replace the last fully connected layer
        self.model.fc = modification

    def forward(self, x):
        output = self.model(x)
        # print("\tIn Model: input size", x.size(), "output size", output.size())
        return output

    # def predict(self, x):
    #     self.eval()
    #     with torch.no_grad():
    #         output = self.model(x)
    #         rv = torch.argmax(output, 1)
    #     self.train()
    #     return rv
