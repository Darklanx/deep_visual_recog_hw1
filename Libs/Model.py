import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import skimage.io as io
from torch.nn import Conv2d


class Net(nn.Module):

    def __init__(self, n_class=196, use_att=False):
        super(Net, self).__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model = nn.Sequential(
            *list(self.model.children())[:-1])  # remove last fc

        self.use_att = use_att
        # print(self.model)
        if use_att == True:
            out1 = 516
            out2 = 1
            self.attention = nn.Sequential(Conv2d(in_features, out1, 1),
                                           Conv2d(out1, out2, 1),
                                           Conv2d(out2, in_features, 1))

        self.fc = nn.Sequential(nn.Linear(in_features, n_class))
        ## Replace the last fully connected layer

    def forward(self, x):
        x = self.model(x)
        if self.use_att:
            attention = self.attention(x)
            # print("{}, {}".format(x.size(), attention.size()))
            # print(attention.size())
            x = (x * attention).view(x.size(0), -1)
            # print(x.size())

        output = self.fc(x)
        # print("\tIn Model: input size", x.size(), "output size", output.size())
        return output

    # def predict(self, x):
    #     self.eval()
    #     with torch.no_grad():
    #         output = self.model(x)
    #         rv = torch.argmax(output, 1)
    #     self.train()
    #     return rv
