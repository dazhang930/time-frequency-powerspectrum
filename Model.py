from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Audio Classification Model
# ----------------------------
class Model(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []
        self.conv1 = nn.Conv2d(1,  5, 5,  padding=1)
        self.pool = nn.MaxPool2d(3, 3, padding=1)
        self.fc3 = nn.Linear(5 * 17 * 16, 2)

        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        # nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        # nn.init.kaiming_normal_(self.fc1.weight, a=0.1)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = self.pool(x)
        # print(x.shape)
        x = x.view(-1, 5 * 17 * 16)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x