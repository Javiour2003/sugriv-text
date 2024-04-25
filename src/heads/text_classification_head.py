import torch
import torch.nn as nn
import torch.nn.functional as F

class TextClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextClassificationHead, self).__init__()

        # adds the cls to the input
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        input = self.fc(x)
        # apply the llm
        return F.log_softmax(input, dim=-1)