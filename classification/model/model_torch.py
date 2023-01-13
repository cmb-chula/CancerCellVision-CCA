import torchvision.models as models
from torch import nn
import torch
class ClassificationModel(nn.Module):
    def __init__(self, input_shape = (128, 128, 3), backbone = None,num_classes = 3):
        super(ClassificationModel, self).__init__()
        self.backbone = None
        self.input_shape = input_shape
        
        if(backbone == 'effnet'):
            self.backbone = models.efficientnet_b4(weights = models.EfficientNet_B4_Weights.DEFAULT)
            self.backbone.classifier = nn.Identity(1792, num_classes)
            self.cls = nn.Linear(1792, num_classes)
            if(self.input_shape[2] != 3):
                self.backbone.features[0][0] = nn.Conv2d(self.input_shape[2], 48,  kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        elif(backbone == 'resnet'):
            self.backbone = models.resnet50(pretrained=True)
            if(self.input_shape[2] != 3):
                self.backbone.conv1 = nn.Conv2d(self.input_shape[2], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.backbone.fc = nn.Identity(2048)
            self.cls = nn.Linear(2048, num_classes)

        elif(backbone == 'densenet'):
            self.backbone = models.densenet169(pretrained=True)
            # print(self.backbone)
            if(self.input_shape[2] != 3):
                self.backbone.features[0] = nn.Conv2d(self.input_shape[2], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.backbone.classifier = nn.Identity(1664)
            self.cls = nn.Linear(1664, num_classes)

        elif(backbone == 'convnext'):
            self.backbone = models.convnext_base(weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
            self.backbone.classifier[2] = nn.Identity(1024)
            if(self.input_shape[2] != 3):
                self.backbone.features[0][0] = nn.Conv2d(self.input_shape[2], 128,  kernel_size=(4, 4), stride=(4, 4))
            self.cls = nn.Linear(1024, num_classes)
        assert self.backbone != None

    def forward(self, x, return_embedding = False):
        embedding = self.backbone(x)
        # print(embedding.shape)
        conf = self.cls(embedding)
        if(return_embedding):
          return conf, embedding
        else:
          return conf
