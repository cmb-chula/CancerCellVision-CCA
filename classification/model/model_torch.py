import torchvision.models as models
from torch import nn
import torch

def load_pretrained_VIT(img_size = 128, mode = 'VIT-B'):
  if(mode == 'VIT-B'):
    base_model = models.vit_b_16(weights='IMAGENET1K_V1').cuda()
    target_model = models.vit_b_16(image_size = img_size).cuda()
  if(mode == 'VIT-L'):
    base_model = models.vit_l_16(weights='IMAGENET1K_V1').cuda()
    target_model = models.vit_l_16(image_size = img_size).cuda()

  base_model_weight = base_model.state_dict()
  base_model_weight.pop('encoder.pos_embedding')
  target_model.load_state_dict(base_model_weight, strict=False)
  del base_model
  return target_model


class ClassificationModel(nn.Module):
    def __init__(self, input_shape = (128, 128, 3), backbone = None,num_classes = 3):
        super(ClassificationModel, self).__init__()
        self.backbone = None
        self.input_shape = input_shape
        self.projector_dim_size = None

        if(backbone == 'effnet-b7'):
            self.backbone = models.efficientnet_b7(weights = 'IMAGENET1K_V1')
            self.projector_dim_size = 2560
            self.backbone.classifier = nn.Identity(self.projector_dim_size, num_classes)
            if(self.input_shape[2] != 3):
                self.backbone.features[0][0] = nn.Conv2d(self.input_shape[2], 64,  kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        if(backbone == 'effnet-b4'):
            self.backbone = models.efficientnet_b4(weights = 'IMAGENET1K_V1')
            self.projector_dim_size = 1792
            self.backbone.classifier = nn.Identity(self.projector_dim_size, num_classes)
            if(self.input_shape[2] != 3):
                self.backbone.features[0][0] = nn.Conv2d(self.input_shape[2], 48,  kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        if(backbone == 'effnet-b1'):
            self.backbone = models.efficientnet_b1(weights = 'IMAGENET1K_V1')
            self.projector_dim_size = 1280
            self.backbone.classifier = nn.Identity(self.projector_dim_size, num_classes)
            if(self.input_shape[2] != 3):
                self.backbone.features[0][0] = nn.Conv2d(self.input_shape[2], 32,  kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)


        elif(backbone == 'resnet-50'):
            self.backbone = models.resnet50(pretrained=True)
            if(self.input_shape[2] != 3):
                self.backbone.conv1 = nn.Conv2d(self.input_shape[2], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.projector_dim_size = 2048
            self.backbone.fc = nn.Identity(self.projector_dim_size)

        elif(backbone == 'resnet-101'):
            self.backbone = models.resnet101(weights='IMAGENET1K_V1')
            if(self.input_shape[2] != 3):
                self.backbone.conv1 = nn.Conv2d(self.input_shape[2], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.projector_dim_size = 2048
            self.backbone.fc = nn.Identity(self.projector_dim_size)

        elif(backbone == 'resnet-152'):
            self.backbone = models.resnet152(weights='IMAGENET1K_V1')
            if(self.input_shape[2] != 3):
                self.backbone.conv1 = nn.Conv2d(self.input_shape[2], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.projector_dim_size = 2048
            self.backbone.fc = nn.Identity(self.projector_dim_size)


        elif(backbone == 'densenet-121'):
            self.backbone = models.densenet121(weights='IMAGENET1K_V1')
            if(self.input_shape[2] != 3):
                self.backbone.features[0] = nn.Conv2d(self.input_shape[2], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.projector_dim_size = 1024
            self.backbone.classifier = nn.Identity(self.projector_dim_size)

        elif(backbone == 'densenet-169'):
            self.backbone = models.densenet169(weights='IMAGENET1K_V1')
            if(self.input_shape[2] != 3):
                self.backbone.features[0] = nn.Conv2d(self.input_shape[2], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.projector_dim_size = 1664
            self.backbone.classifier = nn.Identity(self.projector_dim_size)

        elif(backbone == 'densenet-201'):
            self.backbone = models.densenet201(weights='IMAGENET1K_V1')
            if(self.input_shape[2] != 3):
                self.backbone.features[0] = nn.Conv2d(self.input_shape[2], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.projector_dim_size = 1920
            self.backbone.classifier = nn.Identity(self.projector_dim_size)


        elif(backbone == 'ConvNext-S'):
            self.backbone = models.convnext_small(weights = 'IMAGENET1K_V1')
            if(self.input_shape[2] != 3):
                self.backbone.features[0][0] = nn.Conv2d(self.input_shape[2], 96,  kernel_size=(4, 4), stride=(4, 4))
            self.projector_dim_size = 768
            self.backbone.classifier[2] = nn.Identity(self.projector_dim_size)

        elif(backbone == 'ConvNext-B'):
            self.backbone = models.convnext_base(weights = 'IMAGENET1K_V1')
            if(self.input_shape[2] != 3):
                self.backbone.features[0][0] = nn.Conv2d(self.input_shape[2], 128,  kernel_size=(4, 4), stride=(4, 4))
            self.projector_dim_size = 1024
            self.backbone.classifier[2] = nn.Identity(self.projector_dim_size)

        elif(backbone == 'ConvNext-L'):
            self.backbone = models.convnext_large(weights = 'IMAGENET1K_V1')
            if(self.input_shape[2] != 3):
                self.backbone.features[0][0] = nn.Conv2d(self.input_shape[2], 192,  kernel_size=(4, 4), stride=(4, 4))
            self.projector_dim_size = 1536
            self.backbone.classifier[2] = nn.Identity(self.projector_dim_size)


        elif(backbone == 'VIT-B'):
            self.backbone = load_pretrained_VIT(self.input_shape[0], mode = 'VIT-B')
            if(self.input_shape[2] != 3):
                self.backbone.conv_proj = nn.Conv2d(self.input_shape[2], 768,  kernel_size=(16, 16), stride=(16, 16))
            self.projector_dim_size = 768
            self.backbone.heads.head = nn.Identity(self.projector_dim_size)

        elif(backbone == 'VIT-B-224'):
            self.backbone = models.vit_b_16(weights='IMAGENET1K_V1').cuda()
            if(self.input_shape[2] != 3):
                self.backbone.conv_proj = nn.Conv2d(self.input_shape[2], 768,  kernel_size=(16, 16), stride=(16, 16))
            self.projector_dim_size = 768
            self.backbone.heads.head = nn.Identity(self.projector_dim_size)

        elif(backbone == 'VIT-L'):
            self.backbone = load_pretrained_VIT(self.input_shape[0], mode = 'VIT-L')
            if(self.input_shape[2] != 3):
                self.backbone.conv_proj = nn.Conv2d(self.input_shape[2], 1024,  kernel_size=(16, 16), stride=(16, 16))
            self.projector_dim_size = 1024
            self.backbone.heads.head = nn.Identity(self.projector_dim_size)


        elif(backbone == 'Swin-B'):
            self.backbone = models.swin_b(weights='IMAGENET1K_V1')
            if(self.input_shape[2] != 3):
                self.backbone.features[0][0] = nn.Conv2d(self.input_shape[2], 128,  kernel_size=(4, 4), stride=(4, 4))
            self.projector_dim_size = 1024
            self.backbone.head = nn.Identity(self.projector_dim_size)

        elif(backbone == 'Swin-S'):
            self.backbone = models.swin_s(weights='IMAGENET1K_V1')
            if(self.input_shape[2] != 3):
                self.backbone.features[0][0] = nn.Conv2d(self.input_shape[2], 96,  kernel_size=(4, 4), stride=(4, 4))
            self.projector_dim_size = 768
            self.backbone.head = nn.Identity(self.projector_dim_size)

        self.cls = nn.Linear(self.projector_dim_size, num_classes)
        assert self.backbone != None

    def forward(self, x, return_embedding = False):
        embedding = self.backbone(x)
        # print(embedding.shape)
        conf = self.cls(embedding)
        if(return_embedding):
          return conf, embedding
        else:
          return conf
