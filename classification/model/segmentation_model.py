import torchvision.models as models
from torch import nn
import torch

class SegmentationModel(nn.Module):
    def __init__(self, input_shape = (128, 128, 3), backbone = None,num_classes = 3):
        super(SegmentationModel, self).__init__()
        self.backbone = None
        self.input_shape = input_shape
        out_channel = 3
        conv_dims = 128
        in_features = [1024, 512, 256, 128]
        n_stacks = [3, 2, 1, 1]
        self.scale_heads = []
        self.activation = {}

        def get_activation(name):
          def hook(model, input, output):
            self.activation[name] = output.detach()
          return hook


        if(backbone == 'effnet'):
            self.backbone = models.efficientnet_b4(weights = models.EfficientNet_B4_Weights.DEFAULT)
            self.backbone.classifier = nn.Identity(1792, num_classes)
            self.cls = nn.Linear(1792, num_classes)
            if(self.input_shape[2] != 3):
                self.backbone.features[0][0] = nn.Conv2d(self.input_shape[2], 48,  kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        elif(backbone == 'resnet'):
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = nn.Identity(2048)
            self.cls = nn.Linear(2048, num_classes)

        elif(backbone == 'convnext'):
            self.backbone = models.convnext_base(weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
            self.backbone.classifier[2] = nn.Identity(1024)
            if(self.input_shape[2] != 3):
                self.backbone.features[0][0] = nn.Conv2d(self.input_shape[2], 128,  kernel_size=(4, 4), stride=(4, 4))
            self.cls = nn.Linear(1024, num_classes)
            
            self.backbone.features[7][-1].register_forward_hook(get_activation('stage5'))
            self.backbone.features[5][-1].register_forward_hook(get_activation('stage4'))
            self.backbone.features[3][-1].register_forward_hook(get_activation('stage3'))
            self.backbone.features[1][-1].register_forward_hook(get_activation('stage2'))
            for idx, (in_feature, n_stack) in enumerate(zip(in_features, n_stacks)):
              layers = []
              for block_id, k in enumerate(range(n_stack)):
                in_channel = in_feature if block_id ==0 else conv_dims
                if(idx + 1 == len(n_stacks)):
                  conv_block = nn.Sequential(            
                      nn.Conv2d(in_channels = in_channel, out_channels=conv_dims, 
                                kernel_size=3, stride=1, padding = 1),
                      nn.GroupNorm(32, conv_dims),
                      nn.ReLU())
                else:
                  conv_block = nn.Sequential(            
                        nn.Conv2d(in_channels = in_channel, out_channels=conv_dims, 
                                  kernel_size=3, stride=1, padding = 1),
                        nn.GroupNorm(32, conv_dims),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
                layers.append(conv_block.cuda())
              self.scale_heads.append(nn.ModuleList(layers))  
            self.predictor = nn.Sequential( nn.Conv2d(conv_dims, out_channel, kernel_size=1, stride=1, padding=0),
                                          nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)).cuda()

        assert self.backbone != None


    def forward(self, x, return_embedding = False):
        embedding = self.backbone(x)
        conf = self.cls(embedding)

        decoder_inps = [self.activation['stage5'], self.activation['stage4'], self.activation['stage3'], self.activation['stage2']]
        segmentation_output = []
        for idx, i in enumerate(decoder_inps):
          for j in self.scale_heads[idx]:
            i = j(i)
          segmentation_output.append(i)
        merged = torch.sum(torch.stack(segmentation_output), axis = 0)
        out = self.predictor(merged)
        return conf, out