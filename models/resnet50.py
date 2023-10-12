import torch
import torch.nn as nn
from torchvision.models import resnet50
from models.cossim import CosSim
from models import register_network

@register_network('resnet50')

class ResNet50(nn.Module): 
    def __init__(self,
                 nbit, nclass, pretrained=False, freeze_weight=False,
                 codebook=None,
                 **kwargs):
        super(ResNet50, self).__init__()

        model = resnet50(pretrained=pretrained)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.features = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        in_features = model.fc.in_features
        

        self.hash_fc = nn.Sequential(
            nn.Linear(in_features, nbit, bias=False),
            nn.BatchNorm1d(nbit, momentum=0.1)
        )

        nn.init.normal_(self.hash_fc[0].weight, std=0.01)
        # nn.init.zeros_(self.hash_fc.bias)

        if codebook is None:  # usual CE
            self.ce_fc = nn.Linear(nbit, nclass)
        else:
            # not learning cent, we are doing codebook learning
            self.ce_fc = CosSim(nbit, nclass, codebook, learn_cent=False)

        self.extrabit = 0

        if freeze_weight:
            for param in self.features.parameters():
                param.requires_grad_(False)

    def set_codebook(self, codebook):
        self.ce_fc.set_codebook(codebook, learn_cent=False)

    def get_backbone_params(self):
        return list(self.features.parameters()) 

    def get_hash_params(self):
        return list(self.ce_fc.parameters()) + list(self.hash_fc.parameters())

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        v = self.hash_fc(x)
        u = self.ce_fc(v)
        return u, v

if __name__ == '__main__':
    model = ResNet50(32, 10)
    model.eval()
    x = torch.randn((1, 3, 224, 224))

    u, v = model(x)
    print(u.shape)
    print(v.shape)