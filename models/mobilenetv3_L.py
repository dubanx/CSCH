import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
from models.cossim import CosSim
from models import register_network
 
@register_network('mobilenetv3_L')
class Mobilenet_v3_large(nn.Module):
    def __init__(self,
                 nbit, nclass, pretrained=False, freeze_weight=False,
                 codebook=None,
                 **kwargs):
        super(Mobilenet_v3_large, self).__init__()

        model = mobilenet_v3_large(pretrained=True)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier[:-1]

        self.hash_fc = nn.Sequential(
            nn.Linear(1280, nbit, bias=False),
            nn.BatchNorm1d(nbit, momentum=0.1)
        )
        nn.init.normal_(self.hash_fc[0].weight, std=0.01)

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
        return list(self.features.parameters()) + list(self.classifier.parameters()) 
    

    def get_hash_params(self):
        return list(self.ce_fc.parameters()) + list(self.hash_fc.parameters())

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = self.classifier(x)
        v = self.hash_fc(feat)
        u = self.ce_fc(v)
        return u, v
    
if __name__ == '__main__':
    model = Mobilenet_v3_large(32, 10)
    model.eval()
    x = torch.randn((1, 3, 224, 224))

    u, v = model(x)
    print(u.shape)
    print(v.shape)