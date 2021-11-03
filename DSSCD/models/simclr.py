import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
from copy import deepcopy
from modeling.backbone.resnet import ResNet50
from modeling.backbone.deeplabv2_cosim import deeplab_V2




class SimCLR(nn.Module):
    def __init__(self, args):
        super(SimCLR, self).__init__()
        self.m_backbone = args.m_backbone
        self.m = args.m_update
        self.encoder_type = args.encoder
        self.dense_cl = args.dense_cl
        self.f = get_encoder(args.backbone, args.pre_train, args.output_stride, args.encoder)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # projection head
        self.g = nn.Sequential(
                                nn.Linear(2048, args.hidden_layer, bias=False),
                                nn.BatchNorm1d(args.hidden_layer),
                                nn.ReLU(inplace=True),
                                nn.Linear(args.hidden_layer, args.n_proj, bias=True)
                               )

        # Momentum Encoder
        if args.m_backbone:
            self.fm = deepcopy(self.f)
            self.gm = deepcopy(self.g)
            self.dense_m= deepcopy(self.dense_neck)
            for param in self.fm.parameters():
                param.requires_grad = False
            for param in self.gm.parameters():
                param.requires_grad = False
            for param in self.gm.parameters():
                param.requires_grad = False

    def forward(self, x, y=None):
        x, _ = self.f(x)
        feat_x = self.pool(x)
        feat_x = torch.flatten(feat_x, start_dim=1)
        out_x = self.g(feat_x)
        if y is not None:
            if self.m_backbone:
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update()
                y, _ = self.fm(y)
                feat_y = self.pool(y)
                feat_y = torch.flatten(feat_y, start_dim=1)
                out_y = self.gm(feat_y)
            else:
                y, _ = self.f(y)
                feat_y = self.pool(y)
                feat_y = torch.flatten(feat_y, start_dim=1)
                out_y = self.g(feat_y)

            return F.normalize(feat_x, dim=-1), F.normalize(feat_y, dim=-1), F.normalize(out_x, dim=-1),  F.normalize(out_y, dim=-1)
        else:
            return F.normalize(feat_x, dim=-1), F.normalize(out_x, dim=-1)

    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update of the key encoder
        """
        for param_f, param_fm in zip(self.f.parameters(), self.fm.parameters()):
            param_fm.data = param_fm.data * self.m + param_f.data * (1. - self.m)
        for param_g, param_gm in zip(self.f.parameters(), self.fm.parameters()):
            param_gm.data = param_gm.data * self.m + param_g.data * (1. - self.m)


class LinearEvaluation(nn.Module):
    """
    Linear Evaluation model
    """

    def __init__(self, n_features, n_classes):
        super(LinearEvaluation, self).__init__()
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x1, x2):
        df = torch.abs(x1 - x2)
        return self.model(df)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_encoder(encoder, pre_train, output_stride, encoder_name):
    """
    Get Resnet backbone
    """

    class View(nn.Module):
        def __init__(self, shape=2048):
            super().__init__()
            self.shape = shape

        def forward(self, input):
            '''
            Reshapes the input according to the shape saved in the view data structure.
            '''
            batch_size = input.size(0)
            shape = (batch_size, self.shape)
            out = input.view(shape)
            return out

    def CMU_resnet50():

        if encoder_name=='resnet':
            resnet = ResNet50(BatchNorm=nn.BatchNorm2d, pretrained=pre_train, output_stride=output_stride)
            return resnet
        else:
             vgg16 = deeplab_V2()
             return vgg16
    return {

        'resnet50': CMU_resnet50()
    }[encoder]




if __name__ == "__main__":
    import torch
    model = SimCLR(a)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
