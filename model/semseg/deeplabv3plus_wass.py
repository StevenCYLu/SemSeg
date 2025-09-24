import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()
        if 'resnet' in cfg['backbone']:
            self.backbone = \
                resnet.__dict__[cfg['backbone']](True, 
                    replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
        else:
            assert cfg['backbone'] == 'xception'
            self.backbone = xception(True)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

        self.representation = nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.5),
                nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.5),
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=True),
            )
        self.predictor = nn.Sequential(
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )
        

    def forward(self, x, need_fp=False, need_ft=False, pred=False, need_c4=False):
        h, w = x.shape[-2:]
        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]  


        if need_fp and need_ft:
            outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
                                torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            feature = self.representation(c4)

            if pred:
                pred_f = feature.view(-1, 256)
                pred_f = self.predictor(pred_f)
                pred_f = pred_f.reshape(feature.shape)
                return out, out_fp, pred_f
            return out, out_fp, feature
        
        if need_fp:
            outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
                                torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)

            return out, out_fp
        if need_ft:
            feature = self._decode(c1, c4)
            feature = F.interpolate(feature, size=(h, w), mode="bilinear", align_corners=True)
            out = self.classifier(feature)
            feature = self.representation(c4)

            if pred:
                pred_f = feature.view(-1, 256)
                pred_f = self.predictor(pred_f)
                pred_f = pred_f.reshape(feature.shape)
                return out, pred_f
            return out, feature
        
        if need_c4:
            feature = self._decode(c1, c4)
            feature = F.interpolate(feature, size=(h, w), mode="bilinear", align_corners=True)
            out = self.classifier(feature)
            return out, c4

        feature = self._decode(c1, c4)
        feature = F.interpolate(feature, size=(h, w), mode="bilinear", align_corners=True)
        out = self.classifier(feature)
        
        return out

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)
        # print(feature.shape)
        # feature = F.interpolate(feature, size=(h, w), mode="bilinear", align_corners=True)
        # print(feature.shape)

        # out = self.classifier(feature)

        return feature


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)
