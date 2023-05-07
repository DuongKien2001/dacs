import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeContrastiveLoss(nn.Module):
    def __init__(self, cfg):
        super(PrototypeContrastiveLoss, self).__init__()
        self.cfg = cfg

    def forward(self, Proto, feat, labels, pixelWiseWeight=None):
        """
        Args:
            C: NUM_CLASSES A: feat_dim B: batch_size H: feat_high W: feat_width N: number of pixels except IGNORE_LABEL
            Proto: shape: (C, A) the mean representation of each class
            feat: shape (BHW, A) -> (N, A)
            labels: shape (BHW, ) -> (N, )
        Returns:
        """
        assert not Proto.requires_grad
        assert not labels.requires_grad
        assert feat.requires_grad
        assert feat.dim() == 2
        assert labels.dim() == 1
        # remove IGNORE_LABEL pixels

        feat = F.normalize(feat, p=2, dim=1)
        Proto = F.normalize(Proto, p=2, dim=1)
        logits = feat.mm(Proto.permute(1, 0).contiguous())
        logits = logits / self.cfg.MODEL.CONTRAST.TAU
        
        if pixelWiseWeight is None:
            ce_criterion = nn.CrossEntropyLoss(ignore_index = 255)
            ce_criterion1 = nn.CrossEntropyLoss(ignore_index = 255, reduction='none')
            loss = ce_criterion(logits, labels)
            loss1 = ce_criterion1(logits, labels)
            for i in range(100):
                print('l'+str(i*12), logits[i*12])
                print(labels[i*12])
                print('cel'+str(i*12), loss1[i*12])
            print(loss1)
            print('loss', loss)
        else: 
            ce_criterion = nn.CrossEntropyLoss(ignore_index = 255, reduction='none')
            loss = ce_criterion(logits, labels) 
            loss = torch.mean(loss * pixelWiseWeight)
            print('loss1', loss)
        
        return loss