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
        mask = (labels != self.cfg.INPUT.IGNORE_LABEL)
        print("feat", feat.size())
        labels = labels[mask]
        feat = feat[mask]

        feat = F.normalize(feat, p=2, dim=1)
        Proto = F.normalize(Proto, p=2, dim=1)
        print("feat1", feat.size())
        logits = feat.mm(Proto.permute(1, 0).contiguous())
        logits = logits / self.cfg.MODEL.CONTRAST.TAU

        print("logits", logits.size())
        if pixelWiseWeight is None:
            ce_criterion = nn.CrossEntropyLoss()
            loss = ce_criterion(logits, labels) 
        else: 
            ce_criterion = nn.CrossEntropyLoss(reduction='none')
            loss = ce_criterion(logits, labels) 
            print(loss.size())
            loss = torch.mean(loss * pixelWiseWeight)
        
        return loss