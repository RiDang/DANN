import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.classes = torch.arange(self.num_classes).long().cuda()
        
    def forward(self, x, labels,weight):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        up = torch.mm(x, weight.t())
        down1  = torch.sqrt(torch.pow(x,2).sum(dim=1, keepdim=True)).expand(batch_size, self.num_classes)
        down2 = torch.sqrt(torch.pow(weight, 2).sum(dim=1, keepdim=True)).expand(self.num_classes, batch_size).t()
        distmat = up/(down1*down2+1e-4)
        #distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
        #          torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        #distmat.addmm_(1, -2, x, self.centers.t())

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(self.classes.expand(batch_size, self.num_classes))      # 这个mask 生成对赢得标签。

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return -loss



