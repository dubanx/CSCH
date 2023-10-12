import torch
import torch.nn.functional as F
from models.cossim import CosSim
from functions.loss.base_cls import BaseClassificationLoss

def get_imbalance_mask(sigmoid_logits, labels, nclass, threshold=0.7, imbalance_scale=-1):
    if imbalance_scale == -1:
        imbalance_scale = 1 / nclass

    mask = torch.ones_like(sigmoid_logits) * imbalance_scale

    # wan to activate the output
    mask[labels == 1] = 1

    # if predicted wrong, and not the same as labels, minimize it
    correct = (sigmoid_logits >= threshold) == (labels == 1)
    mask[~correct] = 1

    multiclass_acc = correct.float().mean()

    # the rest maintain "imbalance_scale"
    return mask, multiclass_acc


class CSCHLoss(BaseClassificationLoss):
    def __init__(self,
                 device,
                 ce=1,
                 s=8,
                 m=0.2,
                 m_type='cos',  # cos/arc
                 multiclass=False,
                 quan=0,
                 quan_type='cs',
                 multiclass_loss='label_smoothing',
                 **kwargs):
        super(CSCHLoss, self).__init__()
        self.device = torch.device(device)
        self.ce = ce
        self.s = s
        self.m = m
        self.m_type = m_type
        self.multiclass = multiclass

        self.quan = quan
        self.quan_type = quan_type
        self.multiclass_loss = multiclass_loss
        self.bce_criterion = torch.nn.BCELoss()
        assert multiclass_loss in ['bce', 'imbalance', 'label_smoothing', 'softmax']
 
    def compute_margin_logits(self, logits, labels):
        if self.m_type == 'cos':
            if self.multiclass:
                y_onehot = labels * self.m
                margin_logits = self.s * (logits - y_onehot)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                margin_logits = self.s * (logits - y_onehot)
        else:
            if self.multiclass:
                y_onehot = labels * self.m
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits

        return margin_logits
    
    def label2center(self, y, codebook):
        # to get sign no need to use mean, use sum here
        nclass, nbit = codebook.shape
        multi_label_random_center = torch.randint(2, (nbit,)).float().to(self.device)

        labels = torch.concat((torch.ones(y.shape[0], 1), torch.zeros(y.shape[0], y.shape[1]-1)), 1).float().to(self.device)
        mask = torch.zeros_like(y).float().to(self.device)

        hash_centers = torch.full((y.shape[0], nclass, nbit), -1).float().to(self.device)
        

        center_sum = y @ codebook
        random_center = multi_label_random_center.repeat(center_sum.shape[0], 1)
        center_sum[center_sum == 0] = random_center[center_sum == 0]
        hash_centroid = 2 * (center_sum > 0).float() - 1

        hash_centers[:, 0, :] = hash_centroid

        for i in range(y.shape[0]):
            label = y[i, :]
            hash_center = codebook[label==0]
            nlabel = hash_center.shape[0] + 1
            hash_centers[i, 1 : nlabel, :] = hash_center
            mask[i, :nlabel] = 1

        hash_centers.requires_grad_(False)

        return labels, mask, hash_centers

    def forward(self, logits, codes, labels, codebook, onehot=True):
        if self.multiclass:
            if not onehot:
                labels = F.one_hot(labels, logits.size(1))
            labels = labels.float()

            margin_logits = self.compute_margin_logits(logits, labels)

            if self.multiclass_loss in ['bce', 'imbalance']:
                # loss_ce = F.binary_cross_entropy_with_logits(margin_logits, labels, reduction='none')
                labels, mask, hash_centers = self.label2center(labels, codebook)
                centroid = hash_centers[:, 0, :]
                loss_ce = self.bce_criterion(0.5 * (codes.tanh() + 1), 0.5 * (centroid + 1))
                
                if self.multiclass_loss == 'imbalance':
                    imbalance_mask, multiclass_acc = get_imbalance_mask(torch.sigmoid(margin_logits), labels,
                                                                        labels.size(1))
                    loss_ce = loss_ce * imbalance_mask
                    loss_ce = loss_ce.sum() / (imbalance_mask.sum() + 1e-7)
                    self.losses['multiclass_acc'] = multiclass_acc
                else:
                    loss_ce = loss_ce.mean()

            elif self.multiclass_loss in ['label_smoothing']:
                log_logits = F.log_softmax(margin_logits, dim=1)
                labels_scaled = labels / labels.sum(dim=1, keepdim=True)
                loss_ce = - (labels_scaled * log_logits).sum(dim=1)
                loss_ce = loss_ce.mean()

            elif self.multiclass_loss in ['softmax']:
                import ipdb
                labels, mask, hash_centers = self.label2center(labels, codebook)
                # ipdb.set_trace()
            
                norms = torch.norm(codes, p=2, dim=-1, keepdim=True)
                nfeats = torch.div(codes, norms)

                norms_c = torch.norm(hash_centers, p=2, dim=-1, keepdim=True)
                ncenters = torch.div(hash_centers, norms_c)

                logits = []
                for i in range(nfeats.shape[0]):
                    nfeat = nfeats[i, :]
                    ncenter = ncenters[i, :]
                    logit = torch.matmul(nfeat, torch.transpose(ncenter, 0, 1))
                    logits.append(logit)
                logits = torch.cat(logits).reshape(nfeats.shape[0], -1).to(self.device)

                margin_logits = self.compute_margin_logits(logits, labels)
                
                # ipdb.set_trace()


                minusINF = torch.tensor([-1e100]).float().to(self.device)
                zero = torch.tensor([0]).float().to(self.device)

                margin_logits = torch.where(mask == 1, margin_logits, minusINF[0])

                # ipdb.set_trace()
                log_logits = F.log_softmax(margin_logits, dim=1)
                # ipdb.set_trace()
                log_logits = torch.where(mask == 1, log_logits, zero[0])
                loss_ce = - (labels * log_logits).sum(dim=1)

                # ipdb.set_trace()

                loss_ce = loss_ce.mean()

            else:
                raise NotImplementedError(f'unknown method: {self.multiclass_loss}')
        else:
            if onehot:
                labels = labels.argmax(1)

            margin_logits = self.compute_margin_logits(logits, labels)
            loss_ce = F.cross_entropy(margin_logits, labels)

        if self.quan != 0:
            if self.quan_type == 'cs':
                quantization = (1. - F.cosine_similarity(codes, codes.detach().sign(), dim=1))
            elif self.quan_type == 'l1':
                quantization = torch.abs(codes - codes.detach().sign())
            else:  # l2
                quantization = torch.pow(codes - codes.detach().sign(), 2)

            quantization = quantization.mean()
        else:
            quantization = torch.tensor(0.).to(codes.device)

        self.losses['ce'] = loss_ce
        self.losses['quan'] = quantization
        loss = self.ce * loss_ce + self.quan * quantization
        return loss
