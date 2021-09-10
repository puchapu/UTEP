# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from utils.metric import binary_accuracy
from network import WarmStartGradientReverseLayer


def bias_loss(src_variance):
    return torch.mean(src_variance**2)

def NBCEWLoss(input_,od):
    batch_size = od.size(1)//2
    od = 1. - od
    
    od1 = torch.ones(1, batch_size).squeeze(0).cuda()
    od2 = od[0,batch_size: 2 * batch_size]
    od = torch.cat((od1,od2),dim=0).view(-1,1)

    mask = input_.ge(0.7)
    mask_hw = input_.le(0.2)
    od = od.expand_as(mask)
    od_out = torch.masked_select(od, mask).detach()
    od_out1 = torch.masked_select(od, mask_hw).detach()
    mask_out = torch.masked_select(input_, mask) * od_out
    mask_out1 = torch.masked_select(input_, mask_hw) * od_out1
    if len(mask_out) and not len(mask_out1):
        entropy = -(torch.sum(mask_out * torch.log(mask_out))) / float(mask_out.size(0))
        return entropy
    elif len(mask_out1) and not len(mask_out):
        entropy1 = -(torch.sum((1-mask_out1) * torch.log(1-mask_out1))) / float(mask_out1.size(0))
        return entropy1
    elif len(mask_out1) and len(mask_out):
        entropy = -(torch.sum(mask_out * torch.log(mask_out))) / float(mask_out.size(0))
        entropy1 = -(torch.sum((1 - mask_out1) * torch.log(1 - mask_out1))) / float(mask_out1.size(0))
        return entropy + entropy1
    else:
        return torch.tensor(0.0).cuda()

class DomainAdversarialLoss(nn.Module):
    """
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) = \mathbb{E}_{x_i^s \sim \mathcal{D}_s} log[D(f_i^s)]
            + \mathbb{E}_{x_j^t \sim \mathcal{D}_t} log[1-D(f_j^t)].

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features. Its input shape is (N, F) and output shape is (N, 1)
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        grl (WarmStartGradientReverseLayer, optional): Default: None.

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
        - w_s (tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (tensor, optional): a rescaling weight given to each instance from target domain.

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.

    Examples::

        >>> from dalib.modules.domain_discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> # If you want to assign different weights to each instance, you should pass in w_s and w_t
        >>> w_s, w_t = torch.randn(20), torch.randn(20)
        >>> output = loss(f_s, f_t, w_s, w_t)
    """

    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional = None):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))
        w_s = 1 + w_s
        w_t = 1 + w_t
        if w_s is None:
            w_s = torch.ones_like(d_label_s)
        if w_t is None:
            w_t = torch.ones_like(d_label_t)
        return 0.5 * (self.bce(d_s, d_label_s, w_s.view_as(d_s)) + self.bce(d_t, d_label_t, w_t.view_as(d_t)))
