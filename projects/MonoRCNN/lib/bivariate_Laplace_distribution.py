import math
import torch

# https://search.r-project.org/CRAN/refmans/LaplacesDemon/html/dist.Multivariate.Laplace.html
def bivariate_Laplace_loss(pred_Hs, pred_hrecs, pred_covs, gt_Hs, gt_hrecs):
    pred_L_00s = pred_covs[:, 0:1]
    pred_L_11s = pred_covs[:, 1:2]
    pred_L_10s = pred_covs[:, 2:3]

    pred_Ls = torch.cuda.FloatTensor(pred_Hs.shape[0], 2, 2)
    pred_Ls[:, 0:1, 0:1] = pred_L_00s.exp().unsqueeze(-1)
    pred_Ls[:, 0:1, 1:2] = 0
    pred_Ls[:, 1:2, 0:1] = pred_L_10s.unsqueeze(-1)
    pred_Ls[:, 1:2, 1:2] = pred_L_11s.exp().unsqueeze(-1)
    pred_L_tranposes = torch.transpose(pred_Ls, 1, 2)
    pred_sigma_invs = torch.bmm(pred_Ls, pred_L_tranposes)
    pred_sigma_dets = 1.0 / (2.0 * (pred_L_00s + pred_L_11s)).exp()
    
    pred_deltas = torch.cuda.FloatTensor(pred_Hs.shape[0], 2, 1)
    pred_deltas[:, 0:1, 0]  = pred_Hs - gt_Hs
    pred_deltas[:, 1:2, 0]  = pred_hrecs - gt_hrecs
    pred_muls = torch.bmm(torch.bmm(torch.transpose(pred_deltas, 1, 2), pred_sigma_invs), pred_deltas).squeeze(-1)
    
    loss_dis_0 = -torch.log(1.0 / (math.pi * pred_sigma_dets**0.5))
    loss_dis_1 = -0.5 * torch.log(math.pi / (2 * (2 * pred_muls)**0.5))
    loss_dis_2 = (2 * pred_muls)**0.5
    loss_dis = loss_dis_0 + loss_dis_1 + loss_dis_2
    return loss_dis

def bivariate_Laplace_cov(pred_covs):
    pred_L_00s = pred_covs[:, 0:1]
    pred_L_11s = pred_covs[:, 1:2]
    pred_L_10s = pred_covs[:, 2:3]

    pred_Ls = torch.cuda.FloatTensor(pred_covs.shape[0], 2, 2)
    pred_Ls[:, 0:1, 0:1] = pred_L_00s.exp().unsqueeze(-1)
    pred_Ls[:, 0:1, 1:2] = 0
    pred_Ls[:, 1:2, 0:1] = pred_L_10s.unsqueeze(-1)
    pred_Ls[:, 1:2, 1:2] = pred_L_11s.exp().unsqueeze(-1)
    pred_L_tranposes = torch.transpose(pred_Ls, 1, 2)

    pred_sigma_invs = torch.bmm(pred_Ls, pred_L_tranposes)
    pred_sigmas = torch.inverse(pred_sigma_invs.cpu()).cuda()
    return pred_sigmas
    