"""This file is pytorch implementation of :

Wang, Qianqian, Xiaowei Zhou, and Kostas Daniilidis. "Multi-Image Semantic
Matching by Mining Consistent Features." arXiv preprint arXiv:1711.07641(2017).
"""
import torch


def proj2dpam(Y, tol=1e-4):
    X0 = Y
    X = Y
    I2 = 0

    for iter_ in range(10):
        X1 = project((X0 + I2), 0)
        I1 = X1 - (X0 + I2)
        X2 = project((X0 + I1), 1)
        I2 = X2 - (X0 + I1)

        chg = torch.sum(torch.abs(X2[:] - X[:])) / X.numel()
        X = X2
        if chg < tol:
            return X
    return X


def project(X, dim_to_project):
    if dim_to_project == 0:
        for i in range(X.shape[0]):
            X[i, :] = proj2pav(X[i, :])
    elif dim_to_project == 1:
        for j in range(X.shape[1]):
            X[:, j] = proj2pav(X[:, j])
    else:
        return None
    return X


def proj2pav(y):
    y[y < 0] = 0
    x = torch.zeros_like(y)
    if torch.sum(y) < 1:
        x += y
    else:
        u, _ = torch.sort(y, descending=True)
        sv = torch.cumsum(u, 0)
        to_find = u > (sv - 1) / (
            torch.arange(1, len(u) + 1, device=u.device, dtype=u.dtype))
        rho = torch.nonzero(to_find.reshape(-1))[-1]
        theta = torch.max(
            torch.tensor(0, device=sv.device, dtype=sv.dtype),
            (sv[rho] - 1) / (rho.float() + 1))
        x += torch.max(y - theta,
                       torch.tensor(0, device=sv.device, dtype=y.dtype))
    return x
