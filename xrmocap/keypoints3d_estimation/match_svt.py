import logging
import os
import sys
import time
from typing import Union

import prettytable
import torch

from xrmocap.keypoints3d_estimation.lib.pictorial import transform_closure
from xrmocap.keypoints3d_estimation.match_solver import myproj2dpam
from xrmocap.utils.log_utils import get_logger

# Config project if not exist
project_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
if project_path not in sys.path:
    sys.path.insert(0, project_path)


def match_SVT(S: torch.Tensor,
              dimGroup: torch.Tensor,
              logger: Union[None, str, logging.Logger] = None,
              alpha=0.1,
              _lambda=50,
              dual_stochastic=True,
              **kwargs) -> torch.Tensor:
    """Multi-way matching with cycle consistency.

    Args:
        S (torch.Tensor): Affinity matrix
        dimGroup (torch.Tensor): The cumulative number of people from
                                 different perspectives.
        logger (Union[None, str, logging.Logger], optional): Defaults to None.
        alpha (float, optional): Defaults to 0.1.
        _lambda (int, optional): Defaults to 50.
        dual_stochastic (bool, optional): Defaults to True.

    Returns:
        match_mat (torch.Tensor): Match matrix in shape (N, N), N = n1+n2+...,
            n1 is the number of detected people in cam1. If two people match,
            we set it to True, otherwise to False, and we also set the
            redundancy information to False.

        bin_match (torch.Tensor): Minimum set of successful matches.
            When more than two cameras capture a person at the same time, they
            are matched. bin_match in shape(N, M), M is defined as the maximum
            number of people captured by two or more cameras simultaneously.
    """
    pSelect = kwargs.get('pselect', 1)
    tol = kwargs.get('tol', 5e-4)
    maxIter = kwargs.get('maxIter', 500)
    verbose = kwargs.get('verbose', False)
    eigenvalues = kwargs.get('eigenvalues', False)
    mu = kwargs.get('mu', 64)
    logger = get_logger(logger)
    if verbose:
        logger.info(
            'Run SVT-Matching: alpha = %.2f, pSelect = %.2f _lambda = %.2f \n'
            % (alpha, pSelect, _lambda))
    info = dict()
    N = S.shape[0]
    S[torch.arange(N), torch.arange(N)] = 0
    S = (S + S.t()) / 2
    X = S.clone()
    Y = torch.zeros_like(S)
    W = alpha - S
    t0 = time.time()
    for iter_ in range(maxIter):

        X0 = X
        # update Q with SVT
        U, s, V = torch.svd(1.0 / mu * Y + X)
        diagS = s - _lambda / mu
        diagS[diagS < 0] = 0
        Q = U @ diagS.diag() @ V.t()
        # update X
        X = Q - (W + Y) / mu
        # project X
        for i in range(len(dimGroup) - 1):
            ind1, ind2 = dimGroup[i], dimGroup[i + 1]
            X[ind1:ind2, ind1:ind2] = 0
        if pSelect == 1:
            X[torch.arange(N), torch.arange(N)] = 1
        X[X < 0] = 0
        X[X > 1] = 1

        if dual_stochastic:
            # Projection for double stochastic constraint
            for i in range(len(dimGroup) - 1):
                row_begin, row_end = int(dimGroup[i]), int(dimGroup[i + 1])
                for j in range(len(dimGroup) - 1):
                    col_begin, col_end = int(dimGroup[j]), int(dimGroup[j + 1])
                    if row_end > row_begin and col_end > col_begin:
                        X[row_begin:row_end, col_begin:col_end] = myproj2dpam(
                            X[row_begin:row_end, col_begin:col_end], 1e-2)

        X = (X + X.t()) / 2
        # update Y
        Y = Y + mu * (X - Q)
        # test if convergence
        p_res = torch.norm(X - Q) / N
        d_res = mu * torch.norm(X - X0) / N
        if verbose:
            logger.info(f'Iter = {iter_}, Res = ({p_res}, {d_res}), mu = {mu}')

        if p_res < tol and d_res < tol:
            break

        if p_res > 10 * d_res:
            mu = 2 * mu
        elif d_res > 10 * p_res:
            mu = mu / 2

    X = (X + X.t()) / 2
    info['time'] = time.time() - t0
    info['iter'] = iter_

    if eigenvalues:
        info['eigenvalues'] = torch.eig(X)

    X_bin = X > 0.5
    if verbose:
        table = prettytable.PrettyTable()
        table.title = 'Alg terminated.'
        table.field_names = ['Time', 'Iter', 'Res', 'mu']
        table.add_row([info['time'], info['iter'], f'{p_res}, {d_res}', mu])
        logger.info(table)
    match_mat = torch.tensor(transform_closure(X_bin.numpy()))

    bin_match = match_mat[:,
                          torch.nonzero(torch.sum(match_mat, dim=0) > 1.9).
                          squeeze()] > 0.9
    bin_match = bin_match.reshape(W.shape[0], -1)
    return match_mat, bin_match
