import logging
import numpy as np
import prettytable
import time
import torch
from typing import List, Tuple, Union

from xrmocap.matching.match_solver import proj2dpam
from xrmocap.utils.log_utils import get_logger
from xrmocap.utils.mvpose_utils import geometry_affinity
from .base_matching import BaseMatching


class MultiWayMatching(BaseMatching):

    def __init__(self,
                 lambda_SVT: int,
                 alpha_SVT: float,
                 use_dual_stochastic_SVT=False,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Multi-way matching with cycle consistency.

        Args:
            lambda_SVT (int): Lambda constants in SVT algorithms.
            alpha_SVT (float): Alpha constants in SVT algorithms.
            use_dual_stochastic_SVT (bool, optional): Use dual stochastic in
                SVT algorithms. Defaults to False.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        super().__init__(logger=logger)
        self.logger = logger
        self.lambda_SVT = lambda_SVT
        self.alpha_SVT = alpha_SVT
        self.use_dual_stochastic_SVT = use_dual_stochastic_SVT

    def __call__(self,
                 mview_kps2d: np.ndarray,
                 image_tensor: torch.Tensor,
                 extractor,
                 Fs: torch.Tensor,
                 affinity_type: str,
                 n_kps2d: int,
                 dim_group: Union[torch.Tensor, None] = None,
                 not_matched_dim_group: Union[list, None] = None,
                 not_matched_index: Union[list, None] = None,
                 rerank: bool = False) -> Tuple[List[np.ndarray], np.ndarray]:
        """Match people id from different cameras.

        Args:
            mview_kps2d (np.ndarray): Keypoints points in shape (sum, 17, 2).
                sum = total number of people detected from all cameras.
            image_tensor (torch.Tensor): Image tensor.
            extractor: AppearanceAffinityEstimator object.
            Fs (torch.Tensor): The camera F matrix.
            affinity_type (str): The affinity type.
            n_kps2d (int): The number of keypoints considered in triangulate.
            dim_group (Union[torch.Tensor, None], optional): Defaults to None.
            not_matched_dim_group (Union[list, None], optional):
                Defaults to None.
            not_matched_index (Union[list, None], optional): Defaults to None.
            rerank (bool, optional): Defaults to False.

        Raises:
            NotImplementedError: The affinity type is wrong.

        Returns:
            Tuple[List[np.ndarray], np.ndarray]:
            matched_list: The id of the matched people in different cameras.
                M = len(matched_list), and M is defined as the maximum number
                of people captured by two or more cameras.
            sub_imgid2cam: The camera label of the captured people.
        """
        if not_matched_dim_group is not None and not_matched_index is not None:
            dim_group = not_matched_dim_group
            image_tensor = image_tensor[not_matched_index]
        mview_kps2d[np.isnan(mview_kps2d)] = 1e-9

        # step1. estimate matching matrix with geometry affinity
        # or appearance affinity matrix
        affinity_mat = extractor.get_affinity(
            image_tensor, rerank=rerank).cpu()
        if rerank:
            affinity_mat = torch.max(affinity_mat, affinity_mat.t())
            affinity_mat = 1 - affinity_mat
        geo_affinity_mat = geometry_affinity(
            mview_kps2d, Fs.numpy(), dim_group, n_kps2d=n_kps2d)
        geo_affinity_mat = torch.from_numpy(geo_affinity_mat)

        # step2. calculate the hybrid affinity matrix
        if affinity_type == 'geometry_mean':
            self.W = torch.sqrt(affinity_mat * geo_affinity_mat)
        elif affinity_type == 'circle':
            self.W = torch.sqrt((affinity_mat**2 + geo_affinity_mat**2) / 2)
        elif affinity_type == 'ReID only':
            self.W = affinity_mat
        else:
            raise NotImplementedError('Other types of affinity evaluation are'
                                      'not yet implemented.')
        self.W[torch.isnan(self.W)] = 0
        # step3. multi-way matching with cycle consistency
        match_mat, bin_match = self.__class__.match_SVT(
            self.W,
            dim_group,
            self.logger,
            alpha=self.alpha_SVT,
            _lambda=self.lambda_SVT,
            dual_stochastic=self.use_dual_stochastic_SVT)
        sub_imgid2cam = np.zeros(mview_kps2d.shape[0], dtype=np.int32)
        n_cameras = len(dim_group) - 1
        for idx, i in enumerate(range(n_cameras)):
            sub_imgid2cam[dim_group[i]:dim_group[i + 1]] = idx

        matched_list = [[] for _ in range(bin_match.shape[1])]
        for sub_imgid, row in enumerate(bin_match):
            if row.sum() != 0:
                # pid = row.double().argmax()
                pid = np.where(row.double() > 0)[0]
                if len(pid) == 1:
                    for pid_ in pid:
                        matched_list[pid_].append(sub_imgid)
        matched_list = [np.array(i) for i in matched_list]

        return matched_list, sub_imgid2cam

    @classmethod
    def transform_closure(cls, X_bin):
        """Convert binary relation matrix to permutation matrix.

        :param X_bin: torch.tensor which is binarized by a threshold
        :return:
        """
        temp = torch.zeros_like(X_bin)
        N = X_bin.shape[0]
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    temp[i][j] = X_bin[i, j] or (X_bin[i, k] and X_bin[k, j])
        vis = torch.zeros(N)
        match_mat = torch.zeros_like(X_bin)
        for i, row in enumerate(temp):
            if vis[i]:
                continue
            for j, is_relative in enumerate(row):
                if is_relative:
                    vis[j] = 1
                    match_mat[j, i] = 1
        return match_mat

    @classmethod
    def match_SVT(cls,
                  affinity_mat: torch.Tensor,
                  dim_group: torch.Tensor,
                  logger: Union[None, str, logging.Logger] = None,
                  alpha=0.1,
                  _lambda=50,
                  dual_stochastic=True,
                  p_select: int = 1,
                  tol=5e-4,
                  max_iter=500,
                  verbose=False,
                  eigenvalues=False,
                  mu=64) -> torch.Tensor:
        """Multi-way matching with cycle consistency.

        Args:
            affinity_mat (torch.Tensor): Affinity matrix
            dim_group (torch.Tensor): The cumulative number of people from
                                    different perspectives.
            logger (Union[None, str, logging.Logger], optional):
                Defaults to None.
            alpha (float, optional): Defaults to 0.1.
            _lambda (int, optional): Defaults to 50.
            dual_stochastic (bool, optional): Defaults to True.
            p_select (int, optional): Defaults to 1.
            tol (float, optional): Defaults to 5e-4.
            max_iter (int, optional): Defaults to 500.
            verbose (bool, optional): Defaults to False.
            eigenvalues (bool, optional): Defaults to False.
            mu (int, optional): Defaults to 64.

        Returns:
            match_mat (torch.Tensor): Match matrix in shape (N, N),
                N = n1+n2+..., n1 is the number of detected people in cam1.
                If two people match, we set it to True, otherwise to False,
                and we also set the redundancy information to False.

            bin_match (torch.Tensor): Minimum set of successful matches.
                When more than two cameras capture a person at the same time,
                they are matched. bin_match in shape(N, M), M is defined as
                the maximum number of people captured by two or more cameras
                simultaneously.
        """
        logger = get_logger(logger)
        if verbose:
            logger.info(
                'SVT-Matching: alpha = %.2f, p_select = %.2f, _lambda = %.2f\n'
                % (alpha, p_select, _lambda))
        info = dict()
        N = affinity_mat.shape[0]
        affinity_mat[torch.arange(N), torch.arange(N)] = 0
        affinity_mat = (affinity_mat + affinity_mat.t()) / 2
        X = affinity_mat.clone()
        Y = torch.zeros_like(affinity_mat)
        W = alpha - affinity_mat
        t0 = time.time()
        for iter_ in range(max_iter):

            X0 = X
            # update Q with SVT
            U, s, V = torch.svd(1.0 / mu * Y + X)
            diagS = s - _lambda / mu
            diagS[diagS < 0] = 0
            Q = U @ diagS.diag() @ V.t()
            # update X
            X = Q - (W + Y) / mu
            # project X
            for i in range(len(dim_group) - 1):
                ind1, ind2 = dim_group[i], dim_group[i + 1]
                X[ind1:ind2, ind1:ind2] = 0
            if p_select == 1:
                X[torch.arange(N), torch.arange(N)] = 1
            X[X < 0] = 0
            X[X > 1] = 1

            if dual_stochastic:
                # Projection for double stochastic constraint
                for i in range(len(dim_group) - 1):
                    row_begin, row_end = int(dim_group[i]), int(dim_group[i +
                                                                          1])
                    for j in range(len(dim_group) - 1):
                        col_begin = int(dim_group[j])
                        col_end = int(dim_group[j + 1])
                        if row_end > row_begin and col_end > col_begin:
                            X[row_begin:row_end,
                              col_begin:col_end] = proj2dpam(
                                  X[row_begin:row_end, col_begin:col_end],
                                  1e-2)

            X = (X + X.t()) / 2
            # update Y
            Y = Y + mu * (X - Q)
            # test if convergence
            p_res = torch.norm(X - Q) / N
            d_res = mu * torch.norm(X - X0) / N
            if verbose:
                logger.info(
                    f'Iter = {iter_}, Res = ({p_res}, {d_res}), mu = {mu}')

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
            table.add_row(
                [info['time'], info['iter'], f'{p_res}, {d_res}', mu])
            logger.info(table)
        match_mat = cls.transform_closure(X_bin).clone().detach()

        bin_match = match_mat[:,
                              torch.nonzero(torch.sum(match_mat, dim=0) > 1.9).
                              squeeze()] > 0.9
        bin_match = bin_match.reshape(W.shape[0], -1)
        return match_mat, bin_match
