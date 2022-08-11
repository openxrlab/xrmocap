import logging
import numpy as np
from typing import List, Union

from .base_tracking import BaseTracking


class Perception2dTracking(BaseTracking):

    def __init__(self,
                 mview_n_person: List[int],
                 positive_weight: float = 1.0,
                 negative_weight: float = -1.0,
                 inherit_threshold: float = 1.0,
                 verbose: bool = False,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """3D identity tracking based on tracked 2D perception data. This
        method assigns identities according to tracked 2D IDs in last frame.

        If positive_weight * (number of matched 2d views) +
        negative_weight * (number of mis-matched 2d views) >=
        inherit_threshold, the identity of last frame will be inherited.

        Args:
            mview_n_person (List[int]):
                A list of multi-view person count. In i-th view,
                there are mview_n_person[i] perception2d people at most.
            positive_weight (float, optional):
                Weight of matched views. Defaults to 1.0.
            negative_weight (float, optional):
                Weight of dis-matched views.. Defaults to -1.0.
            inherit_threshold (float, optional):
                Threshold of identity inherit. Defaults to 1.0.
            verbose (bool, optional):
                Whether to print individual losses during registration.
                Defaults to False.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        BaseTracking.__init__(self, verbose=verbose, logger=logger)
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.inherit_threshold = inherit_threshold
        self.mview_n_person = mview_n_person
        if self.positive_weight <= 0:
            self.logger.error(
                'Weight of a matched record should be greater than 0.')
            raise ValueError
        if self.negative_weight > 0:
            self.logger.error(
                'Weight of a dis-matched record should be no more than 0.')
            raise ValueError
        self.mview_identity_cache = []
        self.max_id = -1
        for _, n_person in enumerate(mview_n_person):
            sview_identity_cache = np.zeros(shape=(n_person))
            sview_identity_cache -= 1
            self.mview_identity_cache.append(sview_identity_cache)

    def query(self, association_list: List[List[int]],
              **kwargs: dict) -> List[int]:
        """Query identities, pass information about multi-person multi-view
        association as input, get a list of indentities.

        Args:
            association_list (List[List[int]]):
                A nested list of association result,
                in shape [n_person, n_view], and
                association_list[i][j] = k means
                the k-th 2D perception in view j
                is a 2D obersevation of person i.
        kwargs:
            Keyword args to be ignored.

        Returns:
            List[int]:
                A list of indentities, whose length
                is equal to len(association_list).
        """
        ret_list = []
        for _, mview_association in enumerate(association_list):
            mview_identities = self.__get_mview_identities__(mview_association)
            mview_identities_np = np.array(mview_identities)
            identities_sum = np.sum(mview_identities_np)
            # if cache is empty, assign a new id
            if identities_sum == (-1 * len(mview_identities)):
                self.max_id += 1
                ret_list.append(self.max_id)
                self.__set_mview_identities__(mview_association, self.max_id)
            # else, find the best identity and check threshold
            else:
                valid_idxs = np.where(mview_identities_np >= 0)[0]
                valid_identities = mview_identities_np[valid_idxs]
                identity_counts = np.bincount(valid_identities)
                best_identity = np.argmax(identity_counts)
                positive_count = len(
                    np.where(mview_identities_np == best_identity)[0])
                negative_count = len(valid_idxs) - positive_count
                score = self.positive_weight * positive_count + \
                    self.negative_weight * negative_count
                # inherit the best identity
                if score >= self.inherit_threshold:
                    ret_list.append(best_identity)
                    self.__set_mview_identities__(mview_association,
                                                  best_identity)
                # the best identity is not good enough, assign a new one
                else:
                    self.max_id += 1
                    ret_list.append(self.max_id)
                    self.__set_mview_identities__(mview_association,
                                                  self.max_id)
        return ret_list

    def __get_mview_identities__(self,
                                 mview_association: List[int]) -> List[int]:
        """Get multi-view indentities of a single 3d person from cache.

        Args:
            mview_association (List[int]):
                A list of oberservation index in each view.

        Returns:
            List[int]:
                A list of cached identities in each view.
                If there's no cache, or no obervation,
                ret_list[view_idx] will be -1.
        """
        mview_identities = []
        for view_idx, ob_idx in enumerate(mview_association):
            if np.isnan(ob_idx):
                mview_identities.append(-1)
            else:
                sview_identity = \
                    self.mview_identity_cache[view_idx][ob_idx]
                if sview_identity >= 0:
                    mview_identities.append(sview_identity)
                else:
                    mview_identities.append(-1)
        return mview_identities

    def __set_mview_identities__(self, mview_association: List[int],
                                 identity: int) -> None:
        """Set multi-view indentities of a single 3d person to cache.

        Args:
            mview_association (List[int]):
                A list of oberservation index in each view.
            identity (int):
                The identity of this 3d person.
        """
        for view_idx, ob_idx in enumerate(mview_association):
            if np.isnan(ob_idx):
                continue
            else:
                self.mview_identity_cache[view_idx][ob_idx] = identity
