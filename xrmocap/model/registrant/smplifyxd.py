import torch

from .smplifyx import SMPLifyX


class SMPLifyXD(SMPLifyX):
    """Re-implementation of SMPLify-X with displacement."""
    OPTIM_PARAM = SMPLifyX.OPTIM_PARAM + [
        'displacement',
    ]

    def __prepare_optimizable_parameters__(self, init_dict: dict,
                                           batch_size: int) -> dict:
        """Prepare optimizable parameters in batch for registrant. If some of
        the parameters can be found in init_dict, use them for initialization.

        Args:
            init_dict (dict):
                A dict of init parameters. init_dict.keys() is a
                sub-set of self.__class__.OPTIM_PARAM.
            batch_size (int)

        Returns:
            dict:
                A dict of optimizable parameters, whose keys are
                self.__class__.OPTIM_PARAM and values are
                Tensors in batch.
        """
        smplx_init_dict = init_dict.copy()
        init_displacement = smplx_init_dict.pop('displacement', None)
        OPTIM_PARAM_backup = self.__class__.OPTIM_PARAM
        self.__class__.OPTIM_PARAM = SMPLifyX.OPTIM_PARAM
        ret_dict = SMPLifyX.__prepare_optimizable_parameters__(
            self, init_dict=smplx_init_dict, batch_size=batch_size)
        self.__class__.OPTIM_PARAM = OPTIM_PARAM_backup
        default_displacement = torch.zeros(
            size=(1, self.body_model.NUM_VERTS, 3),
            dtype=self.body_model.betas.dtype,
            device=self.device,
            requires_grad=True)
        displacement = self.__match_init_batch_size__(
            init_param=init_displacement,
            default_param=default_displacement,
            batch_size=batch_size)
        ret_dict['displacement'] = displacement
        return ret_dict
