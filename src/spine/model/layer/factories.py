"""Factories to build generic layers components."""

from torch import nn

from spine.model.experimental.bayes.evidential import EDLRegressionLoss, EVDLoss
from spine.utils.factory import instantiate, module_dict

from .cnn.encoder import SparseResidualEncoder
from .common import final, losses, metric

__all__ = ["loss_fn_factory", "metric_fn_fatory", "encoder_factory", "final_factory"]


def loss_fn_factory(cfg, functional=False, **kwargs):
    """Instantiates a loss function from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Final layer configuration
    functional : bool, default False
        Whether to return the loss function as a functional
    **kwargs : dict, optional
        Additional parameters to pass to the loss function

    Returns
    -------
    object
        Instantiated loss function
    """

    loss_dict = {
        'ce': nn.CrossEntropyLoss,
        'bce': nn.BCELoss,
        'bce_logits': nn.BCEWithLogitsLoss,
        'mm': nn.MultiMarginLoss,
        'huber': nn.HuberLoss,
        'l1': nn.L1Loss,
        'l2': nn.MSELoss,
        'mse': nn.MSELoss,
        'evd': EVDLoss, # TODO move
        'edl': EDLRegressionLoss, # TODO move
        'ce_dice': CE_DICE_Loss,
        'focal_loss': FocalLoss,
        **module_dict(losses)
    }

    loss_dict_func = {
        'ce': nn.functional.cross_entropy,
        'bce': nn.functional.binary_cross_entropy,
        'bce': nn.functional.binary_cross_entropy_with_logits,
        'mm': nn.functional.multi_margin_loss,
        'huber': nn.functional.huber_loss,
        'l1': nn.functional.l1_loss,
        'l2': nn.functional.mse_loss,
        'mse': nn.functional.mse_loss,
    }

    if not functional:
        if isinstance(cfg, dict) and cfg.get('name') == 'ce_dice':
            # Extract CE_DICE specific parameters
            lambda_dice = cfg.get('lambda_dice', 0.5)
            return CE_DICE_Loss(
                reduction=kwargs.get('reduction', 'none'),
                lambda_dice=lambda_dice
            )
        elif isinstance(cfg, dict) and cfg.get('name') == 'focal_loss':
            # Extract focal loss specific parameters
            focal_gamma = cfg.get('focal_gamma', 2)
            return FocalLoss(
                reduction=kwargs.get('reduction', 'none'),
                gamma=focal_gamma
            )
        else:
            # For all other losses, use the standard instantiation
            return instantiate(loss_dict, cfg, **kwargs)

    else:
        assert (isinstance(cfg, str) or
                ('name' in cfg and len(cfg) == 1)), (
            "For a functional, only provide the function name.")

        name = cfg if isinstance(cfg, str) else cfg['name']
        if name == 'ce_dice':
            ce_dice_instance = CE_DICE_Loss(**kwargs)
            return lambda pred, target: ce_dice_instance(pred, target)
        elif name == "focal_loss":
            focal_loss_instance = FocalLoss(**kwargs)
            return lambda pred, target: focal_loss_instance(pred, target)
        try:
            return loss_dict_func[name]
        except KeyError as err:
            raise KeyError(f"Could not find the functional {name} in the "
                           f"available list: {list(loss_dict_func.keys())}")



def metric_fn_factory(cfg):
    """Instantiates a metric function from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Metric function configuration

    Returns
    -------
    object
        Instantiated metric function
    """
    metric_layers = module_dict(metric)
    return instantiate(metric_layers, cfg)


def encoder_factory(cfg):
    """Instantiates an image encoder from a configuration dictionary.

    Parameters
    ----------
    cfg : dict
        Encoder configuration

    Returns
    -------
    object
        Instantiated encoder
    """
    encoder_dict = {"cnn": SparseResidualEncoder}

    return instantiate(encoder_dict, cfg)


def final_factory(in_channels, **cfg):
    """Instantiates a final layer from a configuration dictionary.

    Parameters
    ----------
    in_channels : int
        Number of features input into the final layer
    **cfg : dict
        Final layer configuration

    Returns
    -------
    object
        Instantiated final layer
    """
    final_layers = module_dict(final)
    return instantiate(final_layers, cfg, in_channels=in_channels)
