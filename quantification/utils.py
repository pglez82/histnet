import numpy as np


def absolute_error(p_true, p_pred):
    """Just the absolute difference between both prevalences.

        Parameters
        ----------
        p_true : array_like, shape=(n_classes)
            True prevalences. In case of binary quantification, this parameter could be a single float value.

        p_pred : array_like, shape=(n_classes)
            Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
        """
    return np.abs(p_pred - p_true)


def relative_absolute_error(p_true, p_pred, eps=1e-12):
    """ A binary relative version of the absolute error

        It is the relation between the absolute error and the true prevalence.

            :math:`rae = | \hat{p} - p | / p`

        Parameters
        ----------
        p_true : float
            True prevalence for the positive class

        p_pred : float
            Predicted prevalence for the positive class

        eps : float, (default=1e-12)
            To prevent a division by zero exception

        Returns
        -------
        relative_absolute_error: float
            It is equal to :math:`| \hat{p} - p | / p`
    """
    if p_true == 0:
        return np.abs(p_pred - p_true) / (p_true + eps)
    else:
        return np.abs(p_pred - p_true) / p_true

def binary_kld(p_true, p_pred, eps=1e-12):
    """ A binary version of the Kullback - Leiber divergence (KLD)

            :math:`kld = p \cdot \log(p/\hat{p}) + (1-p) \cdot \log((1-p)/(1-\hat{p}))

        Parameters
        ----------
        p_true : array_like, shape = (n_classes)
            True prevalences

        p_pred : array_like, shape = (n_classes)
            Predicted prevalences.

        eps : float, (default=1e-12)
            To prevent a division by zero exception

        Returns
        -------
        KLD: float
            It is equal to :math:`p \cdot \log(p/\hat{p}) + (1-p) \cdot \log((1-p)/(1-\hat{p}))`
    """
    if p_pred == 0:
        kld = p_true * np.log2(p_true / eps)
    else:
        kld = p_true * np.log2(p_true / p_pred)
    if p_pred == 1:
        kld = kld + (1 - p_true) * np.log2((1 - p_true) / eps)
    else:
        kld = kld + (1 - p_true) * np.log2((1 - p_true) / (1 - p_pred))
    return kld

