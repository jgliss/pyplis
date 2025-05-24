import pytest
import numpy as np
from pyplis import optimisation as mod
from pyplis.model_functions import multi_gaussian_no_offset

def create_multigauss_testdata(x, params, offset, add_noise, noise_frac = 0.01):
    """Create test data set containing multiple Gaussians.

    Parameters
    ----------
    x : ndarray
        index values for the signal
    params : list
        list of parameters for the Gaussians, each Gaussian is defined by
        3 parameters: [amplitude, position, width]
    offset : float
        offset to be added to the signal
    add_noise : bool
        add noise to test data
    noise_frac : float
        determines noise amplitude (fraction relative to max amplitude of
            Gaussian), defaults to 0.01 (1%). Only used if `add_noise` is True.

    Returns
    -------
    tuple
        2-element tuple containing

        - the signal (y)
        - the index (x)
    """
    y = multi_gaussian_no_offset(x, *params) + offset
    if add_noise:
        y = y + max(y) * noise_frac * np.random.normal(0, 1, size=len(x))
    return y

def create_noise_dataset(x):
    """Make pure noise dataset"""
    return 5 * np.random.normal(0, 1, size=len(x))

X = np.linspace(0, 400, 401)

@pytest.mark.parametrize(
    "y,x,mu_main_peak,sigma_main_peak,integral_main_peak,num_add_gaussians",
    [
        pytest.param(
            create_noise_dataset(X),
            X, 
            np.nan, np.nan, np.nan, 0,
            id="pure noise"),
        
        pytest.param(
            create_multigauss_testdata(X,
            [100, 200, 20],
            offset=0,
            add_noise=False),
            X,
            200, 20, 5013.3, 0,
            id="single peak without noise"),

        pytest.param(
            create_multigauss_testdata(X,
            [
                # Amplitude, mu, sigma (5 Gaussians)
                150, 30, 8,
                200, 110, 3,
                300, 150, 20,
                75, 370, 40,
                300, 250, 1
            ],
            offset=30,
            add_noise=False),
            X,
            146.36, 22.27, 16534, 3,
            id="5 Gaussians with noise"),
    ]
)
def test_multigauss_fit(y,x, mu_main_peak, sigma_main_peak, integral_main_peak, num_add_gaussians):
    f = mod.MultiGaussFit(data=y, index=x, do_fit=True)
    mu,sigma,main_peak_integrated,add_gaussians = f.analyse_fit_result()
    assert len(add_gaussians) == num_add_gaussians
    np.testing.assert_allclose(
        actual=[mu, sigma, main_peak_integrated],
        desired=[mu_main_peak, sigma_main_peak, integral_main_peak], 
        rtol=1e-2)