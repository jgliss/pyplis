import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import rc_context
from pyplis import print_log as logger
from pyplis.model_functions import multi_gaussian_no_offset, multi_gaussian_same_offset
from pyplis.optimisation import MultiGaussFit
rc_context({'font.size': '12'})

from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, ARGPARSER, IMG_DIR

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
    return y, x


def create_noise_dataset():
    """Make pure noise dataset"""
    x = np.linspace(0, 400, 401)
    y = 5 * np.random.normal(0, 1, size=len(x))
    return y, x

def plot_result(f: MultiGaussFit, mu_main_peak_analysis, sigma_main_peak_analysis, add_gaussians):
    fig, axes = f.plot_result(True, figsize=(12, 10))
    amp_main_peak = f.get_value(mu_main_peak_analysis)
    for g in add_gaussians:
        amp_tot = g[0] + f.offset
        axes[0].annotate(
            f"Additional peak:\nμ={int(g[1])}, σ={int(g[2])}", 
            xy=(g[1], amp_tot),  # point to annotate
            xytext=(-20, 0),  # offset from point in points
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
            ha='right',  # horizontal alignment
            va='top' if amp_tot > 300 else "bottom",   # vertical alignment
            zorder=10,  # draw on top of other elements,
            fontsize=8
        )

    axes[0].annotate(
         f"Main peak:\nμ={int(mu_main_peak_analysis)}, σ={int(sigma_main_peak_analysis)}", 
                xy=(mu_main_peak_analysis, amp_main_peak),  # point to annotate
                xytext=(mu_main_peak_analysis, amp_main_peak*0.1),  # offset from point in points
                textcoords='data',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
                ha='center',  # horizontal alignment
                va='center',   # vertical alignment
                zorder=10,  # draw on top of other elements,
                fontsize=8
            )
    return fig

def main():
    """Example script to demonstrate the use of the MultiGaussFit class."""
    
    figs_to_save = []

    # Index (x axis) for the multigauss test data
    signal_index = np.linspace(0, 400, 401)
    
    # Case 1: Single Gaussian without offset
    testdata = create_multigauss_testdata(
        x=signal_index,
        params=[100, 200, 10],  # Amplitude, mu, sigma
        offset=0,
        add_noise=False)
    
    # Create MultiGaussFit object and fit the data
    # If data are provided and do_fit is True, the fit 
    # is performed automatically during initialisation
    f = MultiGaussFit(data=testdata[0], index=testdata[1], do_fit=True)
    
    # Get the number of Gaussians that were found in the data
    n_gaussians = f.num_of_gaussians
    logger.info(f"Number of Gaussians found: {n_gaussians}")

    # you can also retrieve the parameters (amp, mu, sigma) of each Gaussian 
    for gaussian in f.gaussians():
        logger.info(f"Gaussian parameters: (A={gaussian[0]:.2f}, "
                    f"μ={gaussian[1]:.2f}, σ={gaussian[2]:.2f})")
    
    # For more complex data comprising a combination of multiple 
    # overlapping or distinct peaks, you can also run a post analysis
    # on the fit result to identify the main peak (and its width) of the
    # parameterised distribution, as well as additional peaks that may be present.
    # See below for an example.
    (mu_main_peak_analysis, 
     sigma_main_peak_analysis, 
     _, 
     add_gaussians) = f.analyse_fit_result(sigma_tol_overlaps=3)
    
    # Visualise the result including the residual
    fig = plot_result(f, mu_main_peak_analysis, sigma_main_peak_analysis, add_gaussians)
    figs_to_save.append(fig)

    # Case 2: 5 Gaussians with offset, some overlapping
    gaussians = [
        # Amplitude, mu, sigma (5 Gaussians)
        150, 30, 8,
        200, 110, 3,
        300, 150, 20,
        75, 370, 40,
        300, 250, 1]
   
    testdata = create_multigauss_testdata(
        x=signal_index,
        params=gaussians,
        offset=45,
        add_noise=True,
        noise_frac=0.03
    )
    
    f = MultiGaussFit()
    f.set_data(*testdata)
    f.run_optimisation()

    # For illustration: compute the 1st and 2nd moment of the parameterised
    # multi Gaussian distribution, i.e. the most distinctive peak. Note that 
    # the 1st moment is the mean and the 2nd moment is the variance.
    # Note that this considers all detected Gaussians, not just the main peak (see fit result)
    # Regardless of this, the result is still a good approximation of the main peak as can
    # be seen in the console output at the end of this script.
    p_norm = f.normalise_params()
    x = f.index
    data_norm = multi_gaussian_no_offset(x, *p_norm)

    mu_all_peaks = f.det_moment(x, data_norm, 0, 1)
    sigma_all_peaks = np.sqrt(f.det_moment(x, data_norm, mu_all_peaks, 2))

    (mu_main_peak_analysis, 
     sigma_main_peak_analysis, 
     _, 
     add_gaussians) = f.analyse_fit_result(sigma_tol_overlaps=3)

    # Visualise the result including the residual
    fig = plot_result(f, mu_main_peak_analysis, sigma_main_peak_analysis, add_gaussians)
    figs_to_save.append(fig)
    logger.info(f"\nMu, sigma (all Gaussians): {mu_all_peaks:.2f}, {sigma_all_peaks:.2f}")
    logger.info(f"\nMu, sigma (Gaussian main peak): {mu_main_peak_analysis:.2f}, {sigma_main_peak_analysis:.2f}")

     ### IMPORTANT STUFF FINISHED
    if SAVEFIGS:
        for i, fig in enumerate(figs_to_save):
            outfile = SAVE_DIR / f"ex0_8_out_{i+1}.png"
            fig.savefig(outfile, format=FORMAT, dpi=DPI)
        
    # Import script options
    options = ARGPARSER.parse_args()
    try:
        if int(options.show) == 1:
            plt.show()
    except BaseException:
        print("Use option --show 1 if you want the plots to be displayed")

if __name__ == "__main__":
    plt.close("all")
    main()