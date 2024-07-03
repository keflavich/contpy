import astropy
from astropy.stats import sigma_clip
import numpy as np
from scipy import stats
from scipy.optimize import leastsq
import astropy.io.ascii as ascii

##======================================================================
def c_max(flux, rms_noise):
    """
    Perform histogram distribution of variable flux, and determine
    the flux level of the maximum of the histogram

    Parameters
    ----------
    flux : np.ndarray
        One-dimension array of flux values
    rms_noise : float
        The estimated RMS noise level of the data

    Returns
    -------
    maximum_flux : float
        The measured continuum flux as the maximum of the histogram
    """

    all_bins, all_hist, sel_bins, sel_hist, sel_flux = cont_histo(flux, rms_noise)

    maximum_flux = all_bins[(np.where(all_hist == all_hist.max())[0])[0]]

    return maximum_flux

##======================================================================
def c_mean(flux, rms_noise):
    """
    Perform mean of the distribution of variable flux, and determine
    the mean of a selected range around the maximum of the histogram

    Parameters
    ----------
    flux : np.ndarray
        One-dimension array of flux values
    rms_noise : float
        The estimated RMS noise level of the data

    Returns
    -------
    mean_flux : float
        The measured continuum flux as the mean of the distribution
    meansel_flux : float
        The measured continuum flux as the mean of a selected range
        around the maximum of the distribution
    """

    all_bins, all_hist, sel_bins, sel_hist, sel_flux = cont_histo(flux, rms_noise)

    mean_flux = np.mean(flux)
    meansel_flux = np.mean(sel_flux)

    return mean_flux, meansel_flux

##======================================================================
def c_median(flux, rms_noise):
    """
    Perform median of the distribution of variable flux, and determine
    the median of a selected range around the maximum of the histogram

    Parameters
    ----------
    flux : np.ndarray
        One-dimension array of flux values
    rms_noise : float
        The estimated RMS noise level of the data

    Returns
    -------
    median_flux : float
        The measured continuum flux as the median of the distribution
    mediansel_flux : float
        The measured continuum flux as the median of a selected range
        around the maximum of the distribution
    """

    all_bins, all_hist, sel_bins, sel_hist, sel_flux = cont_histo(flux, rms_noise)

    median_flux = np.median(flux)
    mediansel_flux = np.median(sel_flux)

    return median_flux, mediansel_flux

##======================================================================
def c_percent(flux, percentile):
    """
    Perform numpy percentile to determine the level of the selected
    percentile

    Parameters
    ----------
    flux : np.ndarray
        One-dimension array of flux values
    percentile : float
        The selected percentile

    Returns
    -------
    percent_flux : float
        The measured continuum flux at the selected percentile
    """

    percent_flux = np.percentile(flux, percentile)

    return percent_flux

##======================================================================
def c_KDEmax(flux, rms_noise, betaversion):
    """
    Perform KDE of the distribution and determine the position of the
    maximum

    Parameters
    ----------
    flux : np.ndarray
        One-dimension array of flux values
    rms_noise : float
        The estimated RMS noise level of the data
    betaversion : logic
        Activate more functionalities for developers

    Returns
    -------
    KDEmax_flux : float
        The measured continuum flux as the position of the maximum
        of the KDE
    """

    # Definition of the kernel following Silverman's rule
    #  - width = [4/(3*ndata)]^(1/5)*min(dispersion, IQR/1/.34)
    # In SciPy the dispersion is included in the covariance factor
    # and therefore we need to divide by this factor
    # Following Silverman 1998, Eqs 3.28 and 3.30
    # B. W. Silverman, Density Estimation for Statistics and Data Analysis (CRC Press, Boca Raton, 1998)
    #
    # For manual definition of the kernel width, in SciPy:
    #  - KDE_bandwidth = rms_noise/np.std(flux)
    # with this expression the kernel width equals "rms_noise"

    Silverman_spread = min(np.std(flux), ((np.percentile(flux, 75)-np.percentile(flux, 25))/1.34))
    KDE_bandwidth = ((4./(3.*len(flux)))**(1./5.)*Silverman_spread)/np.std(flux)
    scipy_kde = stats.gaussian_kde(flux, bw_method=KDE_bandwidth)
    KDExmin, KDExmax = min(flux), max(flux)
    KDEx = np.mgrid[KDExmin:KDExmax:5000j]
    positions = np.vstack([KDEx.ravel()])
    KDEpos = scipy_kde(positions)
    KDEmax_flux = positions.T[np.argmax(KDEpos)]

    # Write out the KDE as ASCII file
    if betaversion:
        ascii.write((KDEx, KDEpos), output='statcont-developers/STATCONT_KDE_distribution.dat')

    return KDEmax_flux

##======================================================================
def c_Gaussian(flux, rms_noise):
    """
    Perform Gaussian fit to the distribution of variable flux, and determine
    the center and width of the Gaussian. Similarly, perform the Gaussian
    fit to a selected range of the distribution around the maximum of
    the histogram, and determine the center and width of the new Gaussian

    Parameters
    ----------
    flux : np.ndarray
        One-dimension array of flux values
    rms_noise : float
        The estimated RMS noise level of the data

    Returns
    -------
    Gaussian_flux : float
    Gaussian_noise : float
        The measured continuum flux and estimated 1-sigma noise as the
        center and width of the Gaussian fit to the histogram distribution
        The estimated 1-sigma per-channel noise around that measurement
    GaussNw_flux : float
    GaussNw_noise : float
        The measured continuum flux and estimated 1-sigma noise as the
        center and width of the Gaussian fit to a selected range around
        the maximum of the distribution
    """

    fitfunc = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2.)
    errfunc = lambda p, x, y: (y - fitfunc(p, x))

    all_bins, all_hist, sel_bins, sel_hist, sel_flux = cont_histo(flux, rms_noise)

    meansel_flux = np.mean(sel_flux)
    meansel_sigma = np.std(sel_flux)

    init = [all_hist.max(), meansel_flux, meansel_sigma]
    out = leastsq(errfunc, init, args=(all_bins, all_hist))
    c = out[0]
    Gaussian_flux = c[1]
    Gaussian_noise = c[2]

    init = [all_hist.max(), meansel_flux, meansel_sigma]
    out = leastsq(errfunc, init, args=(sel_bins, sel_hist))
    d = out[0]
    GaussNw_flux = d[1]
    GaussNw_noise = d[2]

    return Gaussian_flux, Gaussian_noise, GaussNw_flux, GaussNw_noise

##======================================================================
def c_sigmaclip1D(flux, rms_noise, betaversion, sigma_clip_threshold=2.0):
    """
    Perform sigma-clipping to determine the mean flux level, with different
    adaptations for emission- and absorption-dominated spectra
    It runs on one-dimensional arrays

    Parameters
    ----------
    flux : np.ndarray
        One-dimension array of flux values
    rms_noise : float
        The estimated RMS noise level of the data
    betaversion : logic
        Activate more functionalities for developers
    sigma_clip_threshold : float
        The threshold in number of sigma above/below which to reject outlier
        data

    Returns
    -------
    sigmaclip_flux_prev : float
    sigmaclip_flux : float
    sigmaclip_noise : float
        The measured continuum flux and estimated 1-sigma per-channel noise
        around that measurement
        The variable sigmaclip_flux_prev contains the continuum determined
        with the sigma-clipping method without applying the correction
        of STATCONT
    """

    # Sigma-clipping method applied to the flux array
    if astropy.version.major >= 3:
        filtered_data = sigma_clip(flux, sigma=sigma_clip_threshold,
                                   maxiters=None)
    elif astropy.version.major >= 1:
        filtered_data = sigma_clip(flux, sigma=sigma_clip_threshold,
                                   maxiters=None)
    elif astropy.version.major < 1:
        filtered_data = sigma_clip(flux, sig=sigma_clip_threshold, maxiters=None)

    sigmaclip_flux_prev = sigmaclip_flux = np.mean(filtered_data)
    sigmaclip_noise = sigmaclip_sigma = np.std(filtered_data)
    mean_flux = np.mean(flux)

    # Correction of sigma-clip continuum level, making use of the
    # presence of emission and/or absorption line features

    # Set up the fraction of channels (in %) that are in emission
    fraction_emission = 0
    fraction_emission = sum(i > (sigmaclip_flux+1*rms_noise) for i in flux)
    fraction_emission = 100*fraction_emission/len(flux)

    # Set up the fraction of channels (in %) that are in absorption
    fraction_absorption = 0
    fraction_absorption = sum(i < (sigmaclip_flux-1*rms_noise) for i in flux)
    fraction_absorption = 100*fraction_absorption/len(flux)

    # Apply correction to continuum level
    # see details in Sect. 2.4 of Sanchez-Monge et al. (2017)
    if (fraction_emission < 33 and fraction_absorption < 33):
        sigmaclip_flux = sigmaclip_flux_prev
    elif (fraction_emission >= 33 and fraction_absorption < 33):
        if (fraction_emission-fraction_absorption > 25):
            sigmaclip_flux = sigmaclip_flux_prev - 1.0*sigmaclip_sigma
        if (fraction_emission-fraction_absorption <= 25):
            sigmaclip_flux = sigmaclip_flux_prev - 0.5*sigmaclip_sigma
    elif (fraction_emission < 33 and fraction_absorption >= 33):
        if (fraction_absorption-fraction_emission > 25):
            sigmaclip_flux = sigmaclip_flux_prev + 1.0*sigmaclip_sigma
        if (fraction_absorption-fraction_emission <= 25):
            sigmaclip_flux = sigmaclip_flux_prev + 0.5*sigmaclip_sigma
    elif (fraction_emission >= 33 and fraction_absorption >= 33):
        if (fraction_emission-fraction_absorption > 25):
            sigmaclip_flux = sigmaclip_flux_prev - 1.0*sigmaclip_sigma
        if (fraction_absorption-fraction_emission > 25):
            sigmaclip_flux = sigmaclip_flux_prev + 1.0*sigmaclip_sigma
        if (abs(fraction_absorption-fraction_emission) <= 25):
            sigmaclip_flux = sigmaclip_flux_prev

    if betaversion is False:
        return sigmaclip_flux_prev, sigmaclip_flux, sigmaclip_noise

    # Write out the original and filtered data as a two-column ASCII file
    if betaversion:
        ascii.write((flux, filtered_data), output='statcont-developers/STATCONT_sigmaclip_filtered.dat')

        # Determine the numbers of channels above and under the real continuum level,
        # for synthetic files with continuum level set to 50.0
        # (i.e. how many channels are in emission/absorption with respect to the total)
        #
        real_fraction_emission = 0
        real_fraction_emission = sum(i > (50.0+1*rms_noise) for i in flux)
        real_fraction_emission = 100*real_fraction_emission/len(flux)
        real_fraction_absorption = 0
        real_fraction_absorption = sum(i < (50.0-1*rms_noise) for i in flux)
        real_fraction_absorption = 100*real_fraction_absorption/len(flux)

        return sigmaclip_flux_prev, sigmaclip_flux, sigmaclip_noise, real_fraction_emission, fraction_emission, real_fraction_absorption, fraction_absorption

    ### For EMISSION-dominated spectra
    ##if (mean_flux-sigmaclip_flux_prev) > (+1.0*rms_noise):
    ##    sigmaclip_flux = sigmaclip_flux_prev - sigmaclip_sigma
    ### For ABSORPTION-dominated spectra
    ##elif (mean_flux-sigmaclip_flux_prev) < (-1.0*rms_noise):
    ##    sigmaclip_flux = sigmaclip_flux_prev + sigmaclip_sigma

##======================================================================
def c_sigmaclip(flux, rms_noise, freq_axis, sigma_clip_threshold=1.8):
    """
    Perform sigma-clipping to determine the mean flux level, with different
    adaptations for emission- and absorption-dominated spectra
    Different to c_sigmaclip1D function, it works directly on arrays
    It speeds up the process of continuum determination

    Parameters
    ----------
    flux : np.ndarray
        Multi-dimension array of flux values
    rms_noise : float
        The estimated RMS noise level of the data
    freq_axis : integer
        Python-based dimension for the frequency (usually 0 or 1)
    sigma_clip_threshold : float
        The threshold in number of sigma above/below which to reject outlier
        data

    Returns
    -------
    sigmaclip_flux_prev : np.ndarray
    sigmaclip_flux : np.ndarray
    sigmaclip_noise : np.ndarray
        The measured continuum flux and estimated 1-sigma per-channel noise
        around that measurement
        The variable sigmaclip_flux_prev contains the continuum determined
        with the sigma-clipping method without applying the correction
        of STATCONT
    """

    # Sigma-clipping method applied to the flux array
    if astropy.version.major >= 3:
        filtered_data = astropy.stats.sigma_clip(flux, sigma=sigma_clip_threshold,
                                                 maxiters=None, axis=freq_axis)
    elif astropy.version.major < 3:
        filtered_data = astropy.stats.sigma_clip(flux, sigma=sigma_clip_threshold,
                                                 iters=None, axis=freq_axis)

    sigmaclip_flux_prev = sigmaclip_flux = np.mean(filtered_data, axis=freq_axis)
    sigmaclip_noise = sigmaclip_sigma = np.std(filtered_data, axis=freq_axis)
    mean_flux = np.mean(flux)

    # Correction of sigma-clip continuum level, making use of the
    # presence of emission and/or absorption line features

    naxis = len(flux.shape)

    # Handle different shapes; Stokes cube, cube, and single spectra
    if naxis == 4:

        view1 = [0, slice(None), slice(None), slice(None)]

    if naxis == 3:

        view1 = [slice(None), slice(None), slice(None)]

    if naxis == 1:

        view1 = [slice(None)]

    # Set up the fraction of channels (in %) that are in emission
    fraction_emission = (100 * (flux[tuple(view1)] >
                                (sigmaclip_flux+1*rms_noise)).sum(axis=0) /
                         flux[tuple(view1)].shape[0])
    
    # Set up the fraction of channels (in %) that are in absorption
    fraction_absorption = (100 * (flux[tuple(view1)] <
                                  (sigmaclip_flux-1*rms_noise)).sum(axis=0) /
                           flux[tuple(view1)].shape[0])
    
    # Apply correction to continuum level
    # see details in Sect. 2.4 of paper Sanchez-Monge et al. (2018)
    sigmaclip_flux_case1 = np.where((fraction_emission < 33) &
                                    (fraction_absorption < 33),
                                    sigmaclip_flux_prev, 0.0)
    sigmaclip_flux_case2 = np.where((fraction_emission >= 33) &
                                    (fraction_absorption < 33) &
                                    (fraction_emission-fraction_absorption >
                                     25.0), sigmaclip_flux_prev -
                                    1.0*sigmaclip_sigma, 0.0)
    sigmaclip_flux_case3 = np.where((fraction_emission >= 33) &
                                    (fraction_absorption < 33) &
                                    (fraction_emission-fraction_absorption <=
                                     25.0), sigmaclip_flux_prev -
                                    0.5*sigmaclip_sigma, 0.0)
    sigmaclip_flux_case4 = np.where((fraction_emission < 33) &
                                    (fraction_absorption >= 33) &
                                    (fraction_emission-fraction_absorption >
                                     25.0), sigmaclip_flux_prev +
                                    1.0*sigmaclip_sigma, 0.0)
    sigmaclip_flux_case5 = np.where((fraction_emission < 33) &
                                    (fraction_absorption >= 33) &
                                    (fraction_emission-fraction_absorption <=
                                     25.0), sigmaclip_flux_prev +
                                    0.5*sigmaclip_sigma, 0.0)
    sigmaclip_flux_case6 = np.where((fraction_emission >= 33) &
                                    (fraction_absorption >= 33) &
                                    (fraction_emission-fraction_absorption >
                                     25.0), sigmaclip_flux_prev -
                                    1.0*sigmaclip_sigma, 0.0)
    sigmaclip_flux_case7 = np.where((fraction_emission >= 33) &
                                    (fraction_absorption >= 33) &
                                    (fraction_absorption-fraction_emission >
                                     25.0), sigmaclip_flux_prev +
                                    1.0*sigmaclip_sigma, 0.0)
    sigmaclip_flux_case8 = np.where((fraction_emission >= 33) &
                                    (fraction_absorption >= 33) &
                                    (abs(fraction_absorption-fraction_emission)
                                     <= 25.0), sigmaclip_flux_prev, 0.0)

    sigmaclip_flux = (sigmaclip_flux_case1 + sigmaclip_flux_case2 +
                      sigmaclip_flux_case3 + sigmaclip_flux_case4 +
                      sigmaclip_flux_case5 + sigmaclip_flux_case6 +
                      sigmaclip_flux_case7 + sigmaclip_flux_case8)

    # Remove masked values if any
    if isinstance(sigmaclip_flux_prev, np.ma.MaskedArray):
        sigmaclip_flux_prev = sigmaclip_flux_prev.filled()
    if isinstance(sigmaclip_flux, np.ma.MaskedArray):
        sigmaclip_flux = sigmaclip_flux.filled()
    if isinstance(sigmaclip_noise, np.ma.MaskedArray):
        sigmaclip_noise = sigmaclip_noise.filled()

    return sigmaclip_flux_prev, sigmaclip_flux, sigmaclip_noise, filtered_data

##======================================================================
def cont_histo(flux, rms_noise):
    """
    Create histogram distribution of the flux data
    and select a narrower range around the maximum
    of the histogram distribution

    Parameters
    ----------
    flux : np.ndarray
        One-dimension array of flux values
    rms_noise : float
        The estimated RMS noise level of the data

    Returns
    -------
    all_bins : np.ndarray
        One-dimension array with the value of bins of the histogram
    all_hist : np.ndarray
        One-dimension array with the value of the position of the bins
    sel_bins : np.ndarray
        One-dimension array with the value of bins of the histogram
        for the selected bins around the maximum
    sel_hist : np.ndarray
        One-dimension array with the value of position of the bins
        for the selected bins around the maximum
    sel_flux : np.ndarray
        One-dimension array of the flux values selected
        around the maximum of the histogram
    """

    #
    # creating a general histogram of the flux data
    # main variables are:
    #   all_hist     - counts in each bin of the histogram
    #   all_bins     - location of the bins (fluxes)
    #   all_number_* - index of the array
    number_bins = int((np.amax(flux)-np.amin(flux))/(1*rms_noise))
    all_hist, all_bin_edges = np.histogram(flux, number_bins)
    all_bins = all_bin_edges[0:len(all_bin_edges)-1]
    all_bins = [x + (all_bins[1]-all_bins[0])/2. for x in all_bins]
    all_number_max_array = (np.where(all_hist == all_hist.max())[0])
    all_number_max = all_number_max_array[0]
    all_bins_max = (all_bin_edges[all_number_max] + (all_bins[1]-all_bins[0])/2.)

    # Gaussian fit around the maximum of the distribution
    # determining the range to fit the Gaussian function
    all_number_left  = (np.where(((all_hist == 0) & (all_bins <= all_bins_max)) | (all_bins == all_bins[0]))[0]).max()
    all_number_right = (np.where(((all_hist == 0) & (all_bins >= all_bins_max)) | (all_bins == all_bins[number_bins-1]))[0]).min()
    all_number_total = abs(all_number_right-all_number_max)+abs(all_number_left-all_number_max)
    emission_absorption_ratio = abs(all_number_right-all_number_max)*1.0/(all_number_total*1.0)
    if (emission_absorption_ratio >= 0.66):
        lower_all_bins = all_bins_max - 8. * (all_bins[1]-all_bins[0])
        upper_all_bins = all_bins_max + 4. * (all_bins[1]-all_bins[0])
    if (emission_absorption_ratio <= 0.33):
        lower_all_bins = all_bins_max - 4. * (all_bins[1]-all_bins[0])
        upper_all_bins = all_bins_max + 8. * (all_bins[1]-all_bins[0])
    if ((emission_absorption_ratio > 0.33) and (emission_absorption_ratio < 0.66)):
        lower_all_bins = all_bins_max - 5. * (all_bins[1]-all_bins[0])
        upper_all_bins = all_bins_max + 5. * (all_bins[1]-all_bins[0])
    sel_bins_array = np.where((all_bins >= lower_all_bins) & (all_bins <= upper_all_bins))[0]
    if (len(sel_bins_array) < 3):
        sel_bins_array = [sel_bins_array[0]-2, sel_bins_array[0]-1, sel_bins_array[0], sel_bins_array[0]+1, sel_bins_array[0]+2]
        lower_all_bins = all_bins[sel_bins_array[0]]
        upper_all_bins = all_bins[sel_bins_array[len(sel_bins_array)-1]]
    sel_bins = all_bins[sel_bins_array[0]:sel_bins_array[len(sel_bins_array)-1]+1]
    sel_hist = all_hist[sel_bins_array[0]:sel_bins_array[len(sel_bins_array)-1]+1]
    sel_flux = flux[(flux >= lower_all_bins) & (flux <= upper_all_bins)]

    return all_bins, all_hist, sel_bins, sel_hist, sel_flux

def c_sigmaclip_scube(cube, rms_noise, freq_axis=0, sigma_clip_threshold=1.8,
                      rechunk=[-1, 'auto', 'auto'],
                      save_to_tmp_dir=True,
                      verbose=False):
    """
    Perform sigma-clipping to determine the mean flux level, with different
    adaptations for emission- and absorption-dominated spectra
    Different to c_sigmaclip1D function, it works directly on arrays
    It speeds up the process of continuum determination

    Parameters
    ----------
    flux : np.ndarray
        Multi-dimension array of flux values
    rms_noise : float
        The estimated RMS noise level of the data
    freq_axis : integer
        Python-based dimension for the frequency (usually 0 or 1)
    sigma_clip_threshold : float
        The threshold in number of sigma above/below which to reject outlier
        data
    rechunk : None or list
        Shape to rechunk the dask cube to.  -1 is to use the whole axis,
        'auto' is to automatically determine it.  [-1, 'auto', 'auto']
        chunks into full spectral with automatically-determined spatial
        chunk sizes.  [-1,1,1] would be chunking each spectrum separately.
    save_to_tmp_dir : bool
        Save the intermediate calculated ``sigma_clip_spectrally`` result to
        a temporary ``zarr`` file before doing subsequent operations?
        This may speed up calculations by ~2x.  Be sure that your default
        temp directory can store the full cube; the tempdir location can be
        set with the environmental variable TEMPDIR.

    Returns
    -------
    sigmaclip_flux_prev : np.ndarray
    sigmaclip_flux : np.ndarray
    sigmaclip_noise : np.ndarray
        The measured continuum flux and estimated 1-sigma per-channel noise
        around that measurement
        The variable sigmaclip_flux_prev contains the continuum determined
        with the sigma-clipping method without applying the correction
        of STATCONT
    """
    from astropy import units as u

    # ensure the rms_noise is in the right unit
    rms_noise = u.Quantity(rms_noise, cube.unit)

    if rechunk is not None:
        try:
            # if dask, rechunk to use full spectral axes
            cube = cube.rechunk(rechunk)
        except AttributeError:
            pass


    # print out the cube to show its chunking dimensions
    if verbose:
        print(cube)

    # Sigma-clipping method applied to the flux array
    try:
        filtered_cube = cube.sigma_clip_spectrally(threshold=sigma_clip_threshold,
                                                   save_to_tmp_dir=save_to_tmp_dir,
                                                   maxiters=None)
    except TypeError:
        # if the cube is not a dask_spectral_cube
        filtered_cube = cube.sigma_clip_spectrally(threshold=sigma_clip_threshold,
                                                   maxiters=None)

    sigmaclip_flux_prev = sigmaclip_flux = filtered_cube.mean(axis=freq_axis)
    sigmaclip_noise = sigmaclip_sigma = filtered_cube.std(axis=freq_axis)

    # Correction of sigma-clip continuum level, making use of the
    # presence of emission and/or absorption line features

    # Set up the fraction of channels (in %) that are in emission
    fraction_emission = (100 * (cube >
                                (sigmaclip_flux+1*rms_noise)).include().sum(axis=freq_axis) /
                         cube.shape[freq_axis])

    # Set up the fraction of channels (in %) that are in absorption
    fraction_absorption = (100 * (cube <
                                  (sigmaclip_flux-1*rms_noise)).include().sum(axis=freq_axis) /
                           cube.shape[freq_axis])

    # Apply correction to continuum level
    # see details in Sect. 2.4 of paper Sanchez-Monge et al. (2018)
    sigmaclip_flux_case1 = np.where((fraction_emission < 33) &
                                    (fraction_absorption < 33),
                                    sigmaclip_flux_prev, 0.0)
    sigmaclip_flux_case2 = np.where((fraction_emission >= 33) &
                                    (fraction_absorption < 33) &
                                    (fraction_emission-fraction_absorption >
                                     25.0), sigmaclip_flux_prev -
                                    1.0*sigmaclip_sigma, 0.0)
    sigmaclip_flux_case3 = np.where((fraction_emission >= 33) &
                                    (fraction_absorption < 33) &
                                    (fraction_emission-fraction_absorption <=
                                     25.0), sigmaclip_flux_prev -
                                    0.5*sigmaclip_sigma, 0.0)
    sigmaclip_flux_case4 = np.where((fraction_emission < 33) &
                                    (fraction_absorption >= 33) &
                                    (fraction_emission-fraction_absorption >
                                     25.0), sigmaclip_flux_prev +
                                    1.0*sigmaclip_sigma, 0.0)
    sigmaclip_flux_case5 = np.where((fraction_emission < 33) &
                                    (fraction_absorption >= 33) &
                                    (fraction_emission-fraction_absorption <=
                                     25.0), sigmaclip_flux_prev +
                                    0.5*sigmaclip_sigma, 0.0)
    sigmaclip_flux_case6 = np.where((fraction_emission >= 33) &
                                    (fraction_absorption >= 33) &
                                    (fraction_emission-fraction_absorption >
                                     25.0), sigmaclip_flux_prev -
                                    1.0*sigmaclip_sigma, 0.0)
    sigmaclip_flux_case7 = np.where((fraction_emission >= 33) &
                                    (fraction_absorption >= 33) &
                                    (fraction_absorption-fraction_emission >
                                     25.0), sigmaclip_flux_prev +
                                    1.0*sigmaclip_sigma, 0.0)
    sigmaclip_flux_case8 = np.where((fraction_emission >= 33) &
                                    (fraction_absorption >= 33) &
                                    (abs(fraction_absorption-fraction_emission)
                                     <= 25.0), sigmaclip_flux_prev, 0.0)

    sigmaclip_flux = (sigmaclip_flux_case1 + sigmaclip_flux_case2 +
                      sigmaclip_flux_case3 + sigmaclip_flux_case4 +
                      sigmaclip_flux_case5 + sigmaclip_flux_case6 +
                      sigmaclip_flux_case7 + sigmaclip_flux_case8)

    # Remove masked values if any
    if isinstance(sigmaclip_flux_prev, np.ma.MaskedArray):
        sigmaclip_flux_prev = sigmaclip_flux_prev.filled()
    if isinstance(sigmaclip_flux, np.ma.MaskedArray):
        sigmaclip_flux = sigmaclip_flux.filled()
    if isinstance(sigmaclip_noise, np.ma.MaskedArray):
        sigmaclip_noise = sigmaclip_noise.filled()

    return sigmaclip_flux_prev, sigmaclip_flux, sigmaclip_noise
