from __future__ import print_function

from .fits_utils import *
from .cont_finding import *

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import astropy
import astropy.stats
import astropy.io.ascii as ascii
import datetime
#from astropy.io import fits
#from astropy.stats import sigma_clip
#from scipy import stats
#from scipy.optimize import leastsq

def process_files(iname=False,
                  ifile=False,
                  ispec=False,
                  ilist=False,
                  imerge=False,
                  ipath=False,
                  rms_noise=None,
                  continuum=False,
                  cmax=False,
                  cmean=False,
                  cmedian=False,
                  cpercent=False,
                  cKDEmax=False,
                  cGaussian=False,
                  csigmaclip=False,
                  cfree=False,
                  nooffset=False,
                  spindex=False,
                  model=False,
                  plots=False,
                  cutout=False,
                  verbose=None,
                  betaversion=False,
                  localdir=False):

    # Read name of files to be processed
    # - for FITS file (extension if .fits)
    if iname:
        input_files = iname
        extension = '.fits'

    # - for a list of FITS files (extension of files is .fits)
    elif ifile:
        print(ifile[0])
        lines = [line.rstrip('\n') for line in open(ifile[0])]
        input_files = lines
        extension = '.fits'

    # - for an ASCII file (extension is .dat)
    elif ispec:
        input_files = ispec
        extension = '.dat'

    # - for an ASCII file (extension of files is .dat)
    elif ilist:
        print(ilist[0])
        lines = [line.rstrip('\n') for line in open(ilist[0])]
        input_files = lines
        extension = '.dat'
        print("HOLA")
        print(input_files)
        print("ADIOS")

    # Create directories and define working paths
    if betaversion:
        os.system('mkdir -p statcont-developers/')

    if localdir:
        data_path = "./"
        cont_path = "./"
        line_path = cont_path
        if ipath:
            source = ipath[0]
            sourcedir = source + '/'
            data_path = data_path + sourcedir
            cont_path = cont_path + sourcedir
            os.system('mkdir -p ' + cont_path)
            line_path = line_path + sourcedir
            os.system('mkdir -p ' + line_path)
        if cutout:
            cutout_path = data_path + 'cutout/'
            os.system('mkdir -p ' + cutout_path)
    else:
        data_path = "data/"
        cont_path = "products/"
        line_path = cont_path
        os.system('mkdir -p ' + data_path)
        os.system('mkdir -p ' + cont_path)
        
        # Define sub-directory (within data/) containing the files to be processed
        if ipath:
            source = ipath[0]
            sourcedir = source + '/'
            data_path = data_path + sourcedir
            cont_path = cont_path + sourcedir
            os.system('mkdir -p ' + cont_path)
            line_path = line_path + sourcedir
            os.system('mkdir -p ' + line_path)
        if cutout:
            cutout_path = data_path + 'cutout/'
            os.system('mkdir -p ' + cutout_path)

    plots_path = cont_path + 'statcont-plots/'
    os.system('mkdir -p ' + plots_path)

    # Set path and file names ...
    # ... and in case of FITS files, use cutout to create smaller files
    if ispec or ilist:
        tmp_files = []
        for file_name in input_files:
            tmp_path = data_path
            tmp_file = file_name
            tmp_files.append(tmp_file)

    if iname or ifile:
        if cutout:
            if verbose >= 1:
                print("+++ Producing cutout files ...")
            tmp_files = []
            for file_name in input_files:
                data_fitsfile = data_path + file_name + extension
                central_xpixel = cutout[0]
                central_ypixel = cutout[1]
                number_pixels = cutout[2]
                if verbose >= 1:
                    print("  . Cutout of %s \n    at central pixel %i, %i with size %i" % (data_fitsfile, central_xpixel, central_ypixel, number_pixels))
                cutout_fitsfile = cutout_path + file_name + '_cutout_' + str(central_xpixel) + '_' + str(central_ypixel) + '_' + str(number_pixels) + extension
                fits_cutout(data_fitsfile, central_xpixel, central_ypixel, number_pixels, cutout_fitsfile)
                tmp_path = cutout_path
                tmp_file = file_name + '_cutout_' + str(central_xpixel) + '_' + str(central_ypixel) + '_' + str(number_pixels)
                tmp_files.append(tmp_file)
        else:
            tmp_files = []
            for file_name in input_files:
                tmp_path = data_path
                tmp_file = file_name
                tmp_files.append(tmp_file)

    # Merge FITS files if required
    if imerge:
        if verbose >= 1:
            print("+++ Merging files ...")
        
        merged_path = data_path + 'statcont-merged/'
        os.system('mkdir -p ' + merged_path)

        # Name of the output file
        merged_file_name = imerge[0]  

        tmp_files, tmp_path, unmerged_files, unmerged_path = fits_merge(tmp_files, tmp_path, extension, merged_file_name, merged_path)

    # For developers, turn off verbose
    #if betaversion:
    #    verbose = False
        
    # Loop through all the files that will be processed
    for tmp_file in tmp_files:

        if verbose >= 1:
            print("")
            print("+++ PROCESSING " + tmp_path + tmp_file + extension)

        # Read data and header of the FITS file
        if iname or ifile:

            header = fits.getheader(tmp_path + tmp_file + extension)
            data = fits.getdata(tmp_path + tmp_file + extension)

            ndim = header.get('NAXIS')
            nxpix = header.get('NAXIS1')
            nypix = header.get('NAXIS2')
            nchan = header.get('NAXIS3')
            npolz = header.get('NAXIS4')
            bunit = header.get('BUNIT')

        # Read data and header of the ASCII file
        if ispec or ilist:

            specfile = open(tmp_path + tmp_file + extension, 'r')

            nxpix = 1
            nypix = 1
            nchan = 1 # will be updated to the real value later
            npolz = 1

        # Determine the continuum level using different methods
        if continuum:

            # Message informing about required computational time
            # an image of 512 x 512 pixels my require about 1 hour
            # We have run a sample of examples to derive the following relation with time:
            #   log(time) = a * log(nxpix*nypix*nchan) + b
            # where time is in minutes, and a and b are given below
            time_a = +1.14
            time_b = -7.30
            requested_time = 10.**(time_a+np.log10(nxpix*nypix*nchan)+time_b) # in minutes
            requested_time = requested_time/60. # in hours
            
            print("  . your image has " + str(nxpix) + "x" + str(nypix) + " pixels, and " + str(nchan) + " channels")
            print("  . current time is        " + str(datetime.datetime.now()))
            print("  . estimated ending       " + str(datetime.datetime.now()+datetime.timedelta(hours=requested_time)))
            
            # Fast determination of the continuum level using corrected SIGMA-CLIP on arrays
            
            # For FITS files
            if iname or ifile:

                # Write the intensity of a given pixel for all the channels into the array flux
                if ndim == 4:
                    freq_axis = 1
                    flux = data

                if ndim == 3:
                    freq_axis = 0
                    flux = data

                # Write the frequencies in the array freqs
                chans = []
                freqs = []
                for channel in range(nchan):
                    chans.append(channel)
                    freqs.append((header.get('CRVAL3') + (channel - header.get('CRPIX3') - 1) * header.get('CDELT3')) / 1.e9)
                freq = np.array(freqs)

            # For ASCII files
            if ispec or ilist:

                # Write the intensity of a given pixel for all the channels into the array flux
                # and the frequencies in the array freqs
                freq_axis = None
                lflux = []
                freqs = []
                for line in specfile:
                    freqs.append(float(line.split()[0]))
                    lflux.append(float(line.split()[1]))
                flux = np.array(lflux)
                freq = np.array(freqs)
                nchan = len(flux)
                flux = flux

            sigmaclip_flux_prev, sigmaclip_flux, sigmaclip_noise, filtered_data = c_sigmaclip(flux, rms_noise, freq_axis)
            continuum_flux = sigmaclip_flux
            continuum_noise = sigmaclip_noise

            # Write final time to compare with prediction
            print("  . continuum estimated on " + str(datetime.datetime.now()))
            
            if verbose >= 3:
                print("    flux of continuum    = " + str(int(sigmaclip_flux*1.e5)/1.e5) + " +/- " + str(int(sigmaclip_noise*1.e5)/1.e5))

            # Write the output continuum file
            if verbose >= 1:
                if (cmax == False) and (cmean == False) and (cmedian == False) and (cpercent == False) and (cKDEmax == False) and (cGaussian == False) and (csigmaclip == False):
                    print(" ")
                    print("... CONTINUUM FILEs CREATED: ")

            # Create output files with the CONTINUUM estimated values
            output_files = []
            output_fluxs = []

            output_files.append(cont_path + tmp_file + '_continuum' + extension)
            output_fluxs.append(continuum_flux)
            output_files.append(cont_path + tmp_file + '_noise' + extension)
            output_fluxs.append(continuum_noise)

            # Loop though x and y pixels if alternative method is used (old version)
            # Set up the variables that will contain the continuum level and noise
            if cmax:
                continuum_flux_maximum = []
            if cmean:
                continuum_flux_mean = []
                continuum_flux_meansel = []
            if cmedian:
                continuum_flux_median = []
                continuum_flux_mediansel = []
            if cpercent:
                continuum_flux_percent25 = []
                continuum_flux_percent75 = []
            if cKDEmax:
                continuum_flux_KDEmax = []
            if cGaussian:
                continuum_flux_GaussNw = []
                continuum_noise_GaussNw = []
                continuum_flux_Gaussian = []
                continuum_noise_Gaussian = []
            if csigmaclip:
                continuum_flux_sigmaclip_prev = []
                continuum_flux_sigmaclip = []
                continuum_noise_sigmaclip = []

            if cmax or cmean or cmedian or cpercent or cKDEmax or cGaussian or csigmaclip or plots:
                
                # Loop through y pixels
                for ypix in range(nypix):

                    if verbose >= 1:
                        if iname or ifile:
                            print("... analyzing column " + str(ypix+1) + " out of " + str(nypix))

                    if cmax:
                        continuum_flux_maximum.append([])
                    if cmean:
                        continuum_flux_mean.append([])
                        continuum_flux_meansel.append([])
                    if cmedian:
                        continuum_flux_median.append([])
                        continuum_flux_mediansel.append([])
                    if cpercent:
                        continuum_flux_percent25.append([])
                        continuum_flux_percent75.append([])
                    if cKDEmax:
                        continuum_flux_KDEmax.append([])
                    if cGaussian:
                        continuum_flux_GaussNw.append([])
                        continuum_noise_GaussNw.append([])
                        continuum_flux_Gaussian.append([])
                        continuum_noise_Gaussian.append([])
                    if csigmaclip:
                        continuum_flux_sigmaclip_prev.append([])
                        continuum_flux_sigmaclip.append([])
                        continuum_noise_sigmaclip.append([])

                    # Loop through x pixels
                    for xpix in range(nxpix):

                        # For FITS files
                        if iname or ifile:

                            if verbose >= 3:
                                print("  . corresponding to pixel " + str(xpix+1) + "," + str(ypix+1))

                            # Write the intensity of a given pixel for all the channels into the array flux
                            if ndim == 4:
                                flux = data[0, :, ypix, xpix]

                            if ndim == 3:
                                flux = data[:, ypix, xpix]

                            # Write the frequencies in the array freqs
                            chans = []
                            freqs = []
                            for channel in range(nchan):
                                chans.append(channel)
                                freqs.append((header.get('CRVAL3') + (channel - header.get('CRPIX3') - 1) * header.get('CDELT3')) / 1.e9)
                            freq = np.array(freqs)

                        # For ASCII files
                        if ispec or ilist:

                            # Write the intensity of a given pixel for all the channels into the array flux
                            # and the frequencies in the array freqs
                            freq = freq
                            flux = flux
                            ## The following lines were necessary before the implementation of the new (fast) method
                            ## now, the ASCII file is readed above in the script
                            #lflux = []
                            #freqs = []
                            #for line in specfile:
                            #    freqs.append(float(line.split()[0]))
                            #    lflux.append(float(line.split()[1]))
                            #flux = np.array(lflux)
                            #freq = np.array(freqs)
                            #nchan = len(flux)
                            #flux = flux

                        # Determine CONTINUUM as the MAXIMUM of the histogram
                        if cmax:
                            
                            if np.isnan(np.min(flux)):
                                maximum_flux = float('nan')

                                if verbose >= 3:
                                    print("    flux of maximum      = masked")

                            else:
                                maximum_flux = c_max(flux, rms_noise)

                                if verbose >= 3:
                                    print("    flux of maximum      = " + str(int(maximum_flux*1.e5)/1.e5))

                            continuum_flux_maximum[ypix].append(maximum_flux)

                        # Determine CONTINUUM as the MEAN of the intensities
                        if cmean:
                            
                            if np.isnan(np.min(flux)):
                                mean_flux = meansel_flux = float('nan')

                                if verbose >= 3:
                                    print("    flux of mean (all)   = masked")
                                    print("    flux of mean (sel)   = masked")
                            
                            else:
                                mean_flux, meansel_flux = c_mean(flux, rms_noise)

                                if verbose >= 3:
                                    print("    flux of mean (all)   = " + str(int(mean_flux*1.e5)/1.e5))
                                    print("    flux of mean (sel)   = " + str(int(meansel_flux*1.e5)/1.e5))

                            continuum_flux_mean[ypix].append(mean_flux)
                            continuum_flux_meansel[ypix].append(meansel_flux)

                        # Determine CONTINUUM as the MEDIAN of the intensities
                        if cmedian:
                            
                            if np.isnan(np.min(flux)):
                                median_flux = mediansel_flux = float('nan')

                                if verbose >= 3:
                                    print("    flux of median (all) = masked")
                                    print("    flux of median (sel) = masked")
                            
                            else:
                                median_flux, mediansel_flux = c_median(flux, rms_noise)

                                if verbose >= 3:
                                    print("    flux of median (all) = " + str(int(median_flux*1.e5)/1.e5))
                                    print("    flux of median (sel) = " + str(int(mediansel_flux*1.e5)/1.e5))

                            continuum_flux_median[ypix].append(median_flux)
                            continuum_flux_mediansel[ypix].append(mediansel_flux)

                        # Determine CONTINUUM as the 25th and 75th percentiles of the intensities
                        if cpercent:

                            if np.isnan(np.min(flux)):
                                percent25_flux = percent75_flux = float('nan')

                                if verbose >= 3:
                                    print("    flux of percent 25   = masked")
                                    print("    flux of percent 75   = masked")

                            else:
                                percent25_flux = c_percent(flux, percentile=25)
                                percent75_flux = c_percent(flux, percentile=75)

                                if verbose >= 3:
                                    print("    flux of percent 25   = " + str(int(percent25_flux*1.e5)/1.e5))
                                    print("    flux of percent 75   = " + str(int(percent75_flux*1.e5)/1.e5))

                            continuum_flux_percent25[ypix].append(percent25_flux)
                            continuum_flux_percent75[ypix].append(percent75_flux)

                        # Determine CONTINUUM as the maximum of a KDE distribution
                        if cKDEmax:
                            
                            if np.isnan(np.min(flux)):
                                KDEmax_flux = float('nan')

                                if verbose >= 3:
                                    print("    flux of KDEmax       = masked")
                                
                            else:
                                KDEmax_flux = c_KDEmax(flux, rms_noise, betaversion)
                                
                                if betaversion:
                                    if verbose >= 1:
                                        print(" < DEVELOPERS > ")
                                        print(" < DEVELOPERS > ... KDE available in file statcont-developers/STATCONT_KDE_distribution.dat")

                                if verbose >= 3:
                                    print("    flux of KDEmax       = " + str(int(KDEmax_flux*1.e5)/1.e5))

                            continuum_flux_KDEmax[ypix].append(KDEmax_flux)

                        # Determine CONTINUUM as the center of a GAUSSIAN fit to the histogram
                        if cGaussian:
                            
                            if np.isnan(np.min(flux)):
                                Gaussian_flux = Gaussian_noise = GaussNw_flux = GaussNw_noise = float('nan')

                                if verbose >= 3:
                                    print("    flux of Gaussian     = masked")
                                    print("    flux of Gauss (sel)  = masked")

                            else:
                                Gaussian_flux, Gaussian_noise, GaussNw_flux, GaussNw_noise = c_Gaussian(flux, rms_noise)

                                if verbose >= 3:
                                    print("    flux of Gaussian     = " + str(int(Gaussian_flux*1.e5)/1.e5) + " +/- " + str(int(Gaussian_noise*1.e5)/1.e5))
                                    print("    flux of Gauss (sel)  = " + str(int(GaussNw_flux*1.e5)/1.e5) + " +/- " + str(int(GaussNw_noise*1.e5)/1.e5))

                            continuum_flux_Gaussian[ypix].append(Gaussian_flux)
                            continuum_noise_Gaussian[ypix].append(Gaussian_noise)
                            continuum_flux_GaussNw[ypix].append(GaussNw_flux)
                            continuum_noise_GaussNw[ypix].append(GaussNw_noise)

                        # Determine CONTINUUM using a corrected version of SIGMA-CLIPPING
                        if csigmaclip:

                            if np.isnan(np.min(flux)):
                                sigmaclip_flux_prev = sigmaclip_flux = sigmaclip_noise = float('nan')

                                if verbose >= 3:
                                    print("    flux of sigma-clip   = masked")
                                    print("    flux of c-sigma-clip = masked")

                            else:
                                if betaversion is False:
                                    sigmaclip_flux_prev, sigmaclip_flux, sigmaclip_noise = c_sigmaclip1D(flux, rms_noise, betaversion)
                                
                                if betaversion:
                                    sigmaclip_flux_prev, sigmaclip_flux, sigmaclip_noise, real_fraction_emission, fraction_emission, real_fraction_absorption, fraction_absorption = c_sigmaclip1D(flux, rms_noise, betaversion)
                                    if verbose >= 1:
                                        print(" < DEVELOPERS > ")
                                        print(" < DEVELOPERS > ... data and filtered data available in file statcont-developers/STATCONT_sigmaclip_filtered.dat")

                                if verbose >= 3:
                                    print("    flux of sigma-clip   = " + str(int(sigmaclip_flux_prev*1.e5)/1.e5) + " +/- " + str(int(sigmaclip_noise*1.e5)/1.e5))
                                    print("    flux of c-sigma-clip = " + str(int(sigmaclip_flux*1.e5)/1.e5) + " +/- " + str(int(sigmaclip_noise*1.e5)/1.e5))

                            continuum_flux_sigmaclip_prev[ypix].append(sigmaclip_flux_prev)
                            continuum_flux_sigmaclip[ypix].append(sigmaclip_flux)
                            continuum_noise_sigmaclip[ypix].append(sigmaclip_noise)

                        # Create plots with spectra and different continuum levels
                        if plots:
                            if np.isnan(np.min(flux)):
                                if verbose >= 3:
                                    print("    ... masked pixel, no plot produced")

                            else:
                                fig_file = plots_path + tmp_file + '_' + str(xpix+1) + '_' + str(ypix+1) + '.png'
                                fig1 = plt.figure()
                                gs = gridspec.GridSpec(3,1)
                                ax1 = fig1.add_subplot(gs[0,0])
                                ax1.axis('off')
                                ax2 = fig1.add_subplot(gs[1:3,0])

                                ax2.plot(freq,flux, 'k-')
                                plt.xlim(freq.min(), freq.max())
                                plt.ylim(flux.min(), flux.max())
                                ax1.set_title('Spectrum and continuum level at pixel (' + str(xpix+1) + ',' + str(ypix+1) + ')')
                                if cmax:
                                    ax2.axhline(y=maximum_flux, linestyle='--', color='green', linewidth=1.5)
                                    if iname or ifile:
                                        ax1.text(0.0, 0.9, "Maximum = " + str(int(maximum_flux*1.e5)/1.e5) + " " + bunit)
                                    if ispec or ilist:
                                        ax1.text(0.0, 0.8, "Maximum = " + str(int(maximum_flux*1.e5)/1.e5))
                                if cmean:
                                    ax2.axhline(y=mean_flux, linestyle='--', color='orange', linewidth=1.5)
                                    ax2.axhline(y=meansel_flux, linestyle='--', color='yellow', linewidth=1.5)
                                    if iname or ifile:
                                        ax1.text(0.0, 0.6, "Mean = " + str(int(mean_flux*1.e5)/1.e5) + " " + bunit)
                                        ax1.text(0.0, 0.4, "Mean (sel.) = " + str(int(meansel_flux*1.e5)/1.e5) + " " + bunit)
                                    if ispec or ilist:
                                        ax1.text(0.0, 0.6, "Mean = " + str(int(mean_flux*1.e5)/1.e5))
                                        ax1.text(0.0, 0.4, "Mean (sel.) = " + str(int(meansel_flux*1.e5)/1.e5))
                                if cmedian:
                                    ax2.axhline(y=median_flux, linestyle='--', color='orange', linewidth=1.5)
                                    ax2.axhline(y=mediansel_flux, linestyle='--', color='yellow', linewidth=1.5)
                                    if iname or ifile:
                                        ax1.text(0.0, 0.2, "Median = " + str(int(median_flux*1.e5)/1.e5) + " " + bunit)
                                        ax1.text(0.0, 0.0, "Median (sel.) = " + str(int(mediansel_flux*1.e5)/1.e5) + " " + bunit)
                                    if ispec or ilist:
                                        ax1.text(0.0, 0.2, "Median = " + str(int(median_flux*1.e5)/1.e5))
                                        ax1.text(0.0, 0.0, "Median (sel.) = " + str(int(mediansel_flux*1.e5)/1.e5))
                                if cpercent:
                                    ax2.axhline(y=percent25_flux, linestyle='--', color='red', linewidth=1.5)
                                    ax2.axhline(y=percent75_flux, linestyle='--', color='red', linewidth=1.5)
                                    if iname or ifile:
                                        ax1.text(0.4, 0.8, "Percent 25th = " + str(int(percent25_flux*1.e5)/1.e5) + " " + bunit)
                                    if ispec or ilist:
                                        ax1.text(0.4, 0.8, "Percent 25th = " + str(int(percent25_flux*1.e5)/1.e5))
                                if cKDEmax:
                                    ax2.axhline(y=KDEmax_flux, linestyle='-', color='black', linewidth=1.5)
                                    if iname or ifile:
                                        ax1.text(0.4, 0.6, "KDE max = " + str(int(KDEmax_flux*1.e5)/1.e5) + " " + bunit)
                                    if ispec or ilist:
                                        ax1.text(0.4, 0.6, "KDE max = " + str(int(KDEmax_flux*1.e5)/1.e5))
                                if cGaussian:
                                    ax2.axhline(y=Gaussian_flux, linestyle='-', color='blue', linewidth=3.0, alpha=0.5)
                                    ax2.axhline(y=GaussNw_flux, linestyle='-', color='cyan', linewidth=3.0, alpha=0.5)
                                    if iname or ifile:
                                        ax1.text(0.4, 0.4, "Gaussian = " + str(int(Gaussian_flux*1.e5)/1.e5) + " " + bunit + " (+/- " + str(int(Gaussian_noise*1.e5)/1.e5) + ")")
                                        ax1.text(0.4, 0.2, "Gaussian (sel.) = " + str(int(GaussNw_flux*1.e5)/1.e5) + " " + bunit + " (+/- " + str(int(GaussNw_noise*1.e5)/1.e5) + ")")
                                    if ispec or ilist:
                                        ax1.text(0.4, 0.4, "Gaussian = " + str(int(Gaussian_flux*1.e5)/1.e5) + " (+/- " + str(int(Gaussian_noise*1.e5)/1.e5) + ")")
                                        ax1.text(0.4, 0.2, "Gaussian (sel.) = " + str(int(GaussNw_flux*1.e5)/1.e5) + " (+/- " + str(int(GaussNw_noise*1.e5)/1.e5) + ")")
                                if csigmaclip or continuum:
                                    ax2.axhline(y=sigmaclip_flux, linestyle='-', color='red', linewidth=1.5)
                                    if iname or ifile:
                                        ax1.text(0.4, 0.0, "corrSigma-clip = " + str(int(sigmaclip_flux*1.e5)/1.e5) + " " + bunit + " (+/- " + str(int(sigmaclip_noise*1.e5)/1.e5) + ")")
                                    if ispec or ilist:
                                        ax1.text(0.4, 0.0, "corrSigma-clip = " + str(int(sigmaclip_flux*1.e5)/1.e5) + " (+/- " + str(int(sigmaclip_noise*1.e5)/1.e5) + ")")
                                plt.xlabel('Frequency')
                                plt.ylabel('Intensity')

                                fig1.savefig(fig_file)
                                plt.close(fig1)

                # Write the output continuum file
                if verbose >= 1:
                    print(" ")
                    print("... CONTINUUM FILEs CREATED: ")

                # Append new output files with the CONTINUUM estimated values
                if cmax:

                    output_files.append(cont_path + tmp_file + '_continuum_maximum' + extension)
                    output_fluxs.append(continuum_flux_maximum)

                if cmean:

                    output_files.append(cont_path + tmp_file + '_continuum_mean' + extension)
                    output_fluxs.append(continuum_flux_mean)
                    output_files.append(cont_path + tmp_file + '_continuum_meansel' + extension)
                    output_fluxs.append(continuum_flux_meansel)

                if cmedian:

                    output_files.append(cont_path + tmp_file + '_continuum_median' + extension)
                    output_fluxs.append(continuum_flux_median)
                    output_files.append(cont_path + tmp_file + '_continuum_mediansel' + extension)
                    output_fluxs.append(continuum_flux_mediansel)

                if cpercent:

                    output_files.append(cont_path + tmp_file + '_continuum_percent25' + extension)
                    output_fluxs.append(continuum_flux_percent25)
                    output_files.append(cont_path + tmp_file + '_continuum_percent75' + extension)
                    output_fluxs.append(continuum_flux_percent75)

                if cKDEmax:

                    output_files.append(cont_path + tmp_file + '_continuum_KDEmax' + extension)
                    output_fluxs.append(continuum_flux_KDEmax)

                if cGaussian:

                    output_files.append(cont_path + tmp_file + '_continuum_Gaussian' + extension)
                    output_fluxs.append(continuum_flux_Gaussian)
                    output_files.append(cont_path + tmp_file + '_noise_Gaussian' + extension)
                    output_fluxs.append(continuum_noise_Gaussian)
                    output_files.append(cont_path + tmp_file + '_continuum_GaussNw' + extension)
                    output_fluxs.append(continuum_flux_GaussNw)
                    output_files.append(cont_path + tmp_file + '_noise_GaussNW' + extension)
                    output_fluxs.append(continuum_noise_GaussNw)

                if csigmaclip:

                    output_files.append(cont_path + tmp_file + '_continuum_sigmaclip' + extension)
                    output_fluxs.append(continuum_flux_sigmaclip_prev)
                    output_files.append(cont_path + tmp_file + '_noise_sigmaclip' + extension)
                    output_fluxs.append(continuum_noise_sigmaclip)

            for output_file, output_flux in zip(output_files, output_fluxs):
                if verbose >= 1:
                    print("  . " + output_file)
                os.system('rm -rf ' + output_file)
                if iname or ifile:
                    fits.writeto(output_file, np.float32(output_flux), header=header, overwrite=True)
                if ispec or ilist:
                    output_array = [np.median(freq)]
                    if type(output_flux) != list:
                        output_array.append(output_flux)
                    if type(output_flux) == list:
                        output_array.append(output_flux[0][0])
                    np.savetxt(output_file, output_array, newline=" ")

            # For statistics (to be removed)
            if betaversion and cmean and cKDEmax and cGaussian and csigmaclip:
                print('%4.1f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f' % (rms_noise, 50, int(mean_flux*1.e5)/1.e5, int(KDEmax_flux*1.e5)/1.e5, int(Gaussian_flux*1.e5)/1.e5, int(Gaussian_noise*1.e5)/1.e5, int(GaussNw_flux*1.e5)/1.e5, int(GaussNw_noise*1.e5)/1.e5, int(sigmaclip_flux_prev*1.e5)/1.e5, int(sigmaclip_noise*1.e5)/1.e5, int(real_fraction_emission*1.e5)/1.e5, int(real_fraction_absorption*1.e5)/1.e5, int(fraction_emission*1.e5)/1.e5, int(fraction_absorption*1.e5)/1.e5) )
                    
    if continuum:
        
        # Re-set the variables to individual files if --imerge is used
        if imerge:

            merged_continuum_file = cont_path + tmp_file + '_continuum' + extension
            merged_noise_file = cont_path + tmp_file + '_noise' + extension
            tmp_path = unmerged_path
            tmp_files = unmerged_files

        for tmp_file in tmp_files:

            # Copy the merged continuuum file to individual continuum files
            # for each one of the files used during the merging
            if imerge:

                os.system('cp -rp ' + merged_continuum_file + ' ' + cont_path + tmp_file + '_continuum' + extension)
                os.system('cp -rp ' + merged_noise_file + ' ' + cont_path + tmp_file + '_noise' + extension)
                if verbose >= 1:
                    print("  . " + cont_path + tmp_file + '_continuum' + extension)
                    print("  . " + cont_path + tmp_file + '_noise' + extension)

            # Subtract continuum to the original (line+continuum) data file
            # and produce a line-only and a continuum-only file
            if cfree is True:

                if verbose >= 1:
                    print("")
                    print("+++ REMOVING CONTINUUM FROM DATA ... " + tmp_path + tmp_file + extension)

                # Select the original line+continuum file and the created continuum file
                cube_file = tmp_path + tmp_file + extension
                cont_files = []
                cont_files.append(tmp_file + '_continuum')

                for cont_file in cont_files:

                    # For ASCII files
                    if ispec or ilist:

                        fdata_cont = open(cont_path + cont_file + extension, 'r')

                        for line in fdata_cont:

                            data_cont = float(line.split()[1])

                        line_outfile = line_path + cont_file + '.line' + extension
                        ascii.write((freqs, flux[:]-data_cont), output=line_outfile, overwrite=True)

                        if verbose >1:
                            print(" ")
                            print("... FILEs CREATED: ")
                            print("  . " + line_outfile)

                    # For FITS files
                    if iname or ifile:

                        data_cube = fits.getdata(cube_file)
                        header_cube = fits.getheader(cube_file)
                        data_cont = fits.getdata(cont_path + cont_file + extension)
                        header_cont = fits.getheader(cont_path + cont_file + extension)

                        # If --nooffset is selected, try to remove the offset from the map
                        if nooffset:
                            
                            if verbose >= 1:
                                print("... removing general continuum offset")
                            nxpix = header_cube.get('NAXIS1')
                            nypix = header_cube.get('NAXIS2')
                            rmsxpix = int(nxpix / 8)
                            rmsypix = int(nypix / 8)
                            nxpixmin = 16
                            nypixmin = 16

                            # Calculate the rms noise level in different regions throughout the continuum FITS file
                            if nxpix > nxpixmin and nypix > nypixmin:
                                rms = []

                                rms.append(np.mean(data_cont[rmsypix*1:rmsypix*2,rmsxpix*1:rmsxpix*2]))
                                rms.append(np.mean(data_cont[rmsypix*1:rmsypix*2,rmsxpix*2:rmsxpix*3]))
                                rms.append(np.mean(data_cont[rmsypix*1:rmsypix*2,rmsxpix*5:rmsxpix*6]))
                                rms.append(np.mean(data_cont[rmsypix*1:rmsypix*2,rmsxpix*6:rmsxpix*7]))

                                rms.append(np.mean(data_cont[rmsypix*2:rmsypix*3,rmsxpix*1:rmsxpix*2]))
                                rms.append(np.mean(data_cont[rmsypix*2:rmsypix*3,rmsxpix*2:rmsxpix*3]))
                                rms.append(np.mean(data_cont[rmsypix*2:rmsypix*3,rmsxpix*5:rmsxpix*6]))
                                rms.append(np.mean(data_cont[rmsypix*2:rmsypix*3,rmsxpix*6:rmsxpix*7]))

                                rms.append(np.mean(data_cont[rmsypix*5:rmsypix*6,rmsxpix*1:rmsxpix*2]))
                                rms.append(np.mean(data_cont[rmsypix*5:rmsypix*6,rmsxpix*2:rmsxpix*3]))
                                rms.append(np.mean(data_cont[rmsypix*5:rmsypix*6,rmsxpix*5:rmsxpix*6]))
                                rms.append(np.mean(data_cont[rmsypix*5:rmsypix*6,rmsxpix*6:rmsxpix*7]))

                                rms.append(np.mean(data_cont[rmsypix*6:rmsypix*7,rmsxpix*1:rmsxpix*2]))
                                rms.append(np.mean(data_cont[rmsypix*6:rmsypix*7,rmsxpix*2:rmsxpix*3]))
                                rms.append(np.mean(data_cont[rmsypix*6:rmsypix*7,rmsxpix*5:rmsxpix*6]))
                                rms.append(np.mean(data_cont[rmsypix*6:rmsypix*7,rmsxpix*6:rmsxpix*7]))

                                if verbose >= 1:
                                    print("  . correcting the continuum by %f " % (np.absolute(np.median(rms))))
                                data_finalcont = data_cont - np.median(rms)
                                data_line = data_cube - (data_cont - np.median(rms))
                                #data_finalcont = data_cont + np.absolute(np.median(rms))
                                #data_line = data_cube - (data_cont + np.absolute(np.median(rms)))

                            # If the size of the map is too small (less than 30x30 pixels)
                            # no rms noise level is subtracted
                            else:
                                if verbose >= 1:
                                    print("  . WARNING: The image has less than %i x %i pixels" % (nxpixmin, nypixmin))
                                    print("  .          No residual noise level subtracted for")
                                    print("  .          %s " % (cube_file))
                                
                                data_finalcont = data_cont
                                data_line = data_cube - data_cont

                        else:
                            data_finalcont = data_cont
                            data_line = data_cube - data_cont

                        cont_outfile = cont_path + cont_file + '.cont' + extension
                        line_outfile = line_path + cont_file + '.line' + extension

                        os.system('rm -rf ' + cont_outfile)
                        fits.writeto(cont_outfile, np.float32(data_finalcont), header=header_cont, overwrite=True)

                        os.system('rm -rf ' + line_outfile)
                        fits.writeto(line_outfile, np.float32(data_line), header=header_cube, overwrite=True)

                        # Create a line-only cube
                        # and replace the old continuum image with the new one
                        if nooffset:
                            os.system('mv ' + cont_path + cont_file + extension + ' ' + cont_path + cont_file + '_original' + extension)
                        os.system('mv ' + cont_outfile + ' ' + cont_path + cont_file + extension)
                        os.system('mv ' + line_outfile + ' ' + cont_path + tmp_file + '_line' + extension)

                        if verbose >= 1:
                            print(" ")
                            print("... FILEs CREATED: ")
                        if nooffset:
                            if verbose >= 1:
                                print("  . " + cont_path + tmp_file + '_continuum' + extension)
                                print("  . " + cont_path + tmp_file + '_noise' + extension)
                        if verbose >= 1:
                            print("  . " + cont_path + tmp_file + '_line' + extension)

    # Combine several continuum files to determine the spectral index
    # it also combines them and creates a continuum model, and line+continuum model
    if spindex:

        if verbose >= 1:
            print(" ")
            print("+++ DETERMINING SPECTRAL INDEX ...")

        # Combine all the continuum images in one single cube
        if extension=='.dat':
            # Prepare ASCII file to combine all the continuum files 
            tmp_merged_continuum_file = open(cont_path + 'tmp_merged_continuum' + extension, "w")
        
        if extension=='.fits':
            my_frequency = []
            my_cube = np.empty([len(tmp_files),nypix,nxpix])
            icount = 0

        for tmp_file in tmp_files:

            if extension=='.dat':
                # merge all continuum files in one single one
                f = open(cont_path + tmp_file + '_continuum' + extension)
                tmp_merged_continuum_file.write("%s\n" % f.readline())
                
            if extension=='.fits':
                contcube_data = fits.getdata(cont_path + tmp_file + '_continuum' + extension)
                contcube_header = fits.getheader(cont_path + tmp_file + '_continuum' + extension)
                my_frequency.append(contcube_header.get('RESTFRQ'))
                my_cube[icount,:,:] = contcube_data
                icount = icount+1

        if extension=='.dat':
            tmp_merged_continuum_file.close()
            tmpcontfile = open(cont_path + 'tmp_merged_continuum' + extension)
            lmy_frequency = []
            lmy_continuum = []
            for line in tmpcontfile:
                lmy_frequency.append(float(line.split()[0]))
                lmy_continuum.append(float(line.split()[1]))
            my_frequency = np.array(lmy_frequency)
            my_continuum = np.array(lmy_continuum)
            my_continuum[my_continuum<1.e-10] = 1.e-10
            y = np.log10(my_continuum)
            x = np.log10(my_frequency)
            z = np.polyfit(x, y, 1)
            m = z[0]
            n = z[1]
            
            cont_m_file = cont_path + source + '_spindex.dat'
            cont_n_file = cont_path + source + '_intercept.dat'
            
            os.system('rm -rf ' + cont_m_file)
            out_file = open(cont_m_file, "w")
            out_file.write(str(m))
            os.system('rm -rf ' + cont_n_file)
            out_file = open(cont_n_file, "w")
            out_file.write(str(n))

        if extension=='.fits':
            # Apply blanking value of 1.e-10
            my_cube[my_cube<1.e-10] = 1.e-10
            my_frequency = np.array(my_frequency)

            # Fit all the continuum images and determine the spectral index (m)
            # following expression:  flux = A x frequency ^ spindex
            # and in logarithmic version:  log(flux) = log(A) + spindex * log(frequency)
            m = []
            n = []

            for ypix in range(nypix):

                if verbose >= 1:
                    if iname or ifile:
                        print("... analyzing column " + str(ypix+1) + " out of " + str(nypix))

                m.append([])
                n.append([])

                for xpix in range(nxpix):

                    y = np.log10(my_cube[:,ypix,xpix])
                    x = np.log10(my_frequency)
                    z = np.polyfit(x, y, 1)
                    m[ypix].append(z[0])
                    n[ypix].append(z[1])

            cont_m_file = cont_path + source + '_spindex.fits'
            cont_n_file = cont_path + source + '_intercept.fits'

            os.system('rm -rf ' + cont_m_file)
            fits.writeto(cont_m_file, np.float32(m), header=header, overwrite=True)

            os.system('rm -rf ' + cont_n_file)
            fits.writeto(cont_n_file, np.float32(n), header=header, overwrite=True)

        # Indicate where the created files can be found
        if verbose >= 1:
            print("... FILEs CREATED are found in " + cont_path)
            print("  . search for spindex and intercept")
            print(" ")

        # Create a continuum model (from the spectral index)
        # and a line+continuum cube, using the continuum model
        if model:

            if verbose >= 1:
                print(" ")
                print("+++ PRODUCING MODEL USING THE SPECTRAL INDEX ...")

            if extension=='.dat':

                #model_conts = []
                #model_freqs = []
                for tmp_file in tmp_files:

                    # Create a frequency array from the original data cube
                    f = open(cont_path + tmp_file + '_continuum' + extension)
                    for line in f:
                        print("Frequency: " + str(float(line.split()[0])) + "  Continuum: " + str(float(line.split()[1])))
                        #model_conts.append(float(line.split()[1]))
                        #model_freqs.append(float(line.split()[0]))

                #model_cont = np.array(model_conts)
                #model_freq = np.array(model_freqs)

                #cont_model_file = cont_path + tmp_file + '_cont_model' + extension
                #ascii.write((model_freq, model_cont), output=cont_model_file, overwrite=True)

            if extension=='.fits':
                # Read the spectral index (m) and the intercept (n)
                m_data = fits.getdata(cont_m_file)
                n_data = fits.getdata(cont_n_file)

                for tmp_file in tmp_files:

                    # Create a frequency array from the original data cube
                    cube_data =  fits.getdata(tmp_path + tmp_file + extension)
                    cube_header = fits.getheader(tmp_path + tmp_file + extension)
                    frequency_array = []

                    for i in range(cube_header['NAXIS3']):

                        frequency_array.append(cube_header['CRVAL3'] + (i - cube_header['CRPIX3'])*cube_header['CDELT3'])

                    frequency_array = np.array(frequency_array)

                    # Read the continuum real emission (from --continuum)
                    cont_real = fits.getdata(cont_path + tmp_file + '_continuum.fits')

                    # Generate the continuum model (using the n, m parameters)
                    if verbose >= 1:
                        print("... creating continuum model of " + tmp_file)
                    cont_model = np.power(10, n_data[np.newaxis, :, :] + (np.log10(frequency_array[:,np.newaxis, np.newaxis]) * m_data[np.newaxis, :, :]))
                    cont_model_file = cont_path + tmp_file + '_cont_model' + extension
                    os.system('rm -rf ' + cont_model_file)
                    fits.writeto(cont_model_file, np.float32(cont_model), header=cube_header, overwrite=True)

                    # Compute the factor between real and model continuum
                    factor = cont_model[:, :, :] / cont_real[np.newaxis, :, :]
                    factor[cont_real < 0.0] = -1.0e10
                    factor[factor > 2.0] = -1.0e10
                    factor[factor < 0.5] = -1.0e10
                    factor_file = cont_path + tmp_file + '_cont_factor' + extension
                    os.system('rm -rf ' + factor_file)
                    fits.writeto(factor_file, np.float32(factor), header=cube_header, overwrite=True)

                    # Compute the line and continuum model
                    if verbose >= 1:
                        print("... creating the line+continuum model of " + tmp_file)
                    line_cont_model = cube_data[:, :, :, :] * factor[np.newaxis, :, :, :]
                    line_cont_model[factor < -1.0e7] = cube_data[:, :, :, :] + cont_model[np.newaxis, :, :, :]
                    line_cont_model_file = cont_path + tmp_file + '_line_cont_model' + extension
                    os.system('rm -rf ' + line_cont_model_file)
                    fits.writeto(line_cont_model_file, np.float32(line_cont_model), header=cube_header, overwrite=True)

                    # Compute the line only model data
                    line_model = line_cont_model - cont_model
                    line_model_file = cont_path + tmp_file + '_line_model' + extension
                    os.system('rm -rf ' + line_model_file)
                    fits.writeto(line_model_file, np.float32(line_model), header=cube_header, overwrite=True)

            # Indicate where the created files can be found
            if verbose >= 1 and extension=='.fits':
                print("... FILEs CREATED are found in " + cont_path)
                print("  . search for cont_model and line_cont_model")
                print(" ")
        
        # Delete the temporal ASCII file
        os.system('rm ' + cont_path + 'tmp_merged_continuum' + extension)
