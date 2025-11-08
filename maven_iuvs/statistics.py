import numpy as np
from sklearn import linear_model
import math

def chisquared(LL, sigma, p=0, reduced=False):
    """
    Computes chisquared given some log likelihood.
    Parameters
    ----------
        LL : int or float
             Log likelihood computed by fit algorithm
        sigma : array
                vector of data uncertainties for every bin
        p : int
            number of model parameters, used to compute reduced chisq
        reduced : bool
                  False > returns normal chi squared. True > returns reduced

    Returns
    ----------
    chisq : float
            chi squared value, normal or reduced.
    """
    N = len(sigma)

    chisq = -2 * (LL + (N/2)*np.log(2*math.pi) +np.sum(np.log(sigma)))
    # Note that the middle term reduces to N log(σ) in the event all σ are the same.
    if reduced:
        return chisq * (1/(N-p))
    return chisq




def multiple_linear_regression(templates, spectrum, spectrum_error):
    """
    Fits an array of templates to a spectrum using multiple linear regression (MLR).
    
    Parameters
    ----------
    templates : list, arr
        Templates to fit to the spectrum in DN.
    spectrum : list, arr
        An observed spectrum in DN.
    spectrum_error : list, arr
        The uncertainty on the spectrum values in DN.
        
    Returns
    -------
    coeff : float, arr
        The fit coefficients for the templates
    const : float
        The constant term from the fitting.
    """

    # ensure input templates are an array with two dimensions
    X = np.array(templates)
    if X.ndim == 1:
        np.expand_dims(X, axis=0)

    # transpose templates
    X = X.T

    # ensure spectrum and error are numpy arrays
    Y = np.array(spectrum)
    Yerr = np.array(spectrum_error)

    # convert uncertainty to sample weight
    Yw = (1/Yerr)**2

    # make a linear regression model and fit the spectra with templates
    fit = linear_model.LinearRegression().fit(X, Y, sample_weight=Yw)

    # extract the coefficients and constant term
    coeff = fit.coef_
    const = fit.intercept_

    # return the coefficients and constant term
    return coeff, const


def integrate_intensity(template_wavelength, template_spectrum, calibration_curve, coefficient):
    """
    Takes a spectrum template and MLR coefficient and calculates an integrated intensity.
    
    Parameters
    ----------
    template_wavelength : array
        Spectrum wavelengths.
    template_spectrum : array
        Spectrum in DN.
    calibration_curve : array
        Conversion from DN to physical units. Assumes format is [DN/unit].
    coefficient : float
        MLR fit coefficient.
        
    Returns
    -------
    integrated_intensity : float
        Integrated template intensity in physical units.
    """

    # determine spectral bin spacing
    dwavelength = np.diff(template_wavelength)[0]

    # calibrate the template
    calibrated_template_spectrum = coefficient * template_spectrum / calibration_curve

    # integrate the template
    integrated_intensity = np.trapz(calibrated_template_spectrum, dx=dwavelength)

    # return the integrated intensity
    return integrated_intensity
