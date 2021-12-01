"""
    NPFields.ScalarField - Functions for dealing with scalar fields
    Copyright (C) 2021 Amélia O. F. da S.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import scipy.fft
import numpy as np
def partial_derivative(scalarfield:np.ndarray,axis:int)->np.ndarray:
    """
    Partial derivative of a field along an axis (Finite differences)

    Parameters
    ----------
    scalarfield : np.ndarray of shape (x1,x2,x3,...,xn)
        The scalar field to differentiate
    axis : int between 0 and n
        Which axis to differentiate on

    Returns
    -------
    np.ndarray of shape (x1,x2,x3,...,xn)
        The derivative at each point
    """
    f = np.zeros(scalarfield.shape)
    x0 = tuple([slice(None) if dimension!=axis else slice(1,-1) for dimension in range(len(scalarfield.shape))])
    x1 = tuple([slice(None) if dimension!=axis else slice(2,None) for dimension in range(len(scalarfield.shape))])
    x_1 = tuple([slice(None) if dimension!=axis else slice(0,-2) for dimension in range(len(scalarfield.shape))])
    f[x0] = ((scalarfield[x1]-scalarfield[x_1]))/2
    return f
def gradient(scalarfield:np.ndarray)->np.ndarray:
    """
    Gradient of a scalar field. (Finite Differences)
    
    Parameters
    ----------
    scalarfield : np.ndarray of shape (x1,x2,x3,...,xn)
        The scalar field to differentiate

    Returns
    -------
    np.ndarray of shape (x1,x2,x3,...,xn,n)
        The gradient vector at each point
    """
    grad = np.zeros(tuple(list(scalarfield.shape)+[len(scalarfield.shape)]))
    alldims = [slice(None) for _ in scalarfield.shape]
    for dimension in range(len(scalarfield.shape)):
        grad[tuple(alldims+[dimension])] = partial_derivative(scalarfield,dimension)
    return grad
def partial_derivative_fourier(complex_scalarfield:np.ndarray,axis:int,precalc=True)->np.ndarray:
    """
    Partial derivative of a field along an axis
    (Uses fourier transforms for smoothness)

    Parameters
    ----------
    complex_scalarfield : np.ndarray of shape (x1,x2,x3,...,xn)
        The Fourier transform (assumed to follow the scipy.fft.fft format)
        of the scalar field to differentiate if "precalc" is true.
        Otherwise, just a scalar field.
    axis : int between 0 and n
        Which axis to differentiate on
    precalc: bool
        Whether the field is in pure form or transformed already

    Returns
    -------
    np.ndarray of shape (x1,x2,x3,...,xn)
        The derivative at each point
    """
    if not precalc:
        complex_scalarfield = scipy.fft.fftn(complex_scalarfield)
    freqs = scipy.fft.fftfreq(complex_scalarfield.shape[axis],2*np.pi/(complex_scalarfield.shape[axis]-1))
    freqs = freqs[tuple([np.newaxis if ax!=axis else slice(None) for ax in range(len(complex_scalarfield.shape))])]
    return (scipy.fft.ifftn(
        2*np.pi*1j*complex_scalarfield*freqs
    )).real
def gradient_fourier(scalarfield:np.ndarray,precalc=False)->np.ndarray:
    """
    Gradient of a scalar field.
    (Uses fourier transforms for smoothness)
    
    Parameters
    ----------
    scalarfield : np.ndarray of shape (x1,x2,x3,...,xn)
        The scalar field to differentiate
    precalc : bool
        Whether the field is already transformed

    Returns
    -------
    np.ndarray of shape (x1,x2,x3,...,xn,n)
        The gradient vector at each point
    """
    grad = np.zeros(tuple(list(scalarfield.shape)+[len(scalarfield.shape)]))
    trans = scalarfield if precalc else scipy.fft.fftn(scalarfield)
    alldims = [slice(None) for _ in scalarfield.shape]
    for dimension in range(len(scalarfield.shape)):
        grad[tuple(alldims+[dimension])] = partial_derivative_fourier(trans,dimension)
    return grad