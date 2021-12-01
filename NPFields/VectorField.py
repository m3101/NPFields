"""
    NPFields.VectorField - Functions for dealing with vector fields
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

import NPFields.ScalarField as ScalarField
import numpy as np
def partial_derivative(vectorfield:np.ndarray,axis:int,component:int)->np.ndarray:
    """
    d[Field_component]/d[axis]
    Returns a scalar field of shape (x1,x2,x3,...,xn)
    """
    return ScalarField.partial_derivative(
        vectorfield[tuple([slice(None) for _ in vectorfield.shape[:-1]]+[component])],
        axis)
def divergent(vectorfield:np.ndarray):
    """
    Returns the divergent of a vector field
    Returns a scalar field of shape (x1,x2,x3,...,xn)
    """
    field = np.zeros(vectorfield.shape[:-1])
    for dimension in range(len(field.shape)):
        field+=partial_derivative(vectorfield,dimension,dimension)
    return field