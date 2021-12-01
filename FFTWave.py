"""
    Simple Wave simulaion for demonstrating NPFields' funcionalities
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

import cv2
import numpy as np
import NPFields.ScalarField
from scipy.fft import fftn

cv2.namedWindow("MAIN",cv2.WINDOW_KEEPRATIO)

field = np.random.random((300,300))*.1-.05
xx,yy = np.meshgrid(np.arange(field.shape[0]),np.arange(field.shape[1]))
field[((xx-((field.shape[0]-1)/2))**2+(yy-((field.shape[1]-1)/2))**2)<((field.shape[0]*.3)**2)] = .5

delta = np.zeros(field.shape)

bigger = np.zeros((field.shape[0]+200,field.shape[1]+200))

def posnegnor(field:np.ndarray,nam="Un",mi=-1,ma=1,maxi=False):
    r = np.ones(field.shape)/2
    neg = field[field<0]
    if len(neg)>0:
        mi=mi if maxi else neg.min()
        r[field<0] = .5*((neg-mi)/(-mi))
    pos = field[field>0]
    if len(pos)>0:
        ma=ma if maxi else pos.max()
        r[field>0] += .5*((pos)/(ma))
    return (r*255).astype(np.uint8)

time = 0

avg = np.zeros(field.shape)
smooth = .98

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('output.mp4',fourcc, 60.0, (field.shape[0]*2,field.shape[1]), True)

while 1:
    pfield = field.copy()
    #field[((xx-((field.shape[0]-1)/2))**2+(yy-((field.shape[1]-1)/2))**2)<((field.shape[0]*.025)**2)] = 2*np.sin(time)

    time+=0.017/3#.01

    #Circle
    #field[((xx-((field.shape[0]-1)/2))**2+(yy-((field.shape[1]-1)/2))**2)>((field.shape[0]*.5)**2)] = 0

    bigger[100:-100,100:-100] = field
    trans = fftn(bigger-bigger.mean())
    gradient = NPFields.ScalarField.gradient_fourier(trans,True)
    divergence = NPFields.ScalarField.partial_derivative_fourier(gradient[:,:,0],0,False)+NPFields.ScalarField.partial_derivative_fourier(gradient[:,:,1],1,False)
    delta += divergence[100:-100,100:-100]/300000
    field += delta

    avg = (smooth)*avg+(1-smooth)*np.abs(field-pfield)
    
    scr = np.zeros((field.shape[0]*2,field.shape[1]),dtype=np.float32)
    scr[:field.shape[0],:] = posnegnor(field)/255
    scr[scr>1]=1
    scr[scr<0]=0
    #cv2.circle(scr,((field.shape[0]-1)//2,(field.shape[1])//2),field.shape[0]//2,(0,0,0),1,cv2.LINE_AA)
    img = 1-(avg/avg.max())
    #img[((xx-((field.shape[0]-1)/2))**2+(yy-((field.shape[1]-1)/2))**2)>((field.shape[0]*.5)**2)] = 0
    scr[field.shape[0]:,:] = img
    cv2.imshow("MAIN",scr.T)

    out.write((scr*255).astype(np.uint8).T[:,:,np.newaxis].repeat(3,axis=2))

    k=cv2.waitKey(1)&0xff
    if k==ord('q'):
        break
out.release()