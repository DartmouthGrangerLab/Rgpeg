#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Loading Library
=========================================================
"""
import numpy as np
np.set_printoptions(suppress=True)


############################
# Mostly modified from QMLibrary
############################

#modified from QMLibrary
def convert_pixel_per_inch_to_pixel_pitch_mm(ppi=100.0):
    inch_per_mm = 1.0/25.4 # 1 inch / 25.4 mm
    return 1.0 / ( ppi * inch_per_mm )

#modified from QMLibrary
def get_degree_per_pixel(screen_pixel_size=0.282, viewing_distance=24.0):
    inch_per_mm = 1.0/25.4 # 1 inch / 25.4 mm
    degree_per_pixel = np.arctan((screen_pixel_size*inch_per_mm)/viewing_distance)*(180/np.pi)    
    return degree_per_pixel

#modified from QMLibrary
def get_degree_per_d(d=900.0, viewing_distance=24.0):
    inch_per_mm = 1.0/25.4 # 1 inch / 25.4 mm
    d_pitch_mm = convert_pixel_per_inch_to_pixel_pitch_mm(d)
    degree_per_d = np.arctan((d_pitch_mm*inch_per_mm)/viewing_distance)*(180/np.pi)    
    return degree_per_d

#modified from QMLibrary
def get_d_as_cov_matrix(dx=900.0, dy=900.0, dxy=np.inf):
    dx = get_degree_per_d(d=dx)
    dy = get_degree_per_d(d=dy)
    dxy = get_degree_per_d(d=dxy)
    cov = np.matrix([[dx,dxy],[dxy,dy]])
    return cov

#modified from QMLibrary
def get_basis_dmatrix_mv(basis=np.matrix(np.eye(64)), J=None, d=None, normalize=True):
    dmatrix = np.matrix(np.zeros(basis.shape))
    g = J.T * np.linalg.inv(d) * J   

    coeff = 1.0
    for i in range(basis.shape[0]):
        for j in range(basis.shape[0]):
            diff = basis[:,i] - basis[:,j]           
            sqdist = (diff.T * g * diff)[0,0]
            dist = coeff * np.exp(-0.5*sqdist)
            dmatrix[i,j] = dist

    # convert to a PDF by normalizing the sum
    if normalize:
        coeff = 1.0/np.sum(dmatrix[27])  # 27 is near the middle of the 8x8 grid
        dmatrix = coeff*dmatrix
    # print np.round(np.sum(dmatrix,axis=0),2).reshape((8,8)) # this shows that each 8x8 grid sums to 1 except near the edges due to boundary
    return dmatrix

#modified from QMLibrary
def get_inner_region(M, width=64, height=64):
    M = np.array(M)
    R = np.zeros((width,height))
    x = int(np.sqrt(M.shape[0]))
    pos = np.arange(width)
    i = 0
    for p in pos:
        MM = M[p,:].reshape((x,x))
        MM[0,:] = np.inf
        MM[-1,:] = np.inf
        MM[:,0] = np.inf
        MM[:,-1] = np.inf
        MM = MM.ravel()
        MM = MM[MM <> np.inf]

        R[i,:] = MM
        i = i + 1

    return np.matrix(R)

#modified from QMLibrary
def get_potential(x=10, y=10):
    R = np.zeros((x, x))
    R[0,:] = 100000000.0
    R[-1,:] = 100000000.0
    R[:,0] = 100000000.0
    R[:,-1] = 100000000.0
    R = np.matrix(np.diag(R.ravel()))
    return R

#modified from QMLibrary
#computes the adjacency matrix
def compute_adjacency(dx, dy, J, basis, multiplier=1.0, normalize=True):
    dxy = np.inf
    d = get_d_as_cov_matrix(dx=dx, dy=dy, dxy=dxy)
    Adj = get_basis_dmatrix_mv(basis=basis, J=J, d=d, normalize=normalize)
    return Adj*multiplier


