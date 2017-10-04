#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
RGPEG
'''
import numpy as np
np.set_printoptions(suppress=True)
from library import get_degree_per_pixel, compute_adjacency, get_inner_region, get_potential


#completes the first few steps of RGPEG encoding (all lossy steps), then reproduces a decoded (and degraded) image
#qualityLevel = zero to 100
#screen_pixel_size = in units of mm
#viewing_distance = in units of inches
#dx = in pixels per inch
#dy = in pixels per inch
#full_basis = boolean
def rgpeg_encode_decode (imgOrig, qualityLevel, screen_pixel_size, viewing_distance, dx, dy, full_basis):
    t = 100.0 - np.round(qualityLevel,2)
    
    # ----------------------------
    # --generate encoding matrix--
    # ----------------------------
    #get hamiltonian
    H = get_hamiltonian_for_dct(screen_pixel_size, viewing_distance, dx, dy, full_basis)
    
    #compute eigendecomposition of the Hamiltonian
    D,V_large = np.linalg.eigh(H)
    V = V_large
    Vinv = np.linalg.inv(V.T)
    if full_basis:
        V = get_inner_region(V.T).T
        Vinv = get_inner_region(Vinv.T).T
    D = D[0:64]
    D = np.abs(D) # python mistakenly treats -0 as a negative number
    if not np.all(D >= 0):  # checking for negative eigenvalues
        raise Exception('Error: Eigenvalues are negative')
    
    # compute update operator
    xmax = stimulus_max(V.T)
    Coeff_max = 2*np.mean(np.array(np.max(V.T * xmax, axis=0))[0])
    t_max = 100.0
    V0 = np.arange(64)*2
    L_mean = np.mean(D-V0)
    c = np.log(Coeff_max) / (L_mean*t_max)
        
    U = np.matrix(np.diag(np.exp(-1.0*c*(D-V0)*t)))
    Uinv = np.linalg.inv(U)
    
    # ------------------
    # --begin encoding--
    # ------------------
    imgEncodedRGPEG = np.rint(U * V.T * imgOrig)

    # ------------------
    # --begin decoding--
    # ------------------
    imgDecodedRGPEG = Vinv * Uinv * imgEncodedRGPEG  # transform to image coordinates
    imgDecodedRGPEG = np.where(imgDecodedRGPEG<-127.5,-127.5,imgDecodedRGPEG)
    imgDecodedRGPEG = np.where(imgDecodedRGPEG>127.5,127.5,imgDecodedRGPEG)
    imgDecodedRGPEG = np.rint(imgDecodedRGPEG)
    
    return imgEncodedRGPEG, imgDecodedRGPEG, U #U is only returned because we want to print it later


#modified from QMLibrary
def get_hamiltonian_for_dct(screen_pixel_size=0.282, viewing_distance=24.0, dx=900.0, dy=900.0, full_basis=False):
    if full_basis:
        nx,ny = 10,10
    else:
        nx,ny = 8,8
    I = np.matrix(np.eye(nx*ny)) #identity matrix
    
    #compute Jacobian projecting from feature to physical
    J = get_jacobian_feature_to_physical(screen_pixel_size=screen_pixel_size, viewing_distance=viewing_distance, ncells=nx)

    #compute adjacency matrix A
    Adj = compute_adjacency(dx=dx, dy=dy, J=J, basis=I, normalize=True)

    #compute Laplacian from subject viewing parameters
    Dg_diag = np.array(np.sum(Adj,axis=0)).squeeze()
    Dg = np.matrix(np.diag(Dg_diag))
    L = Dg - Adj #L is the Laplacian
    L = np.round(L,4)
    
    #compute hamiltonian from subject viewing parameters
    P = 0 # <--- this is the d of luminance
    if full_basis:
        P = get_potential()
    else:
        P = I + I * np.abs(-np.log(1.0/viewing_distance))
    H = L + P # H is the hamiltonian
    
    #compute eigendecomposition of the Hamiltonian
    D,V  = np.linalg.eigh(H) #D = eigenvalues, V = eigenvectors

    X2 = np.matrix(np.diag(np.arange(nx*ny)*2))
    H = H + V * X2 * V.T
    
    return H


#modified from QMLibrary
def get_jacobian_feature_to_physical(screen_pixel_size=0.282, viewing_distance=24.0, ncells=8):
    dpp = get_degree_per_pixel(screen_pixel_size, viewing_distance)
    xspacing = np.linspace(0,ncells-1,ncells)*dpp
    yspacing = np.linspace(0,ncells-1,ncells)*dpp
    J = np.matrix(np.vstack((np.tile(xspacing,ncells),np.repeat(yspacing,ncells))))
    return J


# returns the max stimulus as COLUMNS
def stimulus_max (t):
    sgn = np.where(t>0, 1.0, -1.0)
    result = 127.5 * (1.0+sgn)
    return result.T
