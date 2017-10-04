#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
JPEG
'''
import numpy as np
np.set_printoptions(suppress=True)


#completes the first few steps of JPEG encoding (all lossy steps), then reproduces a decoded (and degraded) image
#qualityLevel = ranged zero to 100
def jpeg_encode_decode (imgOrig, qualityLevel):
    # ----------------------------
    # --generate encoding matrix--
    # ----------------------------
    dct = get_DCT()
    dctInv = np.linalg.inv(dct)
    Q_jpg = get_Q(qualityLevel)
    Qinv_jpg = np.linalg.inv(Q_jpg)
    
    # ------------------
    # --begin encoding--
    # ------------------
    imgEncodedJPEG = np.rint(Q_jpg * dct * imgOrig)
    
    # ------------------
    # --begin decoding--
    # ------------------
    imgDecodedJPEG = dctInv * Qinv_jpg * imgEncodedJPEG
    imgDecodedJPEG = np.where(imgDecodedJPEG<-127.5,-127.5,imgDecodedJPEG)
    imgDecodedJPEG = np.where(imgDecodedJPEG>127.5,127.5,imgDecodedJPEG)
    imgDecodedJPEG = np.rint(imgDecodedJPEG)
    
    return imgEncodedJPEG, imgDecodedJPEG


#standard JPEG Q matrix - the relative quality of each DCT coefficient. See JPEG documentation for details.
def get_Q(quality=50):
    Q_orig = np.array([[16, 11, 10, 16,  24,  40,  51,  61,
                        12, 12, 14, 19,  26,  58,  60,  55,
                        14, 13, 16, 24,  40,  57,  69,  56,
                        14, 17, 22, 29,  51,  87,  80,  62,
                        18, 22, 37, 56,  68, 109, 103,  77,
                        24, 35, 55, 64,  81, 104, 113,  92,
                        49, 64, 78, 87, 103, 121, 120, 101,
                        72, 92, 95, 98, 112, 100, 103,  99]],dtype=np.float)

    if quality < 50:
        S = 5000.0/quality
    else:
        S = 200.0 - 2*quality

    Q_orig = np.floor((S*Q_orig + 50.0) / 100.0)
    Q_orig = np.where(Q_orig<1.0, 1, Q_orig)
    
    Q_jpg = np.linalg.inv(np.matrix(np.diag(Q_orig.ravel())))
    return Q_jpg


#compute DCT transform matrix, which projects a small image represented by pixels into one represented by DCT coefficient weights
def get_DCT(N=8):
    result = np.zeros((N*N,N*N))
    for u in range(N):
        for v in range(N):
            for x in range(N):
                for y in range(N):
                    result[N*u+v,N*x+y] = np.cos(((2*x+1)*u*np.pi)/float(2*N)) * np.cos(((2*y+1)*v*np.pi)/float(2*N))
    result = np.multiply(0.25,result)
    result[0:N,:] = np.multiply(1.0/np.sqrt(2.0),result[0:N,:])
    indx = N*np.array(range(0,N))
    result[indx,:] = np.multiply(1.0/np.sqrt(2.0),result[indx,:])
    return np.matrix(result)