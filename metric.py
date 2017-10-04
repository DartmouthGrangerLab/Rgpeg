#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Metric class
=========================================================

Implements a metric tensor

"""

print(__doc__)

import numpy as np
from scipy.spatial.distance import cdist
from pylab import plot, show

class Metric():
    def __init__(self,width,height,param=(1.0,2.0),dmatrix=None,normalize=False,new=False,coords=None):
        self.new = new
        self.normalize = normalize
        self.param = param
        if dmatrix is None:
            self.dmatrix = self._compute_dmatrix(width,height,coords)
        else:
            self.dmatrix = dmatrix
        self.levels = np.unique(self.dmatrix)[2:]
        self.g, self.J = self.compute_metric_tensor(width,height)
        self.vmin = 0
        self.vmax = 255

    def _compute_dmatrix(self,width,height,coords=None):
        if coords is None:
            voxel = np.ones((width,height))
            coords = np.array(np.nonzero(voxel)).T

        dmatrix = cdist(coords,coords)
        return dmatrix
        
    def _compute_jacobian(self):
        param = self.param[0]
        exp_param = self.param[1]
        height = 1.0/(param*np.sqrt(2.0*np.pi))
        tol = 1.0e-2
        dmatrix = self.dmatrix 
        
        if self.new is False:
            # linear approximation to gaussian
            dmatrix = (-1.0/param)*dmatrix+1
            dmatrix = np.where(dmatrix<0,0.0,dmatrix)
        else:
            dmatrix = height * np.exp(-(dmatrix**exp_param)/(2.0*param**exp_param))  # gaussian-slope linear receptive field centered on zero
            dmatrix = np.where(dmatrix<tol,0.0,dmatrix)

        if self.normalize is True:
            max_per_row = np.sum(dmatrix,axis=0)
            J = np.matrix(dmatrix / max_per_row[:,np.newaxis]) # normalized per row
        else:
            J = np.matrix(dmatrix)

        return J

    def compute_metric_tensor(self,width,height):
        J = self._compute_jacobian()
        g = J.T * J
        return g,J

    def distance(self, pt1, pt2, dtype=np.int16):
        pt1 = np.matrix(pt1,dtype=dtype)
        pt2 = np.matrix(pt2,dtype=dtype)
        diff = (pt1 - pt2).T
        sqdist = (diff.T * self.g * diff)[0,0]
        if sqdist < 0.0:
            sqdist = 0.0
        dist = np.sqrt(sqdist)
        return dist

    def plot_receptive_field(self,param,exp_param):
        x = np.unique(self.dmatrix)
        y = np.exp(-(x**exp_param)/(2*param**exp_param))
        plot(x,y)
        show()


class MultivariateNormalMetric(Metric):
    def distance(self, pt1, pt2, dtype=np.float):
        pt1 = np.matrix(pt1,dtype=dtype)
        pt2 = np.matrix(pt2,dtype=dtype)
        diff = (pt1 - pt2).T
        sqdist = (diff.T * self.g * diff)[0,0]
        if sqdist < 0.0:
            sqdist = 0.0
        
        dist = np.sqrt(sqdist)
        return dist


class EuclideanMetric(Metric):
    def compute_metric_tensor(self,width,height):
        J = np.matrix(np.eye(width*height))
        g = J
        return g,J


class RandomMetric(Metric):
    def compute_metric_tensor(self,width,height):
        J = np.matrix(np.random.random((width,height)))
        max_per_row = np.sum(J,axis=0)
        J = np.matrix(J / max_per_row)  # normalized per row
        g = J.T * J
        return g,J


class MetricFromJacobian(Metric):
    def __init__(self,J):
        self.param = 0
        self.levels = None
        self.J = J
        self.g = J.T * J
        self.vmin = 0
        self.vmax = 255

        



