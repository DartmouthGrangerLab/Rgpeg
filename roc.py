#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
ROC Figure Code 
=========================================================

Run with:
roc.main()

Prepare CalTech256 images with ImageMagick using:
# mogrify -scale 512x384\! *.jpg
# mogrify -colorspace Gray *.jpg

"""
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import itertools
import PIL.Image as img

from library import get_inner_region, compute_adjacency
from metric import MultivariateNormalMetric
from figurecode import plot_basis_ndim, plot_summary_results, plot_final_results, plot_summary_results_movie
from jpeg import get_Q, jpeg_encode_decode
from rgpeg import rgpeg_encode_decode, get_hamiltonian_for_dct, get_jacobian_feature_to_physical


############################
# Imagery
############################

class Image():
    def __init__(self,dname='images/',fname='image_0422.jpg',field_height=8,field_width=8):
        self.dname = dname
        self.fname = fname
        self.fullname = dname+fname
        self.rawdata = self._load_image()
        self.image_height,self.image_width = self.rawdata.shape
        self.field_height = field_height
        self.field_width = field_width
        self.num_fields_per_height = self.image_height / self.field_height
        self.num_fields_per_width = self.image_width / self.field_width
        self.num_fields = self.num_fields_per_height * self.num_fields_per_width
        self.partitioned_image = self._partition_image()

    def _load_image(self):
        im = img.open(self.fullname).convert('L')
        a = np.array(list(im.getdata()))
        self.width, self.height = im.size
        a = a.reshape((self.height,self.width))
        return a

    def _partition_image(self):
        al = [x for x in [np.split(x,self.num_fields_per_width,axis=1) for x in np.split(self.rawdata,self.num_fields_per_height,axis=0)]]
        items = list(itertools.chain(*al))
        image = np.matrix([x.reshape((1,self.field_height*self.field_width)).squeeze() for x in items]).T-127.5
        return image


def load_image(fname):
    im = Image.open(fname).convert('L')
    a = np.array(list(im.getdata()))
    width, height = im.size
    a = a.reshape((height,width))
    return a


############################
# evaluation/scoring functions
############################

#compute entropy of the encoding
def entropy(data):
    n = data.size
    data = data - np.min(data) # shift to eliminate negative values
    proportions = np.bincount(np.array(data,dtype=np.int).ravel())/(float(n))
    proportions = proportions[np.nonzero(proportions)]
    H = -np.sum(np.multiply(proportions,np.log2(proportions)))
    return H

#compute perceived error
def rrms(data1,data2,m):
    data1 = np.array(data1).ravel()
    data2 = np.array(data2).ravel()
    result = m.distance(data1,data2,dtype=np.float)
    result = result/np.sqrt(8.0*8.0)
    return result


############################
# Main
############################

class TestCompression():
    def __init__(self,screen_pixel_size=0.282,viewing_distance=24.0,dx=1000.0, dy=100.0, dpi=600,process_all=False,save_figs=True,error_match_results=True,full_basis=False,results_fmt='pdf',results_dir='results/',save_movie=False,movie_direction=-1):
        self.img_width = 8
        self.img_height = 8
    
        self.screen_pixel_size = screen_pixel_size
        self.viewing_distance = viewing_distance
        self.dx = dx
        self.dy = dy
        self.di = 1.0
        self.dpi = dpi
        self.process_all = process_all
        self.save_figs = save_figs
        self.error_match_results = error_match_results
        self.full_basis = full_basis
        self.results_dir = results_dir
        self.results_fmt = results_fmt
        self.save_movie = save_movie
        self.movie_direction = movie_direction

    def process(self, dnames=['images/'], stepsize=10):
        jpg_roc_err = list()
        jpg_roc_ent = list()
        rjpg_roc_err = list()
        rjpg_roc_ent = list()

        if not isinstance(dnames, (list,tuple)):
            dnames = [dnames]

        for dname in dnames:
            if os.path.isfile(dname) and os.path.splitext(dname)[1] == '.jpg':
                result_dir_path = os.path.join(self.results_dir, os.path.basename(dname))
                img = Image(os.path.dirname(dname)+'/',os.path.basename(dname))
                self.img_height = img.image_height
                self.img_width = img.image_width
                err_rgpeg, ent_rgpeg, err_jpg, ent_jpg = self.processOneImg(data=img.partitioned_image,stepsize=stepsize,save_figs=self.save_figs,save_fname=result_dir_path)
                jpg_roc_err.append(err_jpg)
                jpg_roc_ent.append(ent_jpg)
                rjpg_roc_err.append(err_rgpeg)
                rjpg_roc_ent.append(ent_rgpeg)
            else:
                for root, dirs, files in os.walk(dname, topdown=True):
                    for name in files:
                        image_path = os.path.join(root, name)
                        if os.path.splitext(image_path)[1] == '.jpg':
                            result_path = os.path.relpath(image_path,start=dname)
                            result_dir_path = os.path.join(self.results_dir, result_path)
                            result_dirname = os.path.dirname(result_dir_path)
                            if not os.path.exists(result_dirname):
                                os.makedirs(result_dirname)
                            img = Image(os.path.dirname(image_path)+'/',os.path.basename(image_path))
                            self.img_height = img.image_height
                            self.img_width = img.image_width
                            err_rgpeg, ent_rgpeg, err_jpg, ent_jpg = self.processOneImg(data=img.partitioned_image,stepsize=stepsize,save_figs=self.save_figs,save_fname=result_dir_path)
                            jpg_roc_err.append(err_jpg)
                            jpg_roc_ent.append(ent_jpg)
                            rjpg_roc_err.append(err_rgpeg)
                            rjpg_roc_ent.append(ent_rgpeg)
                            if not self.process_all:
                                return
        plot_final_results(jpg_roc_err, jpg_roc_ent, rjpg_roc_err, rjpg_roc_ent, self.results_dir, self.dpi)
        
        
    def processOneImg(self, stepsize=10, save_figs=True, save_fname='default.mp4', data=None):
        if data is None:
            imgOrig = np.ones((8,8)) * 2.0
            imgOrig[0,0] = 0.0
            imgOrig[2:5,2:5] = 1.0
            imgOrig = imgOrig - 1.0
            imgOrig = np.matrix(imgOrig.reshape((8*8,1)))
            imgOrig = 127.5 * imgOrig
        else:
            imgOrig = data

        npts = imgOrig.shape[1]
        ndim = imgOrig.shape[0]

        levels = (np.arange(0, 100, stepsize)+stepsize)[::-1]
        num_levels = levels.shape[0]
        
        # ----------------
        # --JPEG, RGPEG for each quality/fidelity level--
        # ----------------
        encodedResultsJPEG = np.zeros((64, npts, num_levels))
        decompImgJPEG = np.zeros((64, npts, num_levels))
        encodedResultsRGPEG = np.zeros((ndim, npts, num_levels))
        decompImgRGPEG = np.zeros((ndim, npts, num_levels))
        for j in range(num_levels): #for each quality/fidelity level
            lvl = levels[j]
            
            imgEncodedJPEG,imgDecodedJPEG = jpeg_encode_decode(imgOrig, lvl)
            encodedResultsJPEG[:,:,j] = imgEncodedJPEG
            decompImgJPEG[:,:,j] = imgDecodedJPEG
            
            imgEncodedRGPEG,imgDecodedRGPEG,tmp = rgpeg_encode_decode(imgOrig, lvl, self.screen_pixel_size, self.viewing_distance, self.dx, self.dy, self.full_basis)
            encodedResultsRGPEG[:,:,j] = imgEncodedRGPEG
            decompImgRGPEG[:,:,j] = imgDecodedRGPEG
            
        # ----------------
        # --now evaluate--
        # ----------------
        m = MultivariateNormalMetric(8,8,(0.01,2.0),normalize=True,new=True)
        if self.full_basis:
            basis = np.matrix(np.eye(100))
            J = get_jacobian_feature_to_physical(screen_pixel_size=self.screen_pixel_size, viewing_distance=self.viewing_distance, ncells=10)
            Adj = compute_adjacency(self.dx, self.dy, J, basis)
            m.g = get_inner_region(Adj)
        else:
            basis = np.matrix(np.eye(64))
            J = get_jacobian_feature_to_physical(screen_pixel_size=self.screen_pixel_size, viewing_distance=self.viewing_distance)
            Adj = compute_adjacency(self.dx, self.dy, J, basis, normalize=True)
            m.g = Adj*10.0
        
        err_jpg = np.zeros(num_levels)
        ent_jpg = np.zeros(num_levels)
        ent_rgpeg = np.zeros(num_levels)
        err_rgpeg = np.zeros(num_levels)
        for j in range(num_levels): #for each quality/fidelity level
            lvl = levels[j]
            
            #measure perceptual error
            for pt in range(npts):
                img_arr = np.array(imgOrig[:,pt]).ravel()
                decompressed_img_arr_jpg = np.array(decompImgJPEG[:,pt,j]).ravel()
                decompressed_img_arr_rgpeg = np.array(decompImgRGPEG[:,pt,j]).ravel()
                
                err_jpg[j] = err_jpg[j] + np.square(rrms(img_arr, decompressed_img_arr_jpg,m))
                err_rgpeg[j] = err_rgpeg[j] + np.square(rrms(img_arr, decompressed_img_arr_rgpeg, m))
            err_jpg[j] = np.round(np.sqrt(err_jpg[j]) / npts,2)
            err_rgpeg[j] = np.round(np.sqrt(err_rgpeg[j]) / npts,2)
            
            #measure entropy (proxy for file size)
            ent_jpg[j] = np.round(entropy(np.array(encodedResultsJPEG[:,:,j]).ravel()),2)
            ent_rgpeg[j] = np.round(entropy(np.array(encodedResultsRGPEG[:,:,j]).ravel()),2) 
            
            tmp1,tmp2,U = rgpeg_encode_decode(imgOrig, lvl, self.screen_pixel_size, self.viewing_distance, self.dx, self.dy, self.full_basis) #we'll display this below
            print '============================================'
            print 'Quality:',lvl,'% ','t=',(100.0 - np.round(lvl,2))
            print 'Entropy: ', ent_rgpeg[j], '(jpg=', ent_jpg[j], ')',
            print 'Error: ', err_rgpeg[j], '(jpg=', err_jpg[j], ')'
            print 'rJPGQ:', np.round(1.0/np.diag(U),2)
            print 'JPGQ:', 1.0/np.diag(get_Q(lvl))
            
            if npts == 1:
                print 'DCT-----------------------------------------'
                for xi in range(8):
                    for xj in range(8):
                        arg1 = int(np.rint(encodedResultsRGPEG[:,:,j].T).reshape((8,8))[xi,xj])
                        arg2 = int(encodedResultsJPEG[:,:,j].T.reshape((8,8))[xi,xj])
                        sep = ':'
                        if arg1 <> arg2:
                            sep = ':'
                        print '{0:4d}{2}{1:4d}'.format(arg1,arg2,sep),
                    print
                print 'Image_output---------------------------------'
                for xi in range(8):
                    for xj in range(8):
                        arg1 = int(decompImgRGPEG[:,:,j].reshape((8,8))[xi,xj]+127.5)
                        arg2 = int(decompImgJPEG[:,:,j].reshape((8,8))[xi,xj]+127.5)
                        sep = ':'
                        if arg1 <> arg2:
                            sep = ':'
                        print '{0:4d}{2}{1:4d}'.format(arg1,arg2,sep),
                    print 
                print 'Error_diff-----------------------------------'
                for xi in range(8):
                    for xj in range(8):
                        ref = int(imgOrig.reshape((8,8))[xi,xj])
                        arg1 = int(np.sqrt((ref-int(decompImgRGPEG[:,:,j].reshape((8,8))[xi,xj]))**2))
                        arg2 = int(np.sqrt((ref-int(decompImgJPEG[:,:,j].reshape((8,8))[xi,xj]))**2))
                        sep = ':'
                        if arg1 <> arg2:
                            sep = ':'
                        print '{0:4d}{2}{1:4d}'.format(arg1,arg2,sep),
                    print 

        if save_figs:
            H = get_hamiltonian_for_dct(screen_pixel_size=self.screen_pixel_size, viewing_distance=self.viewing_distance, dx=self.dx, dy=self.dy, full_basis=self.full_basis)
            D,V_large = np.linalg.eigh(H)
            V = V_large
            if self.full_basis:
                V = get_inner_region(V.T).T
        
            fig = plt.figure(dpi=self.dpi,figsize=(11,6.5))
            plot_summary_results(self.img_width, self.img_height, decompImgRGPEG,decompImgJPEG,err_rgpeg,ent_rgpeg,err_jpg,ent_jpg,self.error_match_results,fig=fig,timeslice=levels,Phi_slices=encodedResultsRGPEG,jpg_dct_slices=encodedResultsJPEG,levels=levels)
            fig.savefig(save_fname[:-4]+'-summary.'+self.results_fmt, format=self.results_fmt,dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            fig = plt.figure(dpi=self.dpi,figsize=(11,6.5))
            plot_basis_ndim(V_large.T,ax=plt.gca())
            fig.savefig(save_fname[:-4]+'-fullbasis.'+self.results_fmt, format=self.results_fmt,dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
            fig = plt.figure(dpi=self.dpi,figsize=(11,6.5))
            plot_basis_ndim(V.T,ax=plt.gca())
            fig.savefig(save_fname[:-4]+'-basis.'+self.results_fmt, format=self.results_fmt,dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)

        if self.save_movie:
            fig = plt.figure(dpi=self.dpi,figsize=(11,6.5))
            metadata = dict(title='RGPEG '+save_fname[:-4]+'-movie.mp4', artist='RGPEG',
                            comment='RGPEG')
            writer = animation.ImageMagickFileWriter(fps=1, metadata=metadata)
            mwriter = animation.FFMpegWriter(fps=1, metadata=metadata)
            with mwriter.saving(fig, save_fname[:-4]+'-movie.' + 'mp4', 300):
                for i in range(num_levels)[::self.movie_direction]:
                    j = i
                    if i > 0:
                        jidx = np.argwhere(ent_rgpeg <= ent_jpg[i])
                        if jidx.size > 0:
                            j = jidx[0,0]
                    lvl = levels[i]
                    print 'Saving movie frame for level:', lvl
                    plot_summary_results_movie(imgOrig, self.img_width, self.img_height, decompImgRGPEG, decompImgJPEG, err_rgpeg, ent_rgpeg, err_jpg, ent_jpg, fig=fig, timeslice=levels, Phi_slices=encodedResultsRGPEG, jpg_dct_slices=encodedResultsJPEG, levels=levels, jpg_i=i, rgpeg_i=j)
                    writer.setup(fig, save_fname[:-4]+'-'+'{:03}'.format(lvl)+'.png', self.dpi)
                    writer.grab_frame()
                    #writer.cleanup()
                    writer.finish()
                    mwriter.grab_frame()
    
        return err_rgpeg, ent_rgpeg, err_jpg, ent_jpg


#run from here to generate figures
def main(dnames=['images/Faces10small3/'], screen_pixel_size=0.282, viewing_distance=24.0, dx=50000.0, dy=25000.0, dpi=600, process_all=True, save_figs=True, stepsize=1, error_match_results=True, full_basis=False, results_fmt='pdf', results_dir='results/', save_movie=False, movie_direction=-1):
    print 'Procesing: ',dnames
    print 'ScreenPixelSize: ', screen_pixel_size
    print 'ViewingDistance: ', viewing_distance
    print 'dx: ',dx
    print 'dy: ',dy
    t = TestCompression(screen_pixel_size, viewing_distance, dx, dy, dpi, process_all, save_figs, error_match_results, full_basis, results_fmt, results_dir, save_movie=save_movie, movie_direction=movie_direction)
    t.process(stepsize=stepsize, dnames=dnames)

if __name__ == "__main__":
    main()

