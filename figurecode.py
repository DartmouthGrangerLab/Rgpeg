'''
Created on Sep 28, 2017

Figure code
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot


# plot final results
def plot_final_results(jpg_roc_err, jpg_roc_ent, rjpg_roc_err, rjpg_roc_ent, results_dir, dpi):
    z_95 = 1.96
    fig = plt.figure()
    n = jpg_roc_err[0].shape[0]
    for i in range(len(jpg_roc_err)):
        jpg_ent_mean = np.mean(np.array(jpg_roc_ent),axis=0)
        jpg_err_mean = np.mean(np.array(jpg_roc_err),axis=0)
        rjpg_ent_mean = np.mean(np.array(rjpg_roc_ent),axis=0)
        rjpg_err_mean = np.mean(np.array(rjpg_roc_err),axis=0)
        jpg_ent_std = np.std(np.array(jpg_roc_ent),axis=0)
        jpg_err_std = np.std(np.array(jpg_roc_err),axis=0)
        rjpg_ent_std = np.std(np.array(rjpg_roc_ent),axis=0)
        rjpg_err_std = np.std(np.array(rjpg_roc_err),axis=0)
        jpg_ent_z = z_95 * (jpg_ent_std / np.sqrt(n))
        jpg_err_z = z_95 * (jpg_err_std / np.sqrt(n))
        rjpg_ent_z = z_95 * (rjpg_ent_std / np.sqrt(n))
        rjpg_err_z = z_95 * (rjpg_err_std / np.sqrt(n))
    plt.errorbar(rjpg_err_mean,rjpg_ent_mean,xerr=rjpg_err_z,yerr=rjpg_ent_z,marker='o',linestyle='-',color='green',markerfacecolor='green',ecolor='black')
    plt.errorbar(jpg_err_mean,jpg_ent_mean,xerr=jpg_err_z,yerr=jpg_ent_z,marker='o',linestyle=':',color='blue',markerfacecolor='blue',ecolor='black')
    plt.legend(['RJPG','JPG'])
    plt.xlabel('RRMS Error')
    plt.ylabel('Entropy (bits/pixel)')
    plt.title('ROC Curve')
    fig.savefig(results_dir+'totalroc.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    

def plot_summary_results(img_width, img_height, decomp_img_slices, decomp_img_jpg, err_rjpg, ent_rjpg, err_jpg, ent_jpg, error_match_results, fig=None,timeslice=0,Phi_slices=None,jpg_dct_slices=None,levels=0):
    ent_color = '#328db8'
    err_color = '#ad59ab'
    plt.rcParams.update({'axes.titlesize': 'small'})
    plt.rcParams.update({'axes.labelsize': 'small'})
    plt.rcParams.update({'xtick.labelsize': 'x-small'})
    plt.rcParams.update({'ytick.labelsize': 'x-small'})   
    
    fig.clear()

    quality_levels = [30,50,60,80]
    j_err_marker = '*'
    j_ent_marker = 'd'
    r_err_marker = 'H'
    r_ent_marker = '^'
    mfc = ['blue','green','yellow','orange']
    qtoi = dict()
    rjpg_qtoi = dict()
    for i in range(len(levels)):
        q = levels[i]
        qtoi[q] = i
        if error_match_results:
            rjpg_qtoi[q] = (np.abs(err_rjpg-err_jpg[i])).argmin()
        else:
            rjpg_qtoi[q] = (np.abs(ent_rjpg-ent_jpg[i])).argmin()
            
    gs = gridspec.GridSpec(4, 14,left=0.0, bottom=0.4, right=1.0, top=1.0, wspace=0.0, hspace=0.22, width_ratios=[8,8,8,8,8,8,8,8,8,8,8,8,1,1], height_ratios=None)
    for q in range(len(quality_levels)):
        i = qtoi[quality_levels[q]]
        rjpg_i = rjpg_qtoi[quality_levels[q]]
        for method in range(2):
            if method == 0:
                method_data = decomp_img_jpg
                dct_data = jpg_dct_slices
                err = 'er='+'{:.4}'.format(err_jpg[i])
                ent = 'en='+'{:.4}'.format(ent_jpg[i])
                err_marker = j_err_marker
                ent_marker = j_ent_marker
                idx = i
            else:
                method_data = decomp_img_slices
                dct_data = Phi_slices
                err = 'er='+'{:.4}'.format(err_rjpg[rjpg_i])
                ent = 'en='+'{:.4}'.format(ent_rjpg[rjpg_i])
                err_marker = r_err_marker
                ent_marker = r_ent_marker
                idx = rjpg_i
            ax3a = plt.subplot(gs[2*method:2*method+2, 0+q*3:0+q*3+2])
            ax3a.set_aspect('auto')

            plot_data(method_data[:,:,idx], ax=ax3a, img_width=img_width, img_height=img_height)
            blue_line = mlines.Line2D([], [], color='none', mfc=mfc[q], marker=err_marker,markersize=15, label=err)
            green_line = mlines.Line2D([], [], color='none', mfc=mfc[q], marker=ent_marker,markersize=15, label=ent)
            plt.legend(handles=[blue_line,green_line],numpoints=1,bbox_to_anchor=(0.1, 0.0, 0.8, -0.05), loc=2,ncol=4, mode="expand", borderaxespad=0.0, borderpad=0.0, fontsize='x-small',markerscale=0.5,frameon=False, handletextpad=0.0)
            ax3a.yaxis.set_tick_params(size=0)
            ax3a.xaxis.set_tick_params(size=0)
            plt.setp(ax3a.get_yticklabels(), visible=False)
            plt.setp(ax3a.get_xticklabels(), visible=False)          
            
            ax3a = plt.subplot(gs[2*method:2*method+2, 2+q*3])
            ax3a.set_aspect('auto')
            ax3a.set_title(str(quality_levels[q]))
            ax3a.yaxis.set_tick_params(size=0)
            ax3a.xaxis.set_tick_params(size=0)
            plt.setp(ax3a.get_yticklabels(), visible=False)
            plt.setp(ax3a.get_xticklabels(), visible=False)
            if quality_levels[q] == quality_levels[-1]:
                show_colorbar = True
                cb_ax = plt.subplot(gs[2*method:2*method+2, -1])
            else:
                show_colorbar = False
                cb_ax = None
            plot_data(dct_data[:,:,idx],ax=ax3a, cb_ax=cb_ax, cmap='Reds',dctdata=True,show_colorbar=show_colorbar,vmin=-127,vmax=128, img_width=img_width, img_height=img_height)

    fig.text(-0.015, 0.87, 'J', ha='center', va='center', fontdict= {'weight' : 'bold'})
    fig.text(-0.015, 0.55, 'R', ha='center', va='center', fontdict= {'weight' : 'bold'})
    
    gs = gridspec.GridSpec(1, 2,left=0.02, bottom=0.0, right=1.0, top=0.34, wspace=0.1, hspace=0.0, width_ratios=[2,3], height_ratios=None)
    ax5 = fig.add_subplot(gs[0,0])
    plt.plot(err_jpg,ent_jpg,'ks',label='J',markersize=2)

    plt.plot(err_rjpg,ent_rjpg,'ko',label='R',markersize=2)
    for q in range(len(quality_levels)):
        i = qtoi[quality_levels[q]]
        rjpg_i = rjpg_qtoi[quality_levels[q]]
        plt.plot(err_jpg[i],ent_jpg[i],mfc=mfc[q],marker='s',markersize=10.0)
        plt.plot(err_rjpg[rjpg_i],ent_rjpg[rjpg_i],mfc=mfc[q],marker='o',markersize=10.0)

    plt.legend(fontsize='x-small')
    plt.xlabel('error (rRMS)')
    plt.ylabel('entropy (bits/pixel)')
    axes = plt.gca()
    
    xmax = 2.0
    ymax = 3.0
    axes.set_xlim([0,xmax])
    axes.set_ylim([0,ymax])
    
    font = { 'size'   : 8}
    ax5.set_title('ROC',fontdict=font)

    host = host_subplot(gs[0,1], axes_class=AA.Axes)
    par1 = host.twinx()
    host.plot(timeslice,ent_jpg,linestyle=':',color=ent_color,linewidth=2,zorder=0)
    host.plot(timeslice,ent_rjpg,linestyle='-',color=ent_color,linewidth=2,zorder=0)
    for q in range(len(quality_levels)):
        i = qtoi[quality_levels[q]]
        rjpg_i = rjpg_qtoi[quality_levels[q]]
        host.plot(timeslice[i],ent_jpg[i],mfc=mfc[q],marker=j_ent_marker,markersize=10.0,zorder=1)
        host.plot(timeslice[rjpg_i],ent_rjpg[rjpg_i],mfc=mfc[q],marker=r_ent_marker,markersize=10.0,zorder=1)

    par1.plot(timeslice,err_jpg,linestyle=':',color=err_color,linewidth=2,zorder=0)
    par1.plot(timeslice,err_rjpg,linestyle='-',color=err_color,linewidth=2,zorder=0)
    for q in range(len(quality_levels)):
        i = qtoi[quality_levels[q]]
        rjpg_i = rjpg_qtoi[quality_levels[q]]
        par1.plot(timeslice[i],err_jpg[i],mfc=mfc[q],marker=j_err_marker,markersize=10.0,zorder=1)
        par1.plot(timeslice[rjpg_i],err_rjpg[rjpg_i],mfc=mfc[q],marker=r_err_marker,markersize=10.0,zorder=1)
    host.set_xlabel('quality')
    host.set_ylabel('entropy (bits/pixel)',color=ent_color)
    par1.set_ylabel('error (rRMS)',color=err_color)
    host.set_title('Entropy and Error vs Quality',fontdict=font)
        

def plot_data(data, title=None, vmin=0, vmax=255, ax=None, cmap=plt.cm.gray, dctdata=False, show_colorbar=False, cb_ax=None, img_width=None, img_height=None):
    nfields_width = img_width / 8
    nfields_height = img_height / 8

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

    if not dctdata:
        img = np.zeros((img_height,img_width))
        data = np.rint(data)+128
        pt = 0
        for i in range(nfields_height):
            for j in range(nfields_width):
                img[8*i:8*i+8,8*j:8*j+8] = data[:,pt].reshape((8,8))
                pt = pt + 1
    else:
        img = np.zeros(8*8)
        for i in range(8*8):
            img[i] = np.unique(data[i]).shape[0] - 1
        img = img.reshape((8,8))
        vmin=0
        vmax=160
                
    image = ax.imshow(img, cmap=cmap, interpolation='none',vmin=vmin,vmax=vmax)
    if show_colorbar:
        plt.colorbar(image, ax=ax,cax=cb_ax,use_gridspec=True)

    if title is not None:
        ax.set_title(title)
    return image


def plot_basis_ndim(basis, title='Basis plot', pp=None, vmin=0, vmax=255, ax=None):
    n = int(np.sqrt(basis.shape[0]))
    use_show = False
    basis = np.rint(basis*128)+128

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        use_show = True

    img = np.zeros(basis.shape)
    for i in range(n):
        for j in range(n):
            img[n*i:n*i+n,n*j:n*j+n] = basis[n*i+j].reshape((n,n))
    ax.imshow(img[::-1], cmap=plt.cm.gray, interpolation='none',extent=(0,n*n,0,n*n),vmin=vmin,vmax=vmax)
    
    majorLocator = MultipleLocator(n)
    formatter = ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(b=True, which='major', color='b', linestyle='-', linewidth=2)
    ax.axis([0, n*n, n*n, 0])
    ax.set_title(title)

    if pp is None and use_show is True:
        plt.show()
    elif pp is not None:
        pp.savefig(fig)
        