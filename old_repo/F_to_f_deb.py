# coding: utf-8

# 3/3 scripts to run

# Import packages ##############################################################
################################################################################

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

n_morph = 6     # Number of poss. morphologies (6 for the various spiral arms)

def load_data():

    # Load the required data
    ############################################################################

    gal_data=fits.getdata("fits/d20.fits",1) # Galaxy zoo data.
    
    params=np.load("npy/kc_fit_params.npy") # Parameters from the function fitting.
    
    cmin=np.load("npy/cmin.npy")
    cmax=np.load("npy/cmax.npy")
    kmin=np.load("npy/kmin.npy")
    kmax=np.load("npy/kmax.npy")

    # Have included some limits to the data- for each arm number the minimum and 
    # maximum c and k values are set as the limits to the functions. 
    
    cols=["t11_arms_number_a31_1_weighted_fraction",
          "t11_arms_number_a32_2_weighted_fraction",
          "t11_arms_number_a33_3_weighted_fraction",
          "t11_arms_number_a34_4_weighted_fraction",
          "t11_arms_number_a36_more_than_4_weighted_fraction",
          "t11_arms_number_a37_cant_tell_weighted_fraction",
          "PETROMAG_MR","R50_KPC","REDSHIFT_1"]
    
    gal_tb=np.array([gal_data.field(c) for c in cols])
    
    i=np.array([np.arange(0,len(gal_tb.T))])
    
    gal_tb=np.concatenate([gal_tb,i]) # i is an index column.
    
    # Find the minimum redshift bin values from means of lowest bins. ##############
    ################################################################################
    
    r_data=np.load("npy/fixed_bin_size_params_2.npy")
    
    min_z=np.zeros((6,1))
    
    for a in range(0,6):
        
        r_data_0=r_data[(r_data[:,1] == a) & (r_data[:,2] == 1)]
        
        min_z[a]=np.mean(r_data_0[:,5])
    
    return gal_tb,params,cmin,cmax,kmin,kmax,min_z

# Define functions for getting log(vf) (x) from CF (y) and v.v. ################
################################################################################

def f(x,k,c): 
        
    # Function for getting log(vf) (x) from CF (y)
    ############################################################################

    L=1+np.exp(c)
        
    return L/(1+np.exp(-k*x+c))

def i_f(y,k,c):
    
    # Function for getting CF (y) from log(vf) (x)
    ############################################################################

    L=1+np.exp(c)
    
    return -(1/k)*(np.log((L/y)-1)-c)

def debias(min_z):

    # Debiasing procedure. #########################################################
    ################################################################################

    debiased=np.zeros((n_morph,len(gal_tb.T)))
    
    # Each galaxy gets a function fit to its M,R and z parameters, which are scaled 
    # to the equivalent M and r functions at low z.
    
    for a in range(0,n_morph):
        
        z_base=min_z[a]
        p=params[a]
        
        k=params[a,1]+params[a,2]*gal_tb[6]+params[a,3]*gal_tb[7]+params[a,4]*gal_tb[8]
        c=params[a,5]+params[a,6]*gal_tb[6]+params[a,7]*gal_tb[7]+params[a,8]*gal_tb[8]
        
        kb=params[a,1]+params[a,2]*gal_tb[6]+params[a,3]*gal_tb[7]+params[a,4]*z_base
        cb=params[a,5]+params[a,6]*gal_tb[6]+params[a,7]*gal_tb[7]+params[a,8]*z_base
        
        kc=np.array([k,c,kb,cb])
        
        # Section for dealing with any functions outside the k and c limits. #
            
        kh=kc[0] > kmax[a,0]
        kl=kc[0] < kmin[a,0]
        ch=kc[1] > cmax[a,1]
        cl=kc[1] < cmin[a,1]
        
        kbh=kc[2] > kmax[a,0]
        kbl=kc[2] < kmin[a,0]
        cbh=kc[3] > cmax[a,1]
        cbl=kc[3] < cmin[a,1]
        
        kc=kc.T
    
        kc[:,0][kh]=kmax[a,0]
        kc[:,0][kl]=kmin[a,0]
        kc[:,1][ch]=cmax[a,1]
        kc[:,1][cl]=cmin[a,1]
        
        kc[:,2][kbh]=kmax[a,0]
        kc[:,2][kbl]=kmin[a,0]
        kc[:,3][cbh]=cmax[a,1]
        kc[:,3][cbl]=cmin[a,1]

        kc=kc.T
        
        ######################################################################
        
        a_d=np.array([gal_tb[-1],gal_tb[a],kc[0],kc[1],kc[2],kc[3]])
        
        a_d=(a_d.T[a_d[1] > 0]).T
        
        a_d[1]=np.log10(a_d[1])
        
        CF=f(a_d[1],a_d[2],a_d[3])
        
        lvd=i_f(CF,a_d[4],a_d[5])
        
        vdeb=10**(lvd)
        
        i=a_d[0].astype(int)
        
        debiased[a][i]=vdeb
    
    return debiased

def plot_debiased(gal_tb,debiased):

    # Plot debiased values vs. raw values for comparison. Blue -> red with z. ##
    ############################################################################
    
    fig,axarr = plt.subplots(2,3,sharex='col',sharey='row')
    
    raw=gal_tb[0:6]
    
    z=gal_tb[-2]
    z=np.log10(z)
        
    scaled_z = (z - z.min()) / z.ptp()
    colors = plt.cm.coolwarm(scaled_z)
    
    tasklabels = {0:'1', 1:'2', 2:'3', 3:'4', 4:'5+', 5:'??'}

    for a,ax in enumerate(axarr.ravel()):
        
        ax.scatter(raw[a],debiased[a],marker="+",s=30,c=colors)
        
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.plot([0,1],[0,1],"k-")
        
        ax.text(-3.9,0.9,r'$N_{arms}=$%s' % tasklabels[a],fontsize=10,ha='left')

        if a > 2:
            ax.set_xlabel(r"$f_v$")
        
        if a == 0 or a == 3:
            ax.set_ylabel(r"$f_{debiased}$")
        
        if a == 1:
            ax.set_title('Debiased vs raw vote fractions')
    
    fig.savefig('plots/debiased.pdf', dpi=200)
        
    return None

def plot_vf_histogram(debiased):

    # Plot histogram of sums of f_v. ###########################################
    ############################################################################
    
    fig = plt.figure(2)
    ax = fig.add_subplot(111)

    sums=np.sum(debiased,axis=0)

    ax.hist(sums,bins=30)
    
    ax.set_xlabel(r"$\Sigma f_v$")
    ax.set_ylabel(r"$N_{gal}$")
    
    fig.savefig('plots/vf_histogram.pdf', dpi=200)
    
    return None

################################################################################
################################################################################

if __name__ == "__main__":

    gal_tb,params,cmin,cmax,kmin,kmax,min_z = load_data()
    debiased = debias(min_z)
    plot_debiased(gal_tb,debiased)
    plot_vf_histogram(debiased)


