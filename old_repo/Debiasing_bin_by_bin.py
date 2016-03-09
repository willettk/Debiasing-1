# Import packages ##############################################################
################################################################################
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import math

# Import the required files (the complete data set FITS and voronoi bin data).##
################################################################################

gal_data=fits.getdata("fits/d20.fits",1) # Raw data from the galaxy zoo files.

bins=np.load("npy/vor_arm_z.npy") # Bin paramaters for each of the galaxies in the
# sample.

cols=["t11_arms_number_a31_1_weighted_fraction",
      "t11_arms_number_a32_2_weighted_fraction",
      "t11_arms_number_a33_3_weighted_fraction",
      "t11_arms_number_a34_4_weighted_fraction",
      "t11_arms_number_a36_more_than_4_weighted_fraction",
      "t11_arms_number_a37_cant_tell_weighted_fraction",
      "PETROMAG_MR","R50_KPC","REDSHIFT_1"]

gal_tb=np.array([gal_data.field(c) for c in cols]) # Raw galaxy data and 
# parameters.

x_guides=np.log10([0.2,0.5,0.8])
y_guides=np.array([0,1])

bins=bins.T

v_min=int(np.min(bins[:,0])) # Gives the highest and lowest numbered voronoi
#bins for future use.
v_max=int(np.max(bins[:,0]))

# Flag array for votes not reaching the minimum vote fraction ##################
################################################################################

# For function plotting, we do not want to keep the very lowest 'noisy' votes
# because plotting in log space will lead to votes that -> -inf. Hence flags 
# are created to remove such votes. The cut is made such that 1 'full' vote (ie.
# 1 vote that is weighted as 1) is required for the galaxy to be included.

flag=np.zeros((6,len(gal_data.T)))

min_vf=bins[:,13]

for a in range(0,6):

    flag[a]=gal_tb[a] >= min_vf

# Add an indexing column to keep all galaxies in the correct order:

i=np.array([np.arange(0,len(bins))])
data=np.concatenate([(bins[:,0:7].T),gal_tb,flag,i])

# Find an array row for a given value. #########################################
################################################################################

# Can return a column from either a given log(vf) (if col=0) or CF (if col=1).
# this is used to match data bwtween the arrays to find vote fractions from 
# corresponding CFs.

def find_value(array,value,col):
    
    if col == 0:
        
        col2=1
            
    else:
        
        col2=0
    
    ind=np.argmin(np.abs(array[col]-value))
    
    v1=(array.T[ind]).T
    
    if v1[col] == value:
        
        v2=v1
        v_out=v1
        
    else:
    
        if v1[col] > value:
            
            v2=v1
        
            v1=(array.T[ind-1]).T
            
            grad=(v2[col2]-v1[col2])/(v2[col]-v1[col])
            
            value_2=v1[col2]+(value-v1[col])*grad
            
            if v2[col]-v1[col] != 0:
            
                value_2=v1[col2]+(value-v1[col])*grad
                
            else: value_2=v1[1]
        
        elif (v1[col] < value) & (ind < len(array.T) - 1):
        
            v2=(array.T[ind+1]).T
            
            grad=(v2[col2]-v1[col2])/(v2[col]-v1[col])
            
            if v2[col]-v1[col] != 0:
            
                value_2=v1[col2]+(value-v1[col])*grad
                
            else: value_2=v1[1]
        
        else:
            
            value_2=v1[1]
    
        v_out=np.array([value,value_2])
    
    return v_out

# Function for plotting the raw data in log space. #############################
################################################################################

def plot_raw(D,plot,style):

# If plot=1, data will be plotted. Otherwise there is no plotted output.

    D_ord=np.argsort(D[a+7])
    
    D_r=np.array([D[a+7],D[a+16],D[22]])
    
    D_r=(D_r.T[D_ord]).T
    
    D_p=D_r
    
    D_p=np.concatenate([D_r,np.array([np.linspace(0,1,len(D_r.T))])])
    
    D_p=(D_p.T[D_p[1] == 1]).T
    
    D_p[0]=np.log10(D_p[0])
    
    if plot ==1:
    
        plt.plot(D_p[0],D_p[3],"-",color=style)
        
        plt.xlabel("$\log(v_f)$")
        plt.ylabel("Cumulative fraction")
        
        plt.ylim([0,1])
        
    return np.array([D_p[0],D_p[3],D_p[2]]) # Returns log(vf), CF and index.

# Function for fitting and plotting a function to the data (the input data is 
# the returned array from 'plot raw'.###########################################
################################################################################

def plot_function(D,plot,style,y_05,f_max):

# If plot=1, data will be plotted. Otherwise there is no plotted output.
    
    def f(x,k,c,L): 
        
        L=y_05*(1+np.exp((-k*(math.log10(f_max))+c)))
        
        if L >=5: 
            L=5 # Set a limit on L as it has a tendency to get very large for 2
# armed spirals.
        return L/(1+np.exp(-k*x+c))
    
    popt,pcov=curve_fit(f,D[0],D[1],maxfev=100000,p0=[1,0,0])
    
    popt[2]=y_05*(1+np.exp((-popt[0]*(math.log10(f_max))+popt[1])))
    
    x=np.linspace(-4,0,1000)
    
    if plot == 1:
        
        plt.plot(x,f(x,popt[0],popt[1],popt[2]),"--",color=style)
        
        plt.ylim([0,1])
    
    return(popt) # Returns the fitted curve parameters.

# Function for getting a log(vf) from a given CF. ##############################
################################################################################
 
def inverse_f(y,k,c,L):
    return -(1/k)*(np.log((L/y)-1)-c)

# Now set up the debiasing. ####################################################
################################################################################

plt.close("all")

plot=0 # if plot=1, each of the plots will be output, so ensure that the v 
# limits aren't vmax and vmin if doing this.

clr=[0,0,0] # Plotting colours from blue -> red with z.
clf=[1,0,0]

f_max=0.5

# Set up the initial array for the output data to be written to:

debiased_fractions=np.zeros((7,len(data.T)))

debiased_fractions[0]=np.arange(0,len(data.T)) # Index column. 

debiased_fractions[1:]=data[7:13]

# Debias the data. #############################################################
################################################################################

# There is an fmax parameter in this code. this is because the functions at the 
# high end of the vf data don't always have the best fit. However, as the noise 
# is low at this end, data is mapped directly on to the low redshift bins for
#  vf  > fmax. Otherwise a function is used to smooth out the noise and give a
# reference point if there is no data at the very low end. 

for v in range(v_min,v_max+1): # Do this for each voronoi bin. 

#for v in range(1,2):
    
    data_v=(data.T[data[0] == v]).T
            
    data_plot=(data.T[(data[0] == v) & (data[1] == 1)]).T

    for a in range(0,6):
        
        z_min=int(np.min(data_v[a+1])) # Each arm no. has a different range of z
# bins. 
        z_max=int(np.max(data_v[a+1]))
        
        if plot == 1:
    
            plt.subplot(2,3,a+1)

        n=plot_raw(data_plot,plot,clr)
        
        y_05=find_value(n,math.log10(f_max),0)
        
        y_05=y_05[1]
    
        p=plot_function(n,plot,clf,y_05,f_max)
        
        locals()["pars{}".format(a)]=p
        
        for z in range(z_min+1,z_max+1):
            
            data_z=(data_v.T[data_v[a+1] == z]).T
        
            nz=plot_raw(data_z,plot,clr)
        
            for r in range(0,len(nz.T)):
            
                CF=nz[1,r]
                logvf=nz[0,r]
                idx=nz[2,r]
        
                out=find_value(n,CF,1)
            
                out=np.array([np.min(out),np.max(out)]) # No idea why this is
# required but apparently it is :-S

# This is where the data is fitted differently depending on the fmax parameter:
                
                if out[0] <= math.log10(f_max):
                
                    logvf=inverse_f(CF,p[0],p[1],p[2])
                
                    out=np.array([logvf,CF])
                
                debiased_fractions[a+1,idx]=10**(out[0])
                
                if plot == 1:

                    plt.plot(nz[0],nz[1],"bo")
                    plt.plot(out[0],out[1],"ro")
            
                    plt.xlim([-2.5,0])
                    plt.ylim([0,1])

if plot == 1:
    
    plt.show()

# Create a normalised array with the sums of the vfs. ##########################
################################################################################

sums=np.sum(debiased_fractions[1:],axis=0)

debiased_fractions_norm=debiased_fractions

debiased_fractions_norm[1:]=(debiased_fractions[1:])/sums

sums_r=np.sum(debiased_fractions_norm[1:],axis=0)

# Plot a sum of vote fractions histogram:

plt.figure(1)

plt.hist(sums,bins=30)

plt.xlabel(r"$\Sigma f_v$")
plt.ylabel(r"$N_{gal}$")

# Plot raw vs debiased for each of the arm numbers. ############################
################################################################################

plt.figure(2)

raw=data[7:13]

z=gal_tb[-1]

z=np.log10(z)
    
scaled_z = (z - z.min()) / z.ptp()
colors = plt.cm.coolwarm(scaled_z)

pld=debiased_fractions

# Points are coloured blue -> red depending on z. 

for a in range(0,6):

    plt.subplot(2,3,a+1)
    
    plt.scatter(raw[a],pld[a+1],marker="+",s=30,c=colors)
    
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.plot([0,1],[0,1],"k-")
    
    plt.xlabel(r"$f_v$")
    plt.ylabel(r"$f_{debiased}$")
    
plt.show()

################################################################################
################################################################################

