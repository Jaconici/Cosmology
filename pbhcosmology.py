'''imports'''
from math import *
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from scipy.integrate import odeint, solve_ivp, quad, simpson
import numpy as np
from numpy import dot,transpose
import random
from scipy.special import kn
from scipy.interpolate import interp1d, interp2d
from tqdm import tqdm
import cmath as cm
from matplotlib.ticker import LogLocator, AutoMinorLocator

from scipy.special import *


'''constants'''
alpha = 0.2 #gravitational collapse factor
g = 106.75 #model d.o.f
GeV_in_g  = 1.782661907e-24  # 1 GeV in g
GCF = 6.70883E-39
Mpl =1.22e19 # [M_pl] = GeV #Mpl in GeVV
Msolar = 1.116E57 #Solar mass in GeV
h =0.673
rhoc = 1.878E-29 * h**2 / GeV_in_g / (1.9733E14)**3 #critical density in GeV^4

'''parameters'''
M1 = 1E13 #mass of the decaying RNH in GeV
MPBH = 1 / GeV_in_g    #mass of the PBHs in GeV where number is the grams


def T0(M):
    '''calculates the temperature at which a PBH of mass M forms, for relativistic degrees
    of freedom g'''
    #M is in grams
    #returns in GeV
    return 0.5 * ( 5 / (g * pi**3))**0.25 * np.sqrt( 3*0.2*Mpl**3 / M)
def aT0(T):
    ''' scale factor in terms of the temperature. The scale factor is normalised to  = 1 today'''
    coef =  (30*rhoc/(pi**2*g))**(1/4)
    return float(coef/(T))
def Ta0(a):
    '''The Temperature in terms of the scale factor. The scale factor is normalised to  = 1 today'''
    coef =  (30*rhoc/(pi**2*g))**(1/4)
    return coef/(a)
aformation = 8.539941141605994e-29                                                  #scale factor when the PBHs of MPBH = 1g form
def aT(T):
    ''' scale factor in terms of the temperature. The scale factor is normalised to  = 1 when a PBH of mass 1g is formed, hence the hard coded value'''
    coef =  (30*rhoc/(pi**2*g))**(1/4)
    return float(coef/(T*aformation))
def Ta(a):
    '''The Temperature in terms of the scale factor. The scale factor is normalised to  = 1 when a PBH of mass 1g is formed, hence the hard coded value'''
    coef =  (30*rhoc/(pi**2*g))**(1/4)
    return coef/(a*aformation)


def rad(T=None):
    '''returns the comoving radiation density in the standard rad-dom era'''
    if T==None:
        T = T0(MPBH)
    return (pi**2/30*g*T**4)

def s(T):
    '''the entropy in terms of plasma temperature'''
    return 4*pi**2*g*T**3/90
def NBH(beta,MBH,a=None,alt=False):
    '''this function returns the comoving number density of the PBHs of mass MBH, beta' = beta'''
    ''' the parameter a is functionally useless, present however as a check that the returned result is really comoving'''
    if a == None:
        Tf = T0(MBH) #if a is provided calculate the number density at a rather than at formation
    else:
        Tf = Ta0(a)
    if a == None:
        a = aT(Tf)
    return beta * rad(T=T0(MBH)) / MBH

def epN(M,Mv,MN,species="SM"):
    G = GCF
    '''first, define the spin dependent parameters'''
    f0,f1,f2 = 0.267,0.06,0.007
    fq1,fq0  = 0.142,0.147
    B0,B12,B1 = 2.66,4.53,6.04
    
    '''mass matrixes,Mv gives neutrinos, all in GeV'''
    Ml = [0.511e-3,105.7e-3,1.78] #leptons
    Mq = [2.2e-3,4.7e-3,1.28,96e-3,173,4.18] #quarks
    
    '''calculate the mass of a PBH with temperature equal to boson masses'''
    MW,MZ,MH = 1/(8*pi*G*80.4),1/(8*pi*G*91.2),1/(8*pi*G*125.25)
    
    '''calculate the exponential sum over leptons'''
    expl = 0
    for l in Ml:
        Mj = 1/(8*pi*G*l)
        expl+= exp(-M/(B12*Mj))
        
    '''calculate the sum over the quarks'''
    expq = 0
    for q in Mq:
        Mj = 1/(8*pi*G*q)
        expq += exp(-M/(B12*Mj))
        
    '''calculate the sum over the neutrino masses'''
    expv = 0
    for v in Mv:
        if v!=0:
            Mj = 1/(8*pi*G*v)
            expv += exp(-M/(B1*Mj))
            
    '''calculate the sum over the RHNs'''
    expN = 0
    for N in MN:
        Mj = 1/(8*pi*G*N)
        expN += exp(-M/(4.53*Mj))
    
    '''calculate the evaporation function'''
    if species == "SM":
        eps = 2*f1 + 16*f1 + 4*fq1 * ( expl + 3*expq ) + 2*fq0*expv + 3*f1*( 2*exp(-M/(B1*MW)) + exp(-M/(B1*MZ))) + f0*exp(-M/(B0*MH))
    elif species == "RHN":
        
        eps = 6*fq0 * expN
    return eps
    


def dlnMPBH(alpha,H,MBH,ep):
    kappa = 416.3/(30720*pi) * Mpl**4
    return -kappa*log(10) / (H * MBH**3)

def dlnrRad(H,rPBH,alpha,MBH,ep,epSM,rR):
    return -epSM * dlnMPBH(alpha,H,MBH,ep) * rPBH * 10**alpha / (rR*ep)

def dlnrPBH(H,rPBH,alpha,MBH,ep):
    return dlnMPBH(alpha,H,MBH,ep)



def coupledivp(alpha,ys,Mini,Mv,MN):
    MBH,rR,rBH,T = np.exp(ys[0]),exp(ys[1]),exp(ys[2]),ys[3]
    epSM,epRHN = epN(MBH,Mv,MN),epN(MBH,Mv,MN,species = "RHN")
    ep = epSM+epRHN
    '''Hubble parameter'''
    H = ( 8*pi/(3*Mpl**2)*(rBH * 10**(-3*alpha) + rR * 10**(-4*alpha)))**(0.5)
    dlnMBH = float(dlnMPBH(alpha,H,MBH,ep))                                                      #16a
    dlnrR = float(dlnrRad(H,rBH,alpha,MBH,ep,epSM,rR))                                            #16b
    dlnrBH = dlnrPBH(H,rBH,alpha,MBH,ep)                                                #16c
    dT = -T * (log(10) + epSM * dlnMPBH(alpha,H,MBH,ep) * rBH * 10**alpha / (ep * 4 * rR)) #16d
    dS = -epSM * dlnMBH * rBH / (T*ep)
    return (dlnMBH,dlnrR,dlnrBH,dT,dS)

def coupled(T,alpha):                           
    dT = -T * (log(10))
    
    return dT[0]

def masscut(alpha,ys,Mini,Mv,MN):
    MBH = e**(ys[0]) 
    return MBH - Mpl

def clean(array,tarray,minimum):
    cleanarray,cleantarray = [],[]
    for i in range(1,np.size(array)):
        if array[i] != array[i-1] or tarray[i] < minimum or array[i] == 0:
            cleanarray.append(array[i])
            cleantarray.append(tarray[i])
    return(cleanarray,cleantarray)



def functions(MBH,betaf,MN,Mv,rtol = 1e-5,atol = 1e-5):
    '''function which returns the functions Ta, aT, Ha, sa which are valid within the region alphai to alphaf'''
    '''make a list of alpha, 8 orders of magnitude from formation'''
    '''for all calculations, scale factor = 1 when 1g PBHs form'''
    global maximum
    alphai,alphaf = -3,25
    Tform = T0(MBH)
    alphaform = float(log(aT(Tform),10))
    if alphaform > -0.001 and alphaform < 0.001:
        alphaform  = 0.0
    '''set the initial conditions'''
    '''the initial conditions pertain to the solver, which starts at the formation of the PBHs'''
    num=200
    Rini,BHini,Tini = float(rad(Tform) * 10**(4*alphaform)),float(rad(Tform) * 10**(4*alphaform) * betaf),Tform 
    ratio = (3/4*2.7*Tform)/Rini 
    sini = 2 * pi * 10**(3*alphaform) * Tini**3 * 106.75 /45 
    ics = [log(MBH),log(Rini),log(BHini),Tini,sini]
    Mlist,radlist,pbhlist,Tlist,alphatotal,slist = [],[],[],[],[],[]
    alphapre = np.linspace(alphaform+alphai,alphaform,num) 
    evolpre = odeint(coupled,Ta(10**(alphaform)),alphapre)
    for j in range(0,np.size(alphapre)):
        Mlist.append(0)
        radlist.append(Rini)
        pbhlist.append(0)
        Tlist.append(evolpre[:,0][j])
        alphatotal.append(alphapre[j])
        slist.append(sini)
    '''solve the differential equations'''
    masscut.terminal = True
    evolivp = solve_ivp(coupledivp,[alphaform,alphaf],ics,method="BDF",args=tuple([MBH,Mv,MN]))
    
    evol,alphalist1 = evolivp.y,evolivp.t
    for k in range(0,np.size(evol[1])):
        Mlist.append(e**(evol[0][k]))
        radlist.append(exp(evol[1][k]))
        pbhlist.append(exp(evol[2][k]))
        Tlist.append(evol[3][k])
        slist.append(evol[4][k])
        alphatotal.append(alphalist1[k])
    alphacut = alphatotal[-1]
    
    alphalistT = np.linspace(alphatotal[-1],alphaf,100)
    evolT = odeint(coupled,Tlist[-1],alphalistT)
    for j in range(0,np.size(alphalistT)):
        if alphalistT[j] > alphatotal[-1]:
            Mlist.append(0)
            radlist.append(radlist[-1])
            pbhlist.append(0)
            Tlist.append(evolT[:,0][j])
            alphatotal.append(alphalistT[j])
            slist.append(slist[-1])

    Talpha = interp1d(alphatotal,Tlist,fill_value="extrapolate")
    Mlist,Malphatotal = clean(Mlist,alphatotal,alphacut)
    Malpha = interp1d(Malphatotal,Mlist,fill_value = "extrapolate")
    radalpha = interp1d(alphatotal,radlist,fill_value = "extrapolate")
    pbhlist,pbhalphatotal = clean(pbhlist,alphatotal,alphacut)
    pbhalpha = interp1d(pbhalphatotal,pbhlist,fill_value = "extrapolate")
    salpha = interp1d(alphatotal,slist,fill_value="extrapolate")
    alphaT = interp1d(Tlist,alphatotal,fill_value = "extrapolate")
    return [Malpha,radalpha,pbhalpha,Talpha,salpha,alphaT,alphatotal]