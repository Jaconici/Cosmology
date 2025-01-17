'''imports'''
from math import *
from sympy import polylog
from scipy.integrate import odeint, solve_ivp, quad, simpson
import numpy as np
from numpy import dot,transpose
import random
from scipy.special import kn
from scipy.interpolate import interp1d,  RegularGridInterpolator ,splrep,splev
import cmath as cm
from scipy.special import *
from decimal import Decimal, getcontext
import os
import importlib.machinery
import importlib.util
import glob

path, filename = os.path.split(os.path.realpath(__file__))


loaderDMz = importlib.machinery.SourceFileLoader( 'DMz', path + '/DMZprime.py')
specDMz = importlib.util.spec_from_loader( 'DMz', loaderDMz )
DMz = importlib.util.module_from_spec( specDMz )
loaderDMz.exec_module( DMz )


'''GeV convsersions'''
GeV_in_g = 1.782661907e-24  # 1 GeV in g
cm_in_GeV = 1/1.9733e-14
s_in_GeV = 1/6.5821e-25


'''parameters'''
MPBH = 1 / GeV_in_g    #mass of the PBHs in GeV where numerical factor is the grams
g = 106.75 #SM model high temperature d.o.f






''''constants'''
GCF = 6.70883E-39
Mpl =1.22e19 # [M_pl] = GeV #Mpl in GeV
h =0.673
Mpls = Mpl * (90/(8*pi**3 *g))**0.5 #the reduced planck mass
vEW = 174 #Higgs vev in GeV

Msolar = 1.116E57 #Solar mass in GeV
TEWPT = 160 #Temperature of the EW phase transition

rhoc0 = 1.878E-29 * h**2 / GeV_in_g / (1.9733E14)**3 #critical density in GeV^4




'''particle  masses, see PDG review'''
mtau = 1.776
me = 0.511E-3
mmuon = 105.7E-3
Mleptons = [me,mmuon,mtau]

splitsolar = np.sqrt(7.6*1e-5)*1e-9 #solar neutrino mass mixing
splitatm = np.sqrt(2.47*1e-3)*1e-9 #atmospheric neutrino mass mixing in GeV


Wmass = 80.377

mu = 2.16e-3
md = 4.67e-3
ms = 93e-3
mc = 1.274
mb = 4.18
mtop = 174


'''Coupling constants, see https://pdg.lbl.gov/2023/reviews/rpp2022-rev-standard-model.pdf'''
a_fs = 1/137.036 #Fine structure constant
G_Fermi = 1.166e-5
g_EW = np.sqrt(G_Fermi * 8 * Wmass**2 / np.sqrt(2))













































def T0(M,acc = 1000):
    '''calculates the temperature at which a PBH of mass M forms, for relativistic degrees
    of freedom g'''
    #M is in grams
    #returns in GeV
    Tlist = np.logspace(12,-10,acc)
    for T in Tlist:
        Tform = 0.5 * ( 5 / (gs(T) * pi**3))**0.25 * np.sqrt( 3*0.2*Mpl**3 / M)
        if T <= Tform:
            return Tform
            break
        

def aT0(T):
    ''' scale factor in terms of the temperature. The scale factor is normalised to  = 1 today'''
    coef =  (30*rhoc/(pi**2*g))**(1/4)
    return float(coef/(T))
def Ta0(a):
    '''The Temperature in terms of the scale factor. The scale factor is normalised to  = 1 today'''
    coef =  (30*rhoc0/(pi**2*g))**(1/4)
    return coef/(a)
aformation = 8.539941141605994e-29                                                  #scale factor when the PBHs of MPBH = 1g form
def aT(T):
    ''' scale factor in terms of the temperature. The scale factor is normalised to  = 1 when a PBH of mass 1g is formed, hence the hard coded value'''
    coef =  (30*rhoc0/(pi**2*g))**(1/4)
    return float(coef/(T*aformation))
def Ta(a):
    '''The Temperature in terms of the scale factor. The scale factor is normalised to  = 1 when a PBH of mass 1g is formed, hence the hard coded value'''
    coef =  (30*rhoc0/(pi**2*g))**(1/4)
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
    excluded =[]
    for j in range(0,np.size(tarray)):
        for k in range(0,np.size(tarray)):
            if k!=j:
                if tarray[j] == tarray[k]:
                    excluded.append(j)
    cleanarray,cleantarray = [],[]
    for i in range(1,np.size(array)):
        if (array[i] < array[i-1] or array[i] == 0) and taxrray[i] < minimum and tarray[i] != tarray[i-1] and i not in excluded:
            cleanarray.append(array[i])
            cleantarray.append(tarray[i])
    return(cleanarray,cleantarray)

def sa(funs,MBH):
    '''function which returns a semi-analytical solution for the rho_pbh and M_pbh friedmann eqns'''
    
    '''First, create an interpolated function for 1/H as a func. of alpha'''
    alphalist = np.linspace(0,30,10000)
    H = funs[7]
    Hlist = [H(a) for a in alphalist]
    invHlist = [1/H for H in Hlist]
    invHa = interp1d(alphalist,invHlist,fill_value="extrapolate")
    
    '''Extract the formation temperature and scale factor of the PBHs'''
    Tform = T0(MBH)
    alphaform = float(log10(aT(Tform)))
    
    '''Find the initial mass and energy densities after formation and calculate their ratio'''
    alphacheck = np.linspace(alphaform,alphaform+1,1000)
    for i in range(0,np.size(alphacheck)):
        if funs[0](alphacheck[i]) != 0 and funs[2](alphacheck[i]) != 0:
            alphaform = alphacheck[i]
            break
    if funs[2](alphaform) != 0:
        ratio = funs[0](alphaform)/funs[2](alphaform)
    else:
        ratio = 1
    '''Next calculate the initial hubble rate and cosmic time (universe rad-dominated until formation)'''
    Hini =  8*pi/(3*Mpl**2)*( funs[1](alphaform) )**(0.5)
    ti = 1/(2*Hini)
    
    '''make a list of alpha values and an empty list of t values'''
    tlist = []
    alphas = np.linspace(alphaform,30,100)
    
    '''for every value in the alpha list, integrate 1/H up to this value and work out the cosmic time'''
    for a in alphas:
        alpha = np.linspace(alphaform,a,100)
        tlist.append( simpson([invHa(A) for A in alpha],alpha) + ti)
    '''make interpolated function'''
    ta = interp1d(alphas,tlist,fill_value="extrapolate")
    
    '''calculate the value of the constant of integration'''
    C = MBH**3/3 + 416.3/(30720*pi) * Mpl**4 * ta(alphaform)
    
    '''find out when the evaporation occurs, acut'''
    alphatry = np.linspace(0,35,5000)
    for a in alphatry:
        t = ta(a)
        M = ((-416.3/(30720*pi) * Mpl**4 * t +  C ) * 3 )**(1/3)
        if M < 3*MBH/4 and M > 0:
            acut = a
            break

    '''make two lists, one up to acut and then one from acut to a bit beyond'''
    alpha1 = np.linspace(0,acut,100)
    alpha2 = np.linspace(acut,acut+0.5,99900)
    

    Mlist,alphatotal,rholist = [],[],[]
    alphaarray = [alpha1,alpha2]
    '''for every value of alpha, check if MPBH is defined and if so add it to Mlist, then scale energy density'''
    for alphalist in alphaarray:
        for a in alphalist:
            if a < alphaform:
                Mlist.append(0)
                alphatotal.append(a)
                rholist.append(0)
            else:
                t = ta(a)
                if -416.3/(30720*pi) * Mpl**4 * t +  C < 0:
                    Mlist.append(0)
                    rholist.append(0)
                    alphatotal.append(a)
                else:
                    Mlist.append(((-416.3/(30720*pi) * Mpl**4 * t +  C ) * 3 )**(1/3))
                    alphatotal.append(a)
                    rholist.append(((-416.3/(30720*pi) * Mpl**4 * t +  C ) * 3 )**(1/3) / ratio)
    
    '''make the interpolated functions for M and rho'''
    Mi = interp1d(alphatotal,Mlist,fill_value='extrapolate')
    rhoi = interp1d(alphatotal,rholist,fill_value='extrapolate')
    def Ma(alpha):
        if alpha > alphatotal[-1] or alpha < alphaform:
            return 0
        else:
            return Mi(alpha)
    def rhoa(alpha):
        if alpha > alphatotal[-1] or alpha < alphaform:
            return 0
        else:
            return rhoi(alpha)
    return [Ma,rhoa]

def functions(MBH,betaf,MN,Mv,rtol = 1e-5,atol = 1e-5,output = 1,semia = True):
    '''function which returns the functions Ta, aT, Ha, sa which are valid within the region alphai to alphaf'''
    '''make a list of alpha, 8 orders of magnitude from formation'''
    '''for all calculations, scale factor = 1 when 1g PBHs form'''
    global maximum
    alphai,alphaf = -1,25
    Tform = T0(MBH)
    alphaform = float(log(aT(Tform),10))
    if alphaform > -0.001 and alphaform < 0.001:
        alphaform  = 0.0
    '''set the initial conditions'''
    '''the initial conditions pertain to the solver, which starts at the formation of the PBHs'''
    num=200
    Rini,BHini,Tini = float(rad(Tform) * 10**(4*alphaform)),float(rad(Tform) * 10**(3*alphaform) * betaf),Tform 
    sini = 2 * pi * 10**(3*alphaform) * Tini**3 * 106.75 /45 
    ics = [log(MBH),log(Rini),log(BHini),Tini,sini]
    
    Mlist,radlist,pbhlist,Tlist,alphatotal,slist = [],[],[],[],[],[]
    alphapre = np.linspace(alphaform+alphai,alphaform,num) 
    for j in range(0,np.size(alphapre)):
        Mlist.append(0)
        radlist.append(Rini)
        pbhlist.append(0)
        Tlist.append(Tini*(10**(-alphapre[j])/10**(-alphaform)))
        alphatotal.append(alphapre[j])
        slist.append(sini)
        
        
    '''solve the differential equations'''
    masscut.terminal = True
    finished = False
    alphai = alphaform
    while finished == False:
        evolivp = solve_ivp(coupledivp,[alphai,alphaf],ics,method="BDF",args=tuple([MBH,Mv,MN]),events=[masscut],rtol=rtol,atol=atol)
        if (e**(evolivp.y[0][-1]) < Mpl*10 and e**(evolivp.y[0][-1]) > 0) or evolivp.t[-1] == alphai:
            finished = True
            break
        else:
            alphai = evolivp.t[-1]
            alphaf = alphai*1.1
            evol,alphalist1 = evolivp.y,evolivp.t
            for k in range(0,np.size(evol[1])):
                if exp(evol[2][k]) > 0:
                    pbhlist.append(exp(evol[2][k]))
                    Mlist.append(e**(evol[0][k]))
                    radlist.append(exp(evol[1][k]))
                    Tlist.append(evol[3][k])
                    slist.append(evol[4][k])
                    alphatotal.append(alphalist1[k])
            ics = [log(Mlist[-1]),log(radlist[-1]),log(pbhlist[-1]),Tlist[-1],slist[-1]]
        if output ==3:
            print("Solved until ratio = " + str(Mlist[-1]/Mpl))
    alphacut,Tcut = alphatotal[-1],Tlist[-1]
    alphaf = 25

    alphalistT = np.linspace(alphatotal[-1],alphaf,100)
    for j in range(0,np.size(alphalistT)):
        if alphalistT[j] > alphatotal[-1]:
            Mlist.append(0)
            radlist.append(radlist[-1])
            pbhlist.append(0)
            Tlist.append(Tcut * (10**(-alphalistT[j])/10**(-alphacut)) )
            alphatotal.append(alphalistT[j])
            slist.append(slist[-1])
            
    Hlist = [( 8*pi/(3*Mpl**2)*(pbhlist[i] * 10**(-3*alphatotal[i]) + radlist[i] * 10**(-4*alphatotal[i])))**(0.5)
            for i in range(0,np.size(alphatotal))]
    Halpha = interp1d(alphatotal,Hlist,fill_value="extrapolate")
    Talpha = interp1d(alphatotal,Tlist,fill_value="extrapolate")
    Malph = interp1d(alphatotal,Mlist,fill_value = "extrapolate")
    def Malpha(alpha):
        if alpha >= alphaform and alpha < alphatotal[-1]:
            return Malph(alpha)
        else:
            return 0
    radalpha = interp1d(alphatotal,radlist,fill_value = "extrapolate")
    pbhalph = interp1d(alphatotal,pbhlist,fill_value = 'extrapolate')
    def pbhalpha(alpha):
        if alpha >= alphaform and alpha < alphatotal[-1]:
            return pbhalph(alpha)
        else:
            return 0
    salpha = interp1d(alphatotal,slist,fill_value="extrapolate")
    alphaT = interp1d(Tlist,alphatotal,fill_value = "extrapolate")
    funs1 = [Malpha,radalpha,pbhalpha,Talpha,salpha,alphaT,alphatotal,Halpha]
    if semia == True:
        analytic = sa(funs1,MBH)
        return [analytic[0],radalpha,analytic[1],Talpha,salpha,alphaT]
    else:
        return [Malpha,radalpha,pbhalpha,Talpha,salpha,alphaT]





MEavTab  = np.loadtxt(path+"/Data/timedil.txt")

MEadata = splrep(MEavTab[:,0],  MEavTab[:,1],  s=0)

def ME(zBH):
    LzBH = np.log10(zBH)
    if LzBH < -4.:
        return 10.**(-0.5323925683174364 + LzBH)
    elif LzBH >= -4. and LzBH <= 2.85:
        return splev(LzBH, MEadata, der=0)
    else:
        return 1.



path, filename = os.path.split(os.path.realpath(__file__))
gsdata  = np.genfromtxt(path+"/Data/gs.txt")
gsTlist,gslist = gsdata[:,0],gsdata[:,1]
gsint = interp1d(gsTlist,gslist,fill_value = 'extrapolate')



def gs(T):
    '''Returns the effective number of entropic elativistic degrees of freedom, interpolated from Fig 6  panel a in https://arxiv.org/pdf/1609.04979.pdf'''
    '''T is given in GeV'''
    if T > 1e3:
        return gsint(1e3*1e3)
    elif T < 1e-5:
        return gsint(1e-5*1e3)
    else:
        return gsint(T*1e3)














def mm(mh):
    '''Defines the middle neutrino mass m2 for some heaviest mass mh,
    mh must be greater than 0.05eV'''
    return np.sqrt(-splitatm**2 + mh**2)  #
def ml(mh):
    '''Calculates the lightest neutrino mass for some heaviest scale mh, mh must be greater than 0.05eV'''
    m2 = mm(mh)
    return np.sqrt(-splitsolar**2 + m2**2)      

def meff(xf,yf,mhf,form = "Y",M = None,i=1):
    '''effective mass of the light neutrinos, defined in Nardi review page 74 bottom and Strumina w/ mistakes
    R is approximated in the 2 neutrino case, taking m_sol << m_atm
    form defines which variable is used to calculate the effective mass, the Yukawa or the CI matrices
    if form = Y, user must provide the right handed neutrino mass matrix'''
    
    
    #mhf is the light neutrino mass matrix
    #R is the Casas Ibarra matrix
    #xf and yf are the real and imaginary part of the single relevant mixing angle
    
    '''calculate the neutrino mass matrix'''
    m = Mvgen(mhf,NO=False)
    
    
    if form == "R":
        '''Calculate the effective mass from a fixed R matrix and m.'''
        
        '''calculate the sin and cosine of the mixing angle'''
        theta = xf + 1j*yf
        s = cm.sin(theta)
        c = cm.cos(theta)
    
        '''The Casas-Ibarra matrix for two neutrinos, see 0312203 pg 5 top'''
        R = [[0,   c, s],
             [0., -s , c],
             [0, 0., 1]]
        
        '''calculate and return mtilde'''
        mtilde = 0
        for i in range(0,3):
            mtilde += m[i]*np.abs(R[0][i])**2
        return mtilde
    
    elif form == "Y":
        '''calculate the neutrino mass from the Yukawa couplings'''
        
        '''calculate the Yukawa matrix and its product Y^\daggerY'''
        Y = Ygen(xf,yf,M,mhf)
        YY = np.array(dot(np.matrix(Y).H,Y))
        
        '''calculate and return the effective mass'''
        return np.abs(YY[i-1][i-1]) * vEW**2 / M[i-1]
    
    elif form == "approx":
        '''Exact when  = m2 = 0'''
        return mhf * np.abs(cm.sin(xf+1j*yf)**2)
    
    
    
def Mvgen(m_h,NO = True):
    if NO == True:
        return [ml(m_h),mm(m_h),m_h]
    elif NO == False:
        return [mm(m_h),ml(m_h),m_h]
        

def Ygen(x,y,M,mh,alpha23=pi/4,delta=0,param = '23',massless= False,degen = False,NO = True):
    '''this function generates and returns the Yukawa matrix Y
    by solving the equation (10), works for a general structure for Mhat but specifically 2 neutrino case'''
    '''since the active and right handed neutrino masses are fixed, the yukawa matrix returned is the one which
    produces the requested active neutrino spectrum, for the right handed mass spectrum given'''
    alpha21 = float(pi/5)
    alpha31 = alpha21-alpha23                               #only the difference, the majorana phase 23 is physical
    theta_12 = 33.44 * pi/180.
    theta_13 = 8.57 * pi/180.
    theta_23 = 49.2 * pi/180.
    
    c12 = cos(theta_12)
    s12 = sin(theta_12)
    
    c13 = cos(theta_13)
    s13 = sin(theta_13)
    
    c23 = cos(theta_23)
    s23 = sin(theta_23)
    M_hat = np.diag(M)

    Mv = Mvgen(mh,NO=NO)
    m_1 = Mv[0]
    m_2 = Mv[1]                                                  # GeV
    m_3 = Mv[2] # GeV
    if massless == True:  
        m_nu_hat = [[0, 0., 0.],
                [0., splitsolar, 0.],
                [0., 0., splitatm]]
    elif massless == False:
        if degen == True:     
            m_nu_hat = [[m_1, 0., 0.],
                    [0., m_1, 0.],
                    [0., 0., m_3]]
        elif degen == False:
            m_nu_hat = [[m_1 + 0j, 0.+ 0j, 0.+ 0j],
                    [0.+ 0j, m_2+ 0j, 0.+ 0j],
                    [0.+ 0j, 0.+ 0j, m_3+ 0j]]

    theta = x + 1j*y

    U_1 = np.array([[0. + 0.j, 0. + 0.j, 0. + 0.j], 
                [0. + 0.j, 0. + 0.j, 0. + 0.j],
                [0. + 0.j, 0. + 0.j, 0. + 0.j]])

    U_1[0, 0] += c12 * c13
    U_1[0, 1] += s12 * c13
    U_1[0, 2] += s13 * np.exp(- 1j * delta)
    
    U_1[1, 0] += - s12 * c23 - c12 * s23 * s13 * np.exp(1j * delta)
    U_1[1, 1] += c12 * c23 - s12 * s23 * s13 * np.exp(1j * delta)
    U_1[1, 2] += s23 * c13
    
    U_1[2, 0] += s12 * s23 - c12 * c23 * s13 * np.exp(1j * delta)
    U_1[2, 1] += - c12 * s23 - s12 * c23 * s13 * np.exp(1j * delta)
    U_1[2, 2] += c23 * c13
    
    U_2 = [[1., 0.                  , 0.                  ],
           [0., np.exp(1j * alpha21/2), 0.                  ],
           [0., 0.                  , np.exp(1j * alpha31/2)]]

    U = dot(U_1, U_2)
    #U = U_1
    s_theta = complex(cm.sin(theta))
    c_theta = complex(cm.cos(theta))
    if param == 'Petcov':
        O = [[0. + 0j,   c_theta, s_theta],
             [0. + 0j, - s_theta, c_theta],
             [1. + 0j,        0. + 0j,      0. + 0j]]
    elif param == '23':
             O = [[1.,    0.,    0.],
             [0.,   c_theta, -s_theta],
             [0.,  s_theta, c_theta]] 
    elif param == '13':
        O = [[c_theta , 0, s_theta],
             [ 0,   1,    0],
             [-s_theta, 0, c_theta]]
        
        
        
    
    return 1/vEW * dot(np.sqrt(M_hat), dot(O, dot(np.sqrt(m_nu_hat),np.matrix(U).H)))
    #return 1/vEW * dot(np.matrix(U), dot(np.sqrt(m_nu_hat),dot(np.transpose(O) ,np.sqrt(M_hat))))

def I(iN,jN,l,YY,Y,M):
    '''calculating the quantity Iij,aa from eqn 36 (Petcov)'''
    '''iN and jN are both the indexes for the Yukawa matrix not labels'''
    '''alpha is the index for the flavour'''
    '''YY is the product Y^daggerY, Y is the Yukawa matrix'''
    
    '''first define the hermitian conjugate of Y'''
    YH = np.array(np.matrix(Y).H)
    
    '''calculate I'''
    denom = YY[iN][iN]*YY[jN][jN]
    term1 = complex(YH[iN][l]*Y[l][jN]*YY[iN][jN]).imag
    term2 = M[iN]/M[jN]*complex(YH[iN][l]*Y[l][jN]*YY[jN][iN]).imag
    num =  term1 + term2
    #return complex(YY[iN][jN]**2).imag/denom
    return num/denom


def xTz(z,Gamma11,Gamma22,Gamma12):
    '''calculates the thermal corrections to the mass splitting'''
    
    return pi/(4*z**2) * np.sqrt ( (1-Gamma11/Gamma22)**2 + 4*np.abs(Gamma12)**2/Gamma22**2) 
    
def dMz(M,T,Gjj,Gii,Gij):
    '''calculates the thermal corrections to the mass splitting
    M is the common mass of the degenerate neutrinos N_i and N_j'''
    z = M/T
    
    
    return pi*Gjj/(4*z**2) * np.sqrt ( (1-Gii/Gjj)**2 + 4*np.abs(Gij)**2/Gjj**2)

def epsilon(MN,Y,i,l,T,Mv,k = 5,TC = True):
    '''The flavour dependent CP asymmetry in the decay of N_i such that i is an index not the label
    MN is the (diagonal) mass matrix for the RHNs,
    Mv is the (diagonal) active neutrino mass matrix
    T is the temperature of the local plasma
    Y is the Yukawa matrix
    l is the label of the flavour so that l = e,mu,tau (0,1,2)
    if any of the RHNs are degenerate, it is with N_2 so the mass splitting is always defined N_i-N_2
    k is the index of the decoupled RHN which is not summed over'''
    
    '''First initialise epsilon'''
    eptotal=0
    
    '''then extract the particle physics parameters'''
    M1,M2,M3 = MN[0],MN[1],MN[2]
    z = M2/T
    
    '''Next, define the Yukwa matrix products'''
    YY,YH = np.array(dot(Y.H,Y)), np.array(Y.H)
    Y = np.array(Y)
    
    '''Calculate the diagonal and off diagonal decay rates'''    
    #Gamma22,Gamma11 = Gii(M,Mv,x,y,1),Gii(M,Mv,x,y,0)
    Gii = Gammaii(YY,i+1,MN)
    #Gii = GN(Y,T,MN,i=i)
    G22 = Gammaii(YY,2,MN)
    if Gii == 0:
        return 0

    
    '''Now having calculated the necessary constants, perform the sum'''
    for j in [0,1,2]:
        if j != i and j != k:
            
            '''Calculate the masses and decay widths of N_j,i'''
            Mi,Mj = MN[i],MN[j]
            dM = Mi - Mj

            if Mi == Mj:
                print('Exactly degenerate neutrinos detected')
            
            #Gjj = GN(Y,T,MN,i=j)
            Gjj = Gammaii(YY,j+1,MN)
            Gij = Gammaij(YY,i+1,j+1,MN) #check if this should be thermally corrected
            
            '''Then calculate the thermal corrections to the mass splitting'''
            dMT = dMz(MN[1],T,Gjj,Gii,Gij) #this assumes z = M[i]/T
            DM = dM + dMT
            
            
            '''Calculate the self energy regulator as in Pilaftsis, no thermal corrections'''
            fSE = (Mi + Mj)*dM * Mi * Gjj / ( ((Mi + Mj)*dM)**2 + Mi**2 * Gjj**2)

            
            '''Calculate the quantity Iij,aa and the sign of Mi-Mj'''
            Iij = I(i,j,l,YY,Y,MN) #

            sgn = (Mi-Mj)/(np.abs(Mi-Mj)) #
            
            '''calculate the decay asymmetry and add it to the total'''
            Numerator = 2*dM*gammaf(z)*Gjj #
            Denominator = 4*DM**2 + (gammaf(z)*Gjj)**2 
            if TC == True:
                eptotal += sgn*Iij*Numerator/Denominator
                #eptotal += Iij
            else:
                eptotal += Iij * sgn * fSE
    return eptotal  

gam = np.genfromtxt('Data/gamma.txt')
gammaf = interp1d(gam[:,0],gam[:,1],fill_value="extrapolate")

def gf(z):
    if z < 0.93 and z > 0.34:
        return 0
    else:
        return gammaf2(z)

def Gammaii(YY,i,M):
    '''returns the total decay rate of the heavy neutrino i'''
    '''i is the label not the index'''
    iN = i-1
    return np.abs(YY[iN][iN])*M[iN]/(8*pi)

def Gammaij(YY,i,j,M):
    '''returns the off diagonal decay rate Gamma_ij'''
    '''i and j are labels not indices'''
    '''M is the heavy neutrino mass matrix'''
    '''YY is the Yukawa matrix product Y^daggerY'''
    iN,jN = i-1,j-1
    return YY[iN][jN]*np.sqrt(M[iN]*M[jN])/(8*pi)

def Gij2(MN,Mv,x,y):
    return (MN[0]*MN[1])**2 * ((Mv[1] - Mv[2])**2 * (cos(x)*sin(x))**2 + (Mv[1] + Mv[2])**2 *
                               (cosh(y)*sinh(y))**2)/(4*64*pi**2*vEW**4)

def Gii(MN,Mv,x,y,i):
    return MN[i]**2 * ( (Mv[1]-Mv[2]) *cos(2*x) + (Mv[1] + Mv[2]) * cosh(2*y))/(32*pi*vEW**2)




def GN(Y,T,M,i=0):
    YY = np.array(dot(np.matrix(Y).H,Y))
    Mh,Ml= MH(T),Mlep(T,0)
    MN = M[i]
    if MN < Mh + Ml:
        return 0
    else:
        aH,aL = (Mh**2/M[i]**2),(Ml**2/M[i]**2)
        G = M[i]/(8*pi) * YY[i][i] * (1-aH + aL) * np.sqrt(Lambda(1,aH,aL)) 
        if M[i] > Mh + Ml:
            return M[i]/(8*pi) * YY[i][i] * (1- aH + aL) * np.sqrt(Lambda(1,aH,aL))
        elif Mh > Ml + M[i]:
            return Mh/(16*pi) * YY[i][i] * (-1+ aH - aL) * np.sqrt(Lambda(aH,1,aL)) / (aH**2)


def Lambda(a,b,c):
    return (a-b-c)**2 - 4*b*c

def mykn(order,z):
    ''' custom function for the modified bessel functions which returns a tiny non-zero value for all z > 600'''
    '''order is the order of the bessel function and z its argument'''
    if z > 600:
        return 2.1E-218
    else:
        return kn(order,z)

def rate(Gamma, M, T,inverse = False):
    z = M/T
    if inverse == False:
        return mykn(1,z)/mykn(2,z) * Gamma
    elif inverse == True:
        return mykn(1,z)/mykn(2,z) * Gamma * np.exp(-M/T)
    else:
        print("Decay or inverse decay rate incorrectly specified, calculating decay rate.")
        return mykn(1,z)/mykn(2,z) * Gamma
    
def Hf(alpha,pbhf,radf):
    '''function which returns the Hubble rate for comoving pbh and radiation densities pbhf and radf'''
    pbh,rad = pbhf(alpha),radf(alpha)
    return ( 8*pi/(3*Mpl**2)*(pbh * 10**(-3*alpha) + rad* 10**(-4*alpha)))**(0.5)

def Neqsa(T,mass):
    '''semi-approximated version of the function Neq'''
    '''uses the asymptotic limits of the integral form to avoid integration except in the narrow range z = 0.1 to  z = 10'''
    '''mass and T are always provided'''
    g = 0
    if mass/T < 1E3:
        z = mass/T
        if z < 0.1:
            return Neqrel(T)  
        if z > 10:
            return Neqnrel(T,mass) 
        else:
            p_list = np.logspace(float(log10(T/10)), float(log10(T*100)), 100)
            E_list = [sqrt(mass**2 + p**2) for p in p_list]
            int_list = [1/(e**(E_list[i]/T)+ 1)*p_list[i]**2 for i in range(0,100)]
            integral = simpson(int_list, p_list)
            return 2 * g/(2 * np.pi)**2 * integral
    else:
        return 0.
    
def Neqrel(T):
    '''the relativistic limit of Neq'''
    g = 2
    return 0.75 * g * 1.2 * T ** 3 / pi**2 

def Neqnrel(T,mass):
    '''the non-relativistic limit of Neq'''
    g = 2
    return  g * ( mass * T / (2 * pi) ) ** 1.5 * np.exp(-mass/T) 

def TH(M):
    '''returns the Hawking temperature in GeV'''
    if M == 0:
        print("uh oh")
        return 1E-20
    else:
        return float( 1 / ( 8 * pi * GCF * M))

def THinv(TBH):
    '''returns the mass of PBH which gives Hawking temperature TBH in GeV'''
    return float( 1 / ( 8 * pi * GCF * TBH))



swap,za,factor=0,0,1



def Gammaf(species,MBH,M2):
    '''this function returns the analytic solution to equation (A2) in arXiv:2107.00013v2, given by (A4)'''
    if MBH == 0:
        return 0
    TPBH = TH(MBH) #Hawking temperature
    if species=="RHN":
        s = 1/2
        gs = 2 #set the mass, spin and degrees of freedom
    X = M2/TPBH
    ep = (-1)**(2*s) #epsilon function, depends on spin
    L2 = polylog(2,ep*np.exp(-X))
    L3 = polylog(3,ep*np.exp(-X))
    return float((ep*27*gs/(512*pi**4))/(GCF*MBH) *( X*L2 + L3))


def Chi(T):
    Phi = v(T)
    return 4*(27*(Phi/T)**2 + 77)/(333*(Phi/T)**2 + 869)
    
def GB(T):
    Phi = v(T)
    aW = 1e-6
    Tdiff = T**4 * e**( - 147.7 + 0.83*T)
    return 9 * (869 + 333*(Phi/T)**2)/(792 + 306*(Phi/T)**2) * Tdiff / T **3


def dB(alpha,B,Deltaf,Tf,pbhf,radf):
    rPBH,rRad,T = pbhf(alpha),radf(alpha),Tf(alpha)
    H =  ( 8*pi/(3*Mpl**2)*(rPBH * 10**(-3*alpha) + rRad* 10**(-4*alpha)))**(0.5)
    Delta = Deltaf(alpha)
    GammaB = GB(T)
    rate = -GammaB * (B + Chi(T) * Delta) * log(10)/H
    return rate






GTdata = np.genfromtxt('Data/GT.txt')
Glist = GTdata[:,1]
Tlist = GTdata[:,0]
Glist = [e**(Glist[i]) for i in range(0,np.size(Tlist))]
GT = interp1d(Tlist,Glist,fill_value='extrapolate')

def sphal(funs):
    '''function which returns the sphaleron freeze out temperature by solving numerically the equation
    eq 9 in 1404.3565'''
    
    '''first extract the cosmological functions'''
    Malpha,radalpha,pbhalpha,Talpha,salpha,alphaT = funs[0],funs[1],funs[2],funs[3],funs[4],funs[5]
    
    '''Make a list of possible values of alpha* and convert that to T'''
    Tlist = np.linspace(120,170,1000)
    alphalist = [alphaT(T) for T in Tlist]
    Hlist = [( 8*pi/(3*Mpl**2)*(pbhalpha(a) * 10**(-3*a) + radalpha(a)* 10**(-4*a)))**(0.5) for a in alphalist]
    
    '''iterate over the possible values of T* to populate the lists of the left and right hand sides of the eqn'''
    Llist,Rlist =[],[]
    for i in range(0,np.size(alphalist)):
        a,T,H = alphalist[i],Tlist[i],Hlist[i]
        G = GT(T)
        Llist.append(G)
        Rlist.append(0.1015*H/T)

    '''iterate over the lists to determine the crossing point'''
    sign = (Llist[0]-Rlist[0])/np.abs(Llist[0]-Rlist[0])
    jsphal = 0
    finished = False
    for j in range(0,np.size(Llist)):
        if sign == 1:
            if Llist[j] < Rlist[j]:
                jsphal = j
                finished = True
                break
        elif sign == -1:
            if Llist[j] > Rlist[j]:
                jsphal = j
                finished = True
                break
    '''determine alphasphal and Tsphal'''
    if jsphal == 0 and finished == False:
        print("Crossing above than T = 170 GeV")
    if jsphal == 0 and finished == False:
        print("Crossing below T = 120 GeV...")
    return Tlist[jsphal]


def dManalytic3(x,y,M1,Mv):
    num = M1**2 * (Mv[2] * cm.cos(x + 1j*y) * cm.cos(x - 1j*y) + Mv[1] * cm.sin(x + 1j*y) * cm.sin(x - 1j*y))
    denom = 8*pi*vEW**2 - M1 * Mv[2] * cm.cos(x + 1j*y) * cm.cos(x - 1j*y) - M1* Mv[1] * cm.sin(x + 1j*y) * cm.sin(x - 1j*y)
    return num/denom

def U2a(Mv,MN,x,y):
    '''Returns the value of U^2 for active neutrino mass matrix Mv, RHN mass matrix M and x and y the real and imaginary parts of mixing angle'''
    m1,m2,m3 = Mv[0],Mv[1],Mv[2]
    M1,M2,M3= MN[0],MN[1],MN[2]
    return  2*m1/M3 - (M1-M2)*(m2-m3)*cos(2*x)/(M1*M2) + (M2 + M1) * (m2+m3)*cosh(2*y)/(M1*M2)

def U2by2(MN,x,y):
    '''Returns the value of U^2 for active neutrino mass matrix Mv, RHN mass matrix M and x and y the real and imaginary parts of mixing angle'''
    M2,M3= MN[0],MN[1]
    return ( (M2 - M3) * (splitatm - splitsolar)*cos(2*x) + (M2 + M3)*(splitsolar + splitatm) * cosh(2*y))/(M2*M3)



def Uinv(MN,Mv,U,x):
    M1,M2,M3 = MN[0],MN[1],MN[2]
    m1,m2,m3 = Mv[0],Mv[1],Mv[2]
    term = U - (2*m1/M3) - (M2-M1)*(m1-m2)*cos(2*x)/(M1*M2)
    print(term)
    if term <= 0:
        return 0
    else:
        arg = M1*M2/((m1+m2)*(m2+m3)) * term
        return np.arccosh(float(arg))/2
    
def Uinv2by2(MN,U,x):
    M2,M3 = MN[0],MN[1]
    arg = ((U * M2 * M3 ) - (M2-M3)*(splitatm - splitsolar)*cos(2*x))/((M2+M3)*(splitsolar+splitatm))
    if arg <= 1:
        return 0
    else:
        return np.arccosh(float(arg))/2
    

      



def avg(mass,T):
    def Boltzmann(E):
        return (np.exp(float(E/T))+1)**(-1)                   #Distribution of the cosmic temperature
    def fun(E):
        return Boltzmann(E) * mass/E                                       #Include Lorentz factor
    Elist = np.logspace(float(log(mass,10)),float(log(T*50,10)),1000)      #The integral is performed from the mass of the particle to 50* the temp
    Blist = [Boltzmann(E) for E in Elist]
    flist = [fun(E) for E in Elist]
    return simpson(flist,x=Elist)/simpson(Blist,Elist)     


def coupled(alpha,N,funs,M,Yf,Mvf,xf,yf,YY,P,ratio,scatter = [],negative=False,interp = False):
    kappa = 0.14
    '''extract the cosmological variables, S,H,T,alpha'''
    Malpha,radalpha,pbhalpha,Talpha,salpha,alphaT = funs[0],funs[1],funs[2],funs[3],funs[4],funs[5]
    T,MBH = Talpha(alpha),Malpha(alpha)
    dMf,z = M[1]-M[0],M[1]/T
    s,H,a = salpha(alpha) * 10**(-3*alpha), Hf(alpha,pbhalpha,radalpha),10**alpha
    if s == 0:
        return [0,0,0,0,0]
    S = salpha(alpha)
    if pbhalpha(alpha) > radalpha(alpha)/1e30:
        YPBH = pbhalpha(alpha)/MBH / S
    else:
        YPBH = 0
    '''define the entropy normalised quantities Yl, Yleq'''
    Mleptons = [ me, mmuon, mtau]
    Yleq = [ Neqrel(T,ml)/s for ml in Mleptons]
    #Yleq = [ N_ell_eq(alpha,Talpha)/S for ml in Mlep]
    YN,YL = [N[0],N[1]],[N[2],N[3],N[4]]
    YLtotal = YL[0] + YL[1] + YL[2]
    
    total = 0
    '''Calculate dTda'''
    epSM,epRHN =  epN(MBH,Mvf,M), epN(MBH,Mvf,M,species = "RHN")
    ep = epSM+epRHN
    total,kappa = 0, 416.3/(30720*pi) *  Mpl**4
    dTda = T*(1 - 0.25*epSM*10**alpha /(H*MBH**3)*pbhalpha(alpha)/radalpha(alpha))
    rates = []
    for k in (0,1,2):
        total = 0
        '''summing over N = N1,N2'''
        for i in (0,1):
            '''scattering rates'''
            gSt,gSs,gDi = scatter[0][i],scatter[1][i],scatter[2]
            '''RHN number densities'''
            
            #Yeq = 3/8*z**2*kn(2,z) * ratio / S

            Yeq = N_Ni_eq(alpha, Talpha, M[i])/s

            '''calculate the rate'''
            prefactor = log(10)/(H*S)*a**3
            decay =  gammaD(YY,i,z,M[i])
            gD = complex(decay[0]).real   #rate of production gammaD
            if interp == True:
                gD = gDi(T)
            '''CP asymmetry parameter'''
            eps =  epsilon(dM=dMf,M2f=M[1],Y=Yf,i=i+1,l=k,z=z,Mv=Mvf,x=xf,y=yf)
            if negative == True:
                eps  =  eps*decay[1]
            '''PBH production rate'''
            GPBH =  Gammaf("RHN",MBH,M[i])
            D = gD*((YN[i]/Yeq-1)*eps)
            W = -P[i][k]*YL[k]/Yleq[k]*( gD + 2*gSt(T) + YN[i]/Yeq * gSs(T) )
            dY = complex(prefactor*( D + W )).real
            if k == 0:
                DN = (1-YN[i]/Yeq)*(gD+gSs(T)*2 + gSt(T)*4)
                Nrate = prefactor * (DN + YPBH*GPBH)
                rates.append(Nrate)
            total += dY 
        rates.append(total)
    return rates


def N_Ni_eq(alpha,T_f,M): # comoving
        a = 10**alpha
        z = M/T_f(alpha)
        Ta = T_f(alpha) * a
        g_N = 2
        return z**2 * kn(2,z) * Ta**3/np.pi**2 * 3/4 * 1.2 * g_N/2

def N_ell_eq(alpha,T_f):
        a = 10**alpha
        T = T_f(alpha)
        g_ell = 2
        return T**3/np.pi**2 * 3/4 * 1.2 * g_ell * a**3


def solvecoupled(xf,yf,dM,M2f,Mv,funs=None,ic=0,MBH=None,betaf=None,delta = 0,zstart=1e-5,output = 3,rtol2=1e-10,atol2=1e-10,rtol3 = 1e-5,atol3 = 1e-20,inst = False,gammaS = None,negative = False,Scatter = False,Acc = 100,interp = False,NO = False):
    '''solving function for the BEs in the resonant leptogenesis (flavoured) case'''
    '''the leptogenesis parameter space spans x,y,dM,M2 and mhf and the PBH parameters are the mass and abundance'''
    '''if functions is not provided, MBH and betaf must be, and viceversa'''
    
    '''Calculate the RHN and active neutrino mass matrices'''
    mhf = Mv[-1]
    M = [M2f+dM,M2f,1e16]
    
    '''first calculate or extract the cosmological functions'''
    if funs == None:
        funs =  functions(MBH=MBH,betaf=betaf,MN= M,Mv=Mv,semia=semia)
    Malpha,radalpha,pbhalpha,Talpha,salpha,alphaT = funs[0],funs[1],funs[2],funs[3],funs[4],funs[5]
    HN =  Hf(alphaT(M2f),pbhalpha,radalpha)
    '''calculate the starting scale factor'''

    '''Find the moment of evaporation for the PBHs'''
    alphaevap = 0
    alphaform = alphaT( T0(MBH))
    alphalist = np.linspace(alphaform + 1,20,10000)
    for a in alphalist:
        if Malpha(a) < 1:
            alphaevap = a
            break
    
    '''extract relevant cosmological quantities'''
    Tstart = M2f/zstart
    alphai = alphaT(Tstart)
    ratio =  Neqsa(Talpha(alphai),me)/0.75
    Tsphal =  sphal(funs) 
    a150,a120 = alphaT(150),alphaT(120)
    
    '''Next, calculate the Yukawa matrix and its Hermitian conjugate'''
    Y =  Ygen(xf,yf,M,mhf,delta=delta)
    YY = np.array(dot(np.matrix(Y).H,Y))
    G  = [ Gii(M,Mv,xf,yf,0), Gii(M,Mv,xf,yf,1)]
    
    
    '''Calculate the flavour projectors'''
    P  = [[complex(np.abs(Y[l][j-1])**2/YY[j-1][j-1]).real for l in range(0,3)] for j in (1,2)]
    
    '''Check that the flavour projectors are correct'''
    if (P[0][0]+P[0][1]+P[0][2]) < 0.99 or (P[0][0]+P[0][1]+P[0][2]) > 1.01:
        if output > 1:
            print("Sum of i=1 flavour projectors is not equal to unity, check calculations and inputs")
    if (P[1][0]+P[1][1]+P[1][2]) < 0.99 or (P[1][0]+P[1][1]+P[1][2]) > 1.01:
        if output > 1:
            print("Sum of i=2 flavour projectors is not equal to unity, check calculations and inputs")    
    acc = Acc
    Tlist = np.logspace(log10(Tsphal),log10(Tstart),1000)
    if gammaS == None:
        '''Calculate and interpolate the gauge boson scattering rates'''
        gAtLA1,gAtLA2 = [ geq('AtLA',T,M,0,YY,y=True,accuracy=acc) +  geq('AtLA',T,M,0,YY,y=False,accuracy=acc) for T in Tlist],[ geq('AtLA',T,M,1,YY,y=True,accuracy=acc) +  geq('AtLA',T,M,1,YY,y=False,accuracy=acc) for T in Tlist]
        gAs1,gAs2     = [ geq('As',T,M,0,YY,y=True,accuracy=acc*10)+ geq('As',T,M,0,YY,y=False,accuracy=acc*10) for T in Tlist],[ geq('As',T,M,1,YY,y=True,accuracy=acc*10)+ geq('As',T,M,1,YY,y=False,accuracy=acc*10) for T in Tlist]
        gAtLH1,gAtLH2 = [ geq('AtLH',T,M,0,YY,y=False,accuracy=acc)+ geq('AtLH',T,M,0,YY,y=True,accuracy=acc) for T in Tlist],[ geq('AtLH',T,M,1,YY,y=False,accuracy=acc)+ geq('AtLH',T,M,1,YY,y=True,accuracy=acc) for T in Tlist]
        
        gAt1,gAt2     = [(gAtLH1[i] + gAtLA1[i])/2  for i in range(0,np.size(gAtLH1))],[(gAtLH2[i] + gAtLA2[i])/2 for i in range(0,np.size(gAtLH2))]
        
        gHs1,gHs2     = [geq('Hs',T,M,0,YY,y=False,accuracy = acc) for T in Tlist],[geq('Hs',T,M,1,YY,y=False,accuracy = acc) for T in Tlist]
        gHt1,gHt2     = [geq('Ht',T,M,0,YY,y=False,accuracy = acc) for T in Tlist],[geq('Ht',T,M,1,YY,y=False,accuracy = acc) for T in Tlist]
        
        gSs1,gSs2     = [gAs1[i] + gHs1[i] for i in range(0,np.size(Tlist))],[gAs2[i] + gHs2[i] for i in range(0,np.size(Tlist))]
        gSt1,gSt2     = [gAt1[i] + gHt1[i] for i in range(0,np.size(Tlist))],[gAt2[i] + gHt2[i] for i in range(0,np.size(Tlist))]
        
        gSsi,gSti     = [interp1d(Tlist,gSs1,fill_value='extrapolate'),interp1d(Tlist,gSs2,fill_value='extrapolate')],[interp1d(Tlist,gSt1,fill_value='extrapolate'),interp1d(Tlist,gSt2,fill_value='extrapolate')]
    else:
        gSti,gSsi = gammaS[0],gammaS[1]
    
    def zero(z):
        return 0

    gDlist2 = [gammaD(YY,0,M2f/T,M2f)[0] for T in Tlist]


    zeroTs,gDnew = [],[]
    for i in range(0,np.size(gDlist2)):
        if gDlist2[i] != 0:
            zeroTs.append(Tlist[i])
        
    gDlist3 = [gammaD(YY,0,M2f/T,M2f)[0] for T in zeroTs]

    gDi = interp1d(zeroTs,gDlist3,fill_value= 'extrapolate')
    
    scatter = [gSti,gSsi,gDi]
    '''Set the parameters of the numerical solvers'''
    if Scatter == True:
        return [gHs1,gAs1,gAt1,gHt1,gDlist2,Tlist]
    options2 = {'rtol': rtol2,'atol': atol2,'method' : 'BDF'}
    span = [alphai,a120] 
    Nic =  Neqsa(Tstart,M2f)/(salpha(alphai)*10**(-3*alphai))

    #Nic = N_Ni_eq(alphai, Talpha, M2f)/salpha(alphai)

    args = [funs,M,Y,Mv,xf,yf,YY,P,ratio,scatter,negative,interp] 
    ic = [Nic,Nic,0,0,0]

    '''Solve the asymmetry Boltzmann equations'''               
    NBL = solve_ivp(coupled,span,ic,args = tuple(args),**options2)

    '''extract the solutions'''

    NBLe,NBLmu,NBLtau = NBL.y[2],NBL.y[3],NBL.y[4]


    '''Make interpolated functions of the asymmetries'''
    NBLei,NBLmui,NBLtaui = interp1d(NBL.t,NBLe,fill_value="extrapolate"),interp1d(NBL.t,NBLmu,fill_value="extrapolate"),interp1d(NBL.t,NBLtau,fill_value="extrapolate")

    '''Make flavoured summed lists for the asymmetry density'''
    Sratio = salpha(5) / salpha(25)
    NBLT = [ (NBL.y[2][i] +  NBL.y[3][i] + NBL.y[4][i])*Sratio for i in range(0,np.size(NBL.y[2]))]

    '''interpolate the baryonic Yield for use in the B differential equation'''
    NBLi = interp1d(NBL.t,NBLT,fill_value = 'extrapolate')
    
    TB = 150
    aB = alphaT(TB)
    
    if inst == True:      
        return [[Ny1,Ny2],[NBLi,NBL.t],[NBLei,NBLmui,NBLtaui],alphaT,Talpha]
    
    elif inst == False:
        Ytotal,alphatotal = [],[]
        
        '''Solve the B differential equation'''
        Bsol = solve_ivp(dB,[aB,a120],[NBLi(aB) * Chi(Talpha(aB))],args=tuple([NBLi,Talpha,pbhalpha,radalpha])
                                 ,method = 'BDF',rtol = rtol3,atol = atol3)   

        '''Add the solution up to T = 150GeV'''
        for i in range(0,np.size(NBLT)):
            T = Talpha(NBL.t[i])
            if T > TB:
                Ytotal.append(NBLT[i] * Chi(T) )
                alphatotal.append(NBL.t[i])
            else:
                pass

        '''Add the solution up to T = 120GeV'''
        for i in range(0,np.size(Bsol.t)):
            T = Talpha(Bsol.t[i])
            if T < TB:
                Ytotal.append(Bsol.y[0][i])
                alphatotal.append(Bsol.t[i])
        
        '''Interpolate the full solution'''
        YBLi = interp1d(alphatotal,Ytotal,fill_value = 'extrapolate')
        
        return (YBLi,Ytotal,alphatotal,[NBLei,NBLmui,NBLtaui],NBL.y[0],NBL.t)
                                  




''' ################################################################################################################'''
'                              ''Scattering with gauge bosons and quarks'''
''' ################################################################################################################'''
''' ################################################################################################################'''
''' ################################################################################################################'''



''' ################################################################################################################'''
'                              ''Machinery'''
''' ################################################################################################################'''
def Ny(T):
    return 1.2 * 2 /pi * T**3

    
def Lambda(a,b,c):
    L = (a-b-c)**2 - 4*b*c
    return L

def gammaD(Y,i,z,M):
    '''thermal averaged rate as given in Hambye'''
    YY = np.array(dot(np.matrix(Y).H,Y))
    T = M/z
    Mh,Ml=    MH(T),   Mlep(T,   0)
    aH,aL = (Mh**2/M**2),(Ml**2/M**2) 
    G = GN(YY,T,i,M)
    if M > Mh + Ml:
        return [M**3/(pi**2 * z) * kn(1,z) * G,-1]
    elif Mh > Ml + M:
        return [Mh**2 * M /(pi**2 * z)* kn(1,Mh/T) * G * 2,1]
    else:
        return [0,0]
    
def geq(process,T,M,i,YY,y = False,inf = 10,accuracy = 100,v=1e13):
    '''T is Temp, M is RHN mass
    matrix'''
    '''a,i,j should be the mass func(T) of the particles in Na->ij'''
    pref = T/(64*pi**4)
    eps,mBoson = (Mlep(T)/M[0])**2/10000,mW
    total = 0
    if y == True:
        mBoson = mB
    if process == 'As':
        sigma,m1,m2,m3,m4 = sigmaAs,M[i], Mlep(T), MH(T),mBoson(T)
    elif process == 'AtLH':
        sigma,m1,m2,m3,m4 = sigmaAtLH, Mlep(T), MH(T),M[i],mBoson(T)
    elif process == 'AtLA':
        sigma,m1,m2,m3,m4 = sigmaAtLA, Mlep(T),mBoson(T),M[i], MH(T)
    elif process == 'Hs':
        sigma,m1,m2,m3,m4 = sigmaHs, Mlep(T),M[i],mQ(T),mQ(T)
    elif process == 'Ht':
        sigma,m1,m2,m3,m4 = sigmaHt,mQ(T),M[i],mQ(T), Mlep(T)

    def integrand(s):
        return kn(1,np.sqrt(s)/T)*sigma(s,M[i],T,eps,y)*s**0.5
    
    smin = max((m1+m2)**2,(m3 + m4)**2)
    slist = np.logspace(log10(smin) + 0.0001, log10(smin) + inf,accuracy)
    intlist = [integrand(s) for s in slist]
    integral = simpson(intlist,slist)
    result  = pref * integral * np.real(YY[i][i])
    return result

    
    

''' ################################################################################################################'''
'                              ''Thermal masses  + EWPT'''
''' ################################################################################################################'''
    
g2 = 0.65

ht= 0.95 #https://pdglive.lbl.gov/DataBlock.action?node=S126YTC

HQ = (125/(2*174))**2 #Towards Nardio Strumina eqn 84, approx 0.129

g3 = np.sqrt(4*pi*0.1179) #from wikipedia, at z mass scale, should we run this coupling


gY = 0.35


def v(T):
    mh = 125
    if T > 160:
        return 0
    else:
        return np.sqrt(1-T**2/160**2)*mh



def mQ(T,mexp = 0):
    Mthermal = np.sqrt(1/6*g3**2 + gY**2/288 + 3/32*g2**2 + ht**2/16)*T
    if T > 160:
        return Mthermal
    else:
        return Mthermal + np.sqrt(1-T**2/160**2)*mtop

def mW(T):
    Mthermal = np.sqrt(11/12)*g2*T
    if T > 160:
        return Mthermal
    else:
        return np.sqrt(1-T**2/160**2)*Wmass + Mthermal


def mB(T):
    return np.sqrt(11/12*gY**2*T**2)

def Mlep(T,i=0):
    Mbare =  [me,mmuon,mtau]
    Mthermal = np.sqrt( 3/32 * g2**2 + 1/32 * gY**2)*T
    if T > 160:
        return Mthermal
    else:
        return np.sqrt((1-T**2/160**2)*Mbare[i]**2 + Mthermal**2)


def MH(T,EWPT = True,source = 'Riotto'):
    D = dmh()**2
    Tc = 160
    return np.sqrt( v(T)**2*np.heaviside(Tc-T,1) + D*T**2*(1-Tc**2/T**2)*np.heaviside(-Tc+T,0.))

    
def MHR(T):
    D = dmh()**2
    T_c = 160
    m_h = 125
    temp = m_h**2 * (1-T**2/T_c**2) * np.heaviside(T_c-T,1) +  D * (T**2-T_c**2)* np.heaviside(-T_c+T,0.)
    return np.sqrt(temp)
    
def dmh():
    return np.sqrt(3/16*g2**2 + 1/16*gY**2 + 1/4*ht**2 + 1/2*HQ) 



    


def t(m1,m2,m3,m4,s,pm):
    
    term1 = (m1**2-m2**2-m3**2+m4**2)**2/(4*s)
    
    term2 = (s + m1**2 - m2**2)**2/(4*s) - m1**2 

    term3 =  (s + m3**2 - m4**2)**2/(4*s) - m3**2
    
    return term1 - (np.sqrt(term2) + pm*np.sqrt(term3))**2
    
def tbracket(func,m1,m2,m3,m4,s,mN):
    tplus,tminus = t(m1,m2,m3,m4,s,1),t(m1,m2,m3,m4,s,-1)
    f1,f2 = func(tplus/mN**2) , func(tminus/mN**2)
    
    return f1-f2



''' ################################################################################################################'''
'                                                      ''Cross sections'''
''' ################################################################################################################'''
 

def sigmaAs(s,m,T,eps,y = False,method = 'Riotto',Yplus = 0,Yminus = 0,l=0):
    aQ,aL,aH = (mQ(T)/m)**2,( Mlep(T,l)/m)**2,(MH(T)/m)**2
    if y == True:
        aB = (mB(T)/m)**2
        G = gY / np.sqrt(6)
        mBoson = mB
        nV = 1
    else:
        aB = (mW(T)/m)**2
        G = g2
        mBoson = mW
        nV = 3
    x = s/m**2
    
    if method == 'Riotto':
        pref = 3/(16*pi*x**2)*G**2
        def func(t):
            term = 2*(x*(aL-t)*(aL + aL*x - aB) + eps*(2-2*x+x**2))/((aL-t)**2 + eps)
            return 2*t*(x-2) + (2-2*x+x**2)*log((aL-t)**2 + eps) + term
        sig = pref * tbracket(func,m, Mlep(T,0), MH(T),mBoson(T),s,m)
        
    elif method == 'Underwood':
        YpYp,YmYm = np.array(dot(np.matrix(Yplus).H,Yplus)),np.array(dot(np.matrix(Yminus).H,Yminus))
        pref = nV*G**2/(16*pi*x**2) * (YpYp[0][0] + YmYm[0][0])
        sig = 0
        for ar in (aH,aL):
            '''Include both the s and t channel contributions, with different regulator masses'''
            logArg = (x-1+ar)/ar
            if logArg < 0:
                print('As')
            sig += pref * ((5*x-1)*(1-x) + 2*(x**2 + x - 1)*log(logArg))
    return sig


def sigmaAtLH(s,m,T,eps,y = False,method = 'Riotto',Yplus = 0,Yminus = 0,l=0):
    aH,aL = ( MH(T)/m)**2,( Mlep(T,l)/m)**2
    if y == True:
        aB = (mB(T)/m)**2
        G = gY / np.sqrt(6)
        mBoson = mB
        nV = 1
    else:
        aB = (mW(T)/m)**2
        G = g2
        mBoson = mW
        nV = 3
    x = s/m**2
    if method == 'Riotto':
        prefLH = np.abs(3*G**2/(8*pi*x*(1-x)))
        def funcLH(t):
            term1 = 2*x*log(np.abs(t-aH))
            term2 = (1+x**2)*log(np.abs(t+x-1-aB-aH))
            return term1 - term2
        tb = tbracket(funcLH, m,mBoson(T),Mlep(T,l), MH(T),s,m)
        sigLH = prefLH * tb
    elif method == 'Underwood':
        YpYp,YmYm = np.array(dot(np.matrix(Yplus).H,Yplus)),np.array(dot(np.matrix(Yminus).H,Yminus))
        pref = nV*G**2/(8*pi*x) * (YpYp[0][0] + YmYm[0][0])
        sigLH = 0
        for ar in (aH,aL):
            '''Include both the s and t channel contributions, with different regulator masses'''
            logArg = (x-1+ar)/ar
            if logArg < 0:
                print('LH')
            sigLH += pref * ((x + 1)**2/(x-1+2*ar)*log(logArg))
    return sigLH

def sigmaAtLA(s,m,T,eps,y = False,method = 'Riotto',Yplus = 0,Yminus = 0,l=0):
    aH,aL = ( MH(T)/m)**2,( Mlep(T,l)/m)**2
    if y == True:
        aB = (mB(T)/m)**2
        G = gY / np.sqrt(6)
        mBoson = mB
        nV = 1
    else:
        aB = (mW(T)/m)**2
        G = g2
        mBoson = mW
        nV = 3
    x = s/m**2
    if method == 'Riotto':
        prefLA = 3*G**2/(16*pi*x**2)
        def funcLA(t):
            return t**2 + 2*t*(x-2) - 4*(x-1)*log(np.abs(t-aH)) + x*(aB-4*aH)/(aH-t)
        sigLA = prefLA * tbracket(funcLA, Mlep(T,l),mBoson(T),m, MH(T),s,m)
    elif method == 'Underwood':
        YpYp,YmYm = np.array(dot(np.matrix(Yplus).H,Yplus)),np.array(dot(np.matrix(Yminus).H,Yminus))
        pref = nV*G**2/(16*pi*x**2) * (YpYp[0][0] + YmYm[0][0])
        sigLA = 0
        for ar in (aH,aL):
            '''Include both the s and t channel contributions, with different regulator masses'''
            logArg = (x-1+ar)/ar
            if logArg < 0:
                print('LA')
            sigLA += pref * (x-1)*(x-3+4*log(logArg))
    return sigLA


def sigmaHs(s,m,T,eps,smin,method = 'Riotto',Yplus = 0,Yminus = 0,l=0):
    x = s/m**2
    aH,aL,aQ = ( MH(T)/m)**2,( Mlep(T,l)/m)**2,(mQ(T)/m)**2
    if method == 'Riotto':
        pref = 3/(4*pi)*ht**2
        term1 = (x-1-aL)*(x-2*aQ)/(x*(x-aH)**2)
        t2s1,t2s2 = ((1+aL-x)**2 - 4*aL),(1-4*aQ/x)
        t2s = t2s1*t2s2
        term2 = np.sqrt(t2s)
        sig = pref * term1 * term2
    elif method == 'Underwood':
        YpYp,YmYm = np.array(dot(np.matrix(Yplus).H,Yplus)),np.array(dot(np.matrix(Yminus).H,Yminus))
        au = (0.65**2) * mtop**2/(2*Wmass**2)
        pref = 3*au * (YpYp[0][0] + YmYm[0][0])
        sig = pref * ((x-1)/x)**2
    return sig

def sigmaHt(s,m,T,eps,y,method = 'Riotto',Yplus = 0 ,Yminus = 0,l=0):
    x = s/m**2
    aH,aL,aQ = ( MH(T)/m)**2,( Mlep(T,l)/m)**2,(mQ(T)/m)**2
    if method == 'Riotto':
        pref = 3/(4*pi*x)*ht**2
        def tpm(pm):
            term2 =  np.sqrt((aQ**2 + (x-1)**2 - 2*aQ*(1+x))*(aL**2 + (x-aQ)**2 - 2*aL*(x+aQ)))
            term1 =   aQ + x - (aQ-x)**2 + aL*(x+aQ-1)
            return 1/(2*x) * (term1 + pm*term2)
        tp,tm = tpm(1),tpm(-1)
        term1 = tp -tm - (1-aH+ aL)*(aH-2*aQ)*(1/(aH-tp) - 1/(aH-tm)) - ( 1-2*aH + aL + 2*aQ)*log(np.abs((tp-aH)/(tm-aH)))
        sig = pref*term1
    elif method == 'Underwood':
        YpYp,YmYm = np.array(dot(np.matrix(Yplus).H,Yplus)),np.array(dot(np.matrix(Yminus).H,Yminus))
        au = (0.65**2) * mtop**2/(2*Wmass**2)
        pref = 3*au * (YpYp[0][0] + YmYm[0][0])
        sig = 0
        for ar in (aH,aH):
            logArg = (x-1+ar)/ar
            if logArg < 0:
                print('negative log argument detected in sigmaHt at s/m^2 = ' + str(x))
            F = (1-1/x + 1/x*np.log(logArg))
            sig += pref * (1-1/x + 1/x*np.log(logArg))
    return sig












































'''Underwood and Pilaftsis computation'''
def Aij(Y,i,j):
    '''Function used to calculate Aij for the resummed
    Yukawa matrix'''
    
    Total = 0
    for l in (0,1,2):
        Total += Y[l][i]*np.conjugate(Y[l][j])/(16*pi)
    return Total

def Bli(Y,L,i,M,p):
    '''Function which calculates Bli for resum'''
    
    '''Takes as arguments:
    Yukawa matrix Y
    flavour index L
    RHN index i
    RHN mass matrix M
    RHN momentum p which is taken to be the Hawking temperature
    '''
    
    '''Determine j'''
    if i == 0:
        j=1
    elif i ==1:
        j=0
    
    MNj = M[j]
        
    def f(x):
        return np.sqrt(x)*(1-(1+x)*log(1+ 1/x))
    
    Total = 0
    for l in (0,1,2):
        Total += -np.conjugate(Y[l][i] * Y[l][j] * Y[L][j])/(16*pi)*f(MNj**2/p**2)
    return Total


def resum(Y,pm,M,TBH):
    '''Function which calculates the resummed Yukawa
    matrices Y+- in Underwood Pilaftsis paper.
    
    Takes as arguments the Yukawa matrix Y, and pm which
    should equal 1 or -1'''
    
    '''Initialise the resummed matrix'''
    Yr = [[0,0,0] for i in (0,1,2)]
    
    for l in (0,1,2):
        for i in (0,1,2):
            if i == 1:
                j=0
            elif i ==0:
                j=1
            if i != 2:
                B = Bli(Y,l,i,M,TBH)
                Ai,Aj,Ajj= Aij(Y,i,j),Aij(Y,j,i),Aij(Y,j,j)
                if pm == 1:
                    Yr[l][i] = Y[l][i] + 1j*B - (1j*Y[l][j]*M[i]*(M[i]*Ai + M[j]*Aj))/(M[i]**2-M[j]**2 + 2j*M[i]**2*Ajj)
                elif pm == -1:
                    Yr[l][i] = np.conjugate(Y[l][i]) + 1j*np.conjugate(B) - (1j*np.conjugate(Y[l][j])*M[i]*(M[i]*np.conjugate(Ai) + M[j]*np.conjugate(Aj)))/(M[i]**2-M[j]**2 + 2j*M[i]**2*Ajj)
    return Yr











































'''''''''''''''''''''''''PBH hot spots'''''''''''''''''''''''''''
g = 2 #No of dof for RHNs
ca = g**2/(4*pi)
A = 0.1
NRHN = 2
gH = gs #DOF of hawking radiation same as for plasma??


def profile(Mini,funs,aBH,r,M = None,alpha = None):
    '''returns the temperature of the universe at a radial distance r from the BH with initial mass MPBH 
    at scale factor alpha'''
    ca = 106.75**2/pi
    Gf = 4
    '''extract the cosmological parameters'''
    Ma,Ta = funs
    if alpha == None:
        alpha,aevap = paramBH(Ma,aBH) 
    else:
        aevap = alpha/aBH
    T = Ta(alpha)
    if M == None:
        M = Ma(alpha)

    
    
    '''determine the PBH radius and temperature'''
    rBH,TBH,THini = BHradius(M), TH(M), TH(Mini)
    Ms = Mstar(Mini,T0(Mini))
    '''determine the regime'''
    if M >= 0.9 * Mini:
        regime = 0
    elif M >= Ms and M < Mini * 0.9:
        regime = 1
    elif M < Ms:
        regime = 2
    if regime == 0 or regime == 1:
        '''In this regime the mass of the PBH is approximately constant, and the profile is that of a constant
        core temperature with a drop off by the third power to the plasma temperature below'''
        
        '''determine the critical radius below which the profile is a constant core and max diffusion scale'''
        rcr = rcrit(M,T)
        rdc,rdcini = rdec(M,T),rdec(Mini,T0(Mini))
        
        
        '''determine the constant temperature of the core'''
        coreT = Tcore(M,T)
        
        '''build up the temperature profile'''
        if r <= rcr and r > rBH:
            if coreT > T:
                return coreT #constant temperature core
            else:
                return T
        elif r > rcr:
            '''outer region which extends until the plasma temperature is reached'''
            if r < rdc:
                '''within the radius of diffusion'''
                Tth = coreT * (rcr/r)**(1/3)
                return max(T,Tth)
            elif r < rdcini and r > rdc:
                '''between the radius of diffusion and the initial diffusion radius'''
                Tth =   coreT*(rcr/rdc)**(1/3)* (rdc/r)**(7/11)
                return max(Tth,T)
            elif r >= rdcini:
                '''outside the initial diffusion radius'''
                return T
            else:
                print('Unexpected output')
                return T
        elif r < rBH:
            return TBH #Black hole itself
    elif regime == 2:
        '''In this regime the mass of the PBH has become so small that it can no longer heat the plasma
        T_H is no longer defined, the slope of the temperature profile is either flat inside an expanding
        core or T(r) \propto r^-7/11'''
        
        '''determine the critical radius below which the profile is a constant core and max diffusion scale'''
        M = Ms                                                                 #The PBH hot-spot profile is the 
        rcr = rcrit(M,T)
        rdc,rdcini = rdec(M,T),rdec(Mini,T0(Mini))
        
        
        '''determine the constant temperature of the core'''
        coreT = Tcore(M,T)
        
        '''build up the temperature profile'''
        if r <= rcr and r > rBH:
            if coreT > T:
                return coreT #constant temperature core
            else:
                return T
        elif r > rcr:
            '''outer region which extends until the plasma temperature is reached'''
            if r < rdc:
                '''within the radius of diffusion'''
                Tth = coreT * (rcr/r)**(1/3)
                return max(T,Tth)
            elif r < rdcini and r > rdc:
                '''between the radius of diffusion and the initial diffusion radius'''
                Tth =   coreT*(rcr/rdc)**(1/3)* (rdc/r)**(7/11)
                return max(Tth,T)
            elif r >= rdcini:
                '''outside the initial diffusion radius'''
                return T
            else:
                print('Unexpected output')
                return T
        elif r < rBH:
            return TBH #Black hole itself
        
        
        
def prof(Mini,T,r,M):
    '''returns the temperature of the universe at a radial distance r from the BH with initial mass MPBH 
    at scale factor alpha'''
    ca = 106.75**2/pi
    Gf = 4

    
    '''determine the PBH radius and temperature'''
    rBH,TBH,THini = BHradius(M), TH(M), TH(Mini)
    Ms = Mstar(Mini,T0(Mini))
    '''determine the regime'''
    if M >= 0.9 * Mini:
        regime = 0
    elif M >= Ms and M < Mini * 0.9:
        regime = 1
    elif M < Ms:
        regime = 2
    if regime == 0 or regime == 1:
        '''In this regime the mass of the PBH is approximately constant, and the profile is that of a constant
        core temperature with a drop off by the third power to the plasma temperature below'''
        
        '''determine the critical radius below which the profile is a constant core and max diffusion scale'''
        rcr = rcrit(M,T)
        rdc,rdcini = rdec(M,T),rdec(Mini,T0(Mini))
        
        
        '''determine the constant temperature of the core'''
        coreT = Tcore(M,T)
        
        '''build up the temperature profile'''
        if r <= rcr and r > rBH:
            if coreT > T:
                return coreT #constant temperature core
            else:
                return T
        elif r > rcr:
            '''outer region which extends until the plasma temperature is reached'''
            if r < rdc:
                '''within the radius of diffusion'''
                Tth = coreT * (rcr/r)**(1/3)
                return max(T,Tth)
            elif r < rdcini and r > rdc:
                '''between the radius of diffusion and the initial diffusion radius'''
                Tth =   coreT*(rcr/rdc)**(1/3)* (rdc/r)**(7/11)
                return max(Tth,T)
            elif r >= rdcini:
                '''outside the initial diffusion radius'''
                return T
            else:
                print('Unexpected output')
                return T
        elif r < rBH:
            return TBH #Black hole itself
    elif regime == 2:
        '''In this regime the mass of the PBH has become so small that it can no longer heat the plasma
        T_H is no longer defined, the slope of the temperature profile is either flat inside an expanding
        core or T(r) \propto r^-7/11'''
        
        '''determine the critical radius below which the profile is a constant core and max diffusion scale'''
        M = Ms                                                                 #The PBH hot-spot profile is the 
        rcr = rcrit(M,T)
        rdc,rdcini = rdec(M,T),rdec(Mini,T0(Mini))
        
        
        '''determine the constant temperature of the core'''
        coreT = Tcore(M,T)
        
        '''build up the temperature profile'''
        if r <= rcr and r > rBH:
            if coreT > T:
                return coreT #constant temperature core
            else:
                return T
        elif r > rcr:
            '''outer region which extends until the plasma temperature is reached'''
            if r < rdc:
                '''within the radius of diffusion'''
                Tth = coreT * (rcr/r)**(1/3)
                return max(T,Tth)
            elif r < rdcini and r > rdc:
                '''between the radius of diffusion and the initial diffusion radius'''
                Tth =   coreT*(rcr/rdc)**(1/3)* (rdc/r)**(7/11)
                return max(Tth,T)
            elif r >= rdcini:
                '''outside the initial diffusion radius'''
                return T
            else:
                print('Unexpected output')
                return T
        elif r < rBH:
            return TBH #Black hole itself

def BHradius(M):
    TBH =  TH(M)
    return 1/(4*pi*TBH)

def rcrit(M,T):
    return 6e7 * ( A/0.1)**(-6) * (  gs(T)/106.75) / TH(M)

def THS(Mini):
    '''Function which returns the temperature at which a hot-spot forms
    around a PBH of mass Mini'''
    return 1.96e-4 /(8*pi*GCF*Mini)

def rdec(M,T):
    TBH =  TH(M)
    '''returns the maximum lenth scale of diffusion. See eqn 3.17'''
    '''3.17 uses Mini, this result may be invalid for M < Mini'''
    secToGeV = 1/6.5821e-25
    return 3e-10*secToGeV*(A/0.1)**(-8/5)*(  gs(T)/106.75)**(1/5) *(TBH/1e4)**(-11/5)

def Mstar(M,T):
    return 0.8 /  GeV_in_g * (A/0.1)**(-11/3)*(    gs(T)/106.75)**(2/3)


def Tcore(M,T):
    '''determine the constant temperature of the core'''
    TBH =  TH(M)
    gstar = gs(T)
    Tcore = 2e-4 * (A/0.1)**(8/3) * (  gstar / 106.75)**(-2/3) * (gstar/108)**(2/3) * TBH
    return Tcore




def radiusHS(T,M):
    '''Returns the radius of the hot-spot, such that T(rHS) = T_plasma'''
    coreT,Rcore = Tcore(M,T),rcrit(M,T)
    return Rcore * coreT**3 / T**3

def rsphal(M):
    '''Returns the sphaleron radius of the hot-spot, such that T(rsphal) = Tsphal'''
    coreT,Rcore = Tcore(M,T),rcrit(M,T)
    Tsphal = 130
    return Rcore * coreT**3 / Tsphal**3







































def thermalAverage(m1f,m2f,T1f,T2f,sigma,m34f,inff = 1e10,acc = 10000,y = False,op = False,sigArgs = None):
    '''Function to calculate the thermal average of a given cross section sigma for any 2 to 2 process where
    the initial state particles 1 and 2 have mass m1f and m2f and temperature T1f and T2f all of which may differ
    m34f = m3 + m4'''
    
    
    '''sigma should be a function with argument s, and optional arguments sigArgs'''
    
    '''Define the prefactor'''
    #print(m1f,m2f,T1f,T2f)
    pref = 8*m1f**2*m2f**2*kn(2,m1f/T1f)*kn(2,m2f/T2f)
    #print(pref)
    
    '''Define the integrand'''
    def integrand(s):
        C1 = np.abs(m1f**2*T2f**2 - m2f**2*T1f**2)
        C2 = m1f**4 + (m2f**2 - s)**2 - 2*m1f**2*(m2f**2 + s)
        D = s*T1f*T2f + (T2f-T1f)*(m1f**2*T2f - m2f**2*T1f)
        xmin = np.sqrt(D)/(T1f*T2f)
        F = 0.5*np.sqrt( (s-m1f**2-m2f**2)**2 - m1f**2*m2f**2 )
        term1 = C1/D * np.exp(-xmin) * (1+xmin)
        term2 = np.sqrt(C2/D) * kn(1,xmin)
        if sigArgs == None:
            sig = sigma(s,m1f,m2f,y=y)
        else:
            sig = sigma(s,m1f,m2f,Args = sigArgs)
        I =  sig * F * (term1 + term2)
        if I.imag == 0*1j:
            I = I.real
        elif I.imag != 0*1j:
            print('Imaginary integrand detected, discarding imaginary part')
            I = I.real
        if isnan(I) == True:
            print('Nan detected' + str(sig) + str(sigma))
            return 0
        else:
            return I
    '''make the lists'''
    smin = max((m1f+m2f)**2,(m34f)**2)
    smax = smin*inff
    slist = np.logspace(log10(smin*1.01),log10(smax),acc)
    intlist = [integrand(s) for s in slist]

    '''Integrate'''
    Sigma = simpson(intlist,slist)
    
    if pref != 0:
        return Sigma/pref
    else: 
        return 0

def Aij(Y,i,j):
    '''Function used to calculate Aij for the resummed
    Yukawa matrix'''
    
    Total = 0
    for l in (0,1,2):
        Total += Y[l][i]*np.conjugate(Y[l][j])/(16*pi)
    return Total

def Bli(Y,L,i,M,p):
    '''Function which calculates Bli for resum'''
    
    '''Takes as arguments:
    Yukawa matrix Y
    flavour index L
    RHN index i
    RHN mass matrix M
    RHN momentum p which is taken to be the Hawking temperature
    '''
    
    '''Determine j'''
    if i == 0:
        j=1
    elif i ==1:
        j=0
    
    MNj = M[j]
        
    def f(x):
        return np.sqrt(x)*(1-(1+x)*log(1+ 1/x))
    
    Total = 0
    for l in (0,1,2):
        Total += -np.conjugate(Y[l][i] * Y[l][j] * Y[L][j])/(16*pi)*f(MNj**2/p**2)
    return Total


def resum(Y,pm,M,TBH):
    '''Function which calculates the resummed Yukawa
    matrices Y+- in Underwood Pilaftsis paper.
    
    Takes as arguments the Yukawa matrix Y, and pm which
    should equal 1 or -1'''
    
    '''Initialise the resummed matrix'''
    Yr = [[0,0,0] for i in (0,1,2)]
    
    for l in (0,1,2):
        for i in (0,1,2):
            if i == 1:
                j=0
            elif i ==0:
                j=1
            if i != 2:
                B = Bli(Y,l,i,M,TBH)
                Ai,Aj,Ajj= Aij(Y,i,j),Aij(Y,j,i),Aij(Y,j,j)
                if pm == 1:
                    Yr[l][i] = Y[l][i] + 1j*B - (1j*Y[l][j]*M[i]*(M[i]*Ai + M[j]*Aj))/(M[i]**2-M[j]**2 + 2j*M[i]**2*Ajj)
                elif pm == -1:
                    Yr[l][i] = np.conjugate(Y[l][i]) + 1j*np.conjugate(B) - (1j*np.conjugate(Y[l][j])*M[i]*(M[i]*np.conjugate(Ai) + M[j]*np.conjugate(Aj)))/(M[i]**2-M[j]**2 + 2j*M[i]**2*Ajj)
    return Yr
       

def paramBH(Ma,aBH,a0=13,a1=19,acc = 10000):
    '''Calculate the BH parameters'''
    alphalist = np.linspace(a0,a1,acc)
    for i in range(0,np.size(alphalist)):
        '''find the point of evaporation'''
        if Ma(alphalist[i]) == 0:
            alphaevap = alphalist[i-1]
            break
    alpha = aBH * alphaevap
    return alpha,alphaevap

def GammaTT(Y,T,M,mBH,i=0,accT = 10000,output = 0,method = 'Riotto',inff = 1e10,comoving = [False,None]):
    '''Function which calculates the total scattering rate for N_i'''
    
    '''Calculate temp of RHNs,TN, and plasma,T'''
    Y = np.array(Y)
    TN =   TH(mBH)
    Yplus,Yminus =   resum(Y,1,M,TN),  resum(Y,-1,M,TN)
    YY = np.array(dot(np.matrix(Y).H,Y))
    
    '''Check that the RHNs can actually be produced by the PBHs'''
    if TN < M[i]:
        print('N_i cannot be produced by the BH because the Hawking temperature is too small')
        return [0,0]
        
    '''Set up the lists'''
    diagrams = [  sigmaAs,  sigmaAtLH,  sigmaAtLA,  sigmaHs,  sigmaHt]
    
    gammaS = []
    
    '''Iterate over the diagrams'''
    for j in range(0,np.size(diagrams)):
        
        
        if T < TEWPT:
            '''In the broken phase Na is not pre calculated'''
            '''If  we are then we may need the lepton flavour projectors'''
            Pli = [[np.abs(Y[l][i])**2/YY[i][i] for i in range(0,3)] for l in range(0,3)]
            '''The leptons have to be summed over individually'''
            alist = [  [Mlep(T,0),Mlep(T,1),Mlep(T,2)],  mB(T),  MH(T),  [Mlep(T,0),Mlep(T,1),Mlep(T,2)],  mQ(T)]
            
        else:
            '''All particles off which N scatters are relativistic'''
            alist = [  Mlep(T,0),  mB(T),  MH(T),  Mlep(T),  mQ(T)]
            if comoving[0] == False:
                Na = 0.75 * 2 * 1.2 * T ** 3 / pi**2
            else:
                Na = 0.75 * 2 * 1.2 * T ** 3 / pi**2 * 10**(3*comoving[1])
            
        graph = diagrams[j]

        
        '''Function to transform the reduced cross section into the cross section as in Riotto Strumina'''
        def sigT(s,m,ma,y = False):
            if j <= 2:
                sig1,sig2 = graph(s,m,T,1e-10,y = True,method = method,Yplus = Yplus,Yminus = Yminus),graph(s,m,T,1e-10,y = False,method = method,Yplus = Yplus,Yminus = Yminus)
                fac1,fac2 = (2*s*  Lambda(1,m**2/s,ma**2/s)),(2*s*  Lambda(1,m**2/s,ma**2/s))
                if y == True:
                    return sig1/fac1 
                else:
                    return sig2/fac2
            elif j == 3:
                return graph(s,m,T,1e-10,smin =None,method = method,Yplus = Yplus,Yminus = Yminus) / (2*s*  Lambda(1,m**2/s,ma**2/s))
            elif j == 4:
                return graph(s,m,T,1e-10,y = 0,method = method,Yplus = Yplus,Yminus = Yminus) / (2*s*  Lambda(1,m**2/s,ma**2/s))
        
        '''Define the thermal average where the RHNs are particle 1 in the process Na->34, so local plasma
        temperature is T2, and RHNs have temperature TN'''
        if j == 0:
            m341,m342 =   MH(T)+  mB(T),  MH(T)+  mW(T)
            a = alist[j]
            if T < TEWPT :
                '''At temperatures below the EWPT the lepton diagrams must be summed individually'''
                gamma = 0
                for l in range(0,3):
                    al = a[l]
                    if T < al/10:
                        pass
                    else:
                        ta1 = thermalAverage(M[i],al,TN,T,sigT,m341,inff = inff,y = True,acc = accT)
                        ta2 = thermalAverage(M[i],al,TN,T,sigT,m342,inff = inff,y = False,acc = accT)
                        gamma  += (ta1+ta2)*Pli[l][i]*Neqsa(T,al)
                        if output > 1:
                            print(r'$\sigma(Nl_$' + str(l) + r'$\to HA) = $' + str(ta1 + ta2))
            else:
                '''At T > TEWPT the leptons have identical abundances and can be summed over'''
                ta1 = thermalAverage(M[i],a,TN,T,sigT,m341,inff = inff,y = True,acc = accT)
                ta2 = thermalAverage(M[i],a,TN,T,sigT,m342,inff = inff,y = False,acc = accT)
                gamma = (ta1+ta2)*Na
                if output > 1:
                    print(r'$\sigma(NL\to HA)$ = ' + str(ta1 + ta2))
        
            
    
            gammaS.append(gamma)
            
        elif j == 1:
            m34 =   Mlep(T) +   MH(T)
            a1,a2 =   mB(T),  mW(T)
            if T < TEWPT:
                ta = thermalAverage(M[i],a1,TN,T,sigT,m34,inff = inff,y = True,acc = accT)
                Ny = 0.75 * 2 * 1.2 * T ** 3 / pi**2
                gamma = ta * Ny
                if output > 1:
                    print(r'$\sigma(NA\to LH)$ = ' + str(ta))
            else:
                ta1 = thermalAverage(M[i],a1,TN,T,sigT,m34,inff = inff,y = True,acc = accT)
                ta2 = thermalAverage(M[i],a2,TN,T,sigT,m34,inff = inff,y = False,acc = accT)
                if output > 1:
                    print(r'$\sigma(NA\to LH)$ = ' + str(ta1 + ta2))
                gamma = (ta1+ta2)*Na
            gammaS.append(gamma)
            
        elif j == 2:
            m341,m342 =   Mlep(T)+  mB(T),  Mlep(T)+  mW(T)
            a = alist[j]
            if T < TEWPT:
                gamma = 0
                if T < a/10:
                    pass
                else:   
                    ta1 = thermalAverage(M[i],a,TN,T,sigT,m341,inff = inff,y = True,acc = accT)
                    ta2 = thermalAverage(M[i],a,TN,T,sigT,m342,inff = inff,y = False,acc = accT)
                    if output > 1:
                        print(r'$\sigma(NH \to LA)$ = ' + str(ta1 + ta2))
                    
                    gamma += (ta1+ta2)*Neqsa(T,a)
            else:    
                ta1 = thermalAverage(M[i],a,TN,T,sigT,m341,inff = inff,y = True,acc = accT)
                ta2 = thermalAverage(M[i],a,TN,T,sigT,m342,inff = inff,y = False,acc = accT)
                if output > 1:
                    print(r'$\sigma(NH\to LA)$ = ' + str(ta1 + ta2))
                gamma = (ta1+ta2)*Na
            gammaS.append(gamma)
        
        elif j == 3:
            m34 = 2 *   mQ(T)
            a = alist[j]
            if T < TEWPT:
                gamma = 0
                for l in range(0,3):
                    al = a[l]
                    if T < al/10:
                        pass
                    else:
                        ta = thermalAverage(M[i],al,TN,T,sigT,m34,inff = inff,acc = accT)
                        gamma += ta * Pli[l][i] * Neqsa(T,al)
                        if output > 1:
                            print(r'$\sigma(N\ell_$' + str(l) + r'$L\to QU)$ = ' + str(ta))
            else:
                ta = thermalAverage(M[i],a,TN,T,sigT,m34,inff = inff,acc = accT)
                gamma = ta*Na
                if output > 1:
                    print(r'$\sigma(NL\to QU)$ = ' + str(ta))
            gammaS.append(gamma)
        elif j == 4:
            m34 =   Mlep(T) +   mQ(T)
            a = alist[j]
            gamma = 0
            if T < TEWPT:
                if T < a/10:
                    pass
                else:
                    ta = thermalAverage(M[i],a,TN,T,sigT,m34,inff = inff,acc = accT)
                    gamma += ta*Neqsa(T,a)
            else:
                ta = thermalAverage(M[i],a,TN,T,sigT,m34,inff = inff,acc = accT)
                gamma += ta*Na
            if output > 1:
                print(r'$\sigma(NQ\to QL)$ = ' + str(ta))
            gammaS.append(gamma)

    return sum(gammaS),gammaS


def ls(Mf,Yf,Mini,funcs,aBH,i=0,inff = 1e10,method = 'Riotto',output = 1,accT = 10000,accR = 100,full = False):
    '''Returns the scatterling length for any RHN N_i described by mass matrix M and Yukawa matrix Y produced by a PBH 
    at alpha = alpha_evap * aBH.
        
    Assumptions:
    - '2-neutrino' approximation of R matrix
    - Temperature profile according to argument 'prof'
    '''
    
    '''Extract the cosmological functions'''
    Ma,rada,pbha,Ta,sa,aT = funcs[0],funcs[1],funcs[2],funcs[3],funcs[4],funcs[5]

    alpha,mBH = paramBH(Ma,aBH)
    T = Ta(alpha)
    
    '''Build the BH temperature profile'''
    rcore = rcrit(mBH,T)
    Tcr = profile(Mini,funcs,aBH,rcore/2,M = None,alpha = alpha)
    rHS = radiusHS(T,mBH)
    
    '''Now we must create the function Gamma(r) which returns the total scattering rate
    In order to do so we must calculate the total scattering rate of in the core, the background universe
    and if necessary the cooling part of the profile.
    
    It will prove useful to first estimate the scattering length as lambdaCore = 1/Gamma(Tcore) and 
    lambdaBack = 1/Gamma(T_plasma)'''
    Gcore = GammaTT(Tcr,mBH,Yf,Mf,i=i,accT = accT,output = output,method = method,inff = inff)
    lambdaCore = 1/Gcore[0] #Estimate of the scattering length assuming T = Tcr everywhere, underestimate
    Gback = GammaTT(T,mBH,Yf,Mf,i=i,accT = accT,output = output,method = method,inff = inff)
    lambdaBack = 1/Gback[0] #Estimate of scattering length assuming that T  = Tplasma, overestimate
    if full == True:
        return [Gcore[1],Gback[1]]
    
    '''We are now left with three possible options'''
    
    if lambdaCore < rcore:
        '''If the estimate lambdaCore does not exceed Rcore, it is accurate and should be returned'''
        if output == 1:
            print('Core only')
        return lambdaCore
    else:
        '''If lambdaCore > Rcore then we must determine which regime the calculation lies in'''
        rlistOP = np.logspace(log10(rcore),log10(rHS),accR) #list of r values in the outer profile
        Glist,rtemp = [],[]
        for r in rlistOP:
            rtemp.append(r)
            Ttemp = profile(Mini,funcs,aBH,r,M = None,alpha = alpha)
            G = GammaTT(Ttemp,mBH,Yf,Mf,i=i,accT = accT,output = output,inff = inff,method = method)[0]
            Glist.append(G)
        GintOP = simpson(Glist,rlistOP) #This is the scattering rate integrated over the outer profile
        GHSI = GintOP + Gcore[0]*rcore #This is the total integrated scattering rate inside the hotspot

        if log(2) > GHSI:
            case = 1
        else:
            case = 0
            
    if case == 1:
        '''In this case the true scattering length lies outside the hotspot entirely'''
        if output == 1:
            print('Case 1')
        return (log(2) - GHSI )/Gback[0] + rHS
    elif case == 0:
        if output == 1:
            print('Case 0')
        '''In this case the scattering length lies inside the outer profile of the hot-spot'''
        Gint = interp1d(rlistOP,Glist,fill_value = 'extrapolate')
        Gilist = []
        for R in rlistOP:
            tempr = np.logspace(log10(rcore),log10(R),accR)
            Gtemp = [Gint(R) for R in tempr]
            Gilist.append(simpson(Gtemp,tempr))
        Giint = interp1d(Gilist,rlistOP,fill_value = 'extrapolate')
        return Giint(log(2) - rcore * Gcore[0])
        
def kinematics(MNi):
    '''Function which takes a mass MNi and returns the temperature(s)
    at which it equals the combined Higgs and lepton thermal masses'''
    D,Dl =   dmh()**2, 3/32 *   g2**2 + 1/32 *   gY**2
    T1 = (-np.sqrt(Dl)*MNi + np.sqrt(D*MNi**2 + D**2*TEWPT**2 - D*Dl*TEWPT**2))/(D-Dl)
    return T1

def lD(Mf,Yf,Mini,funcs,aBH,i=0,output = 1,alphaevap = None):
    '''returns the decay length for RHNs with mass matrix M and Yukawa matrix Y (2 neutrino case),
    Mini is the initial mass of the PBH, funs track the cosmological evolution of the universe.
    aBH is defined by aBH  = a/alphaevap where the PBHs evaporate completely by alphaevap'''
    
    '''Extract the cosmological functions'''
    Ma,rada,pbha,Ta,sa,aT = funcs[0],funcs[1],funcs[2],funcs[3],funcs[4],funcs[5]
    
    '''Calculate the BH parameters'''
    if alphaevap == None:
        alphalist = np.linspace(13,19,1000)
        for a in alphalist:
            '''find the point of evaporation'''
            if Ma(a) == 0:
                alphaevap = a
                break
    alpha = aBH * alphaevap
    T = Ta(alpha)

    mBH  = Ma(alpha)
    rBH,TBH = BHradius(mBH),  TH(mBH)
    zBH = Mf[i]/TBH
    av = kn(1,zBH)/kn(2,zBH)
    
    YY = np.array(dot(np.matrix(Yf).H,Yf))
    
    '''Check that the RHNs can actually be produced by the PBHs'''
    if TBH < Mf[i]:
        print('N_i cannot be produced by the BH because the Hawking temperature is too small')
        return 0
    
    '''Build the BH temperature profile and find the core temperature and radius'''
    Rcore = rcrit(mBH,T)
    coreT = profile(Mini,funcs,aBH,Rcore,M = None,alpha = alpha)
    Tdecay = kinematics(Mf[i])*0.9999
    
    '''Next check if the RHN decay is kinematically possible in the core'''
    if Tdecay < coreT:
        '''The decay is kinematically impossible inside the core'''
        if Tdecay < T:
            '''The decay is not kinematically possible at all while the background is at this temperature'''
            return 1e50
        else:
            '''The RHN may decay but not inside the core'''
            rstart = radiusHS(Tdecay,mBH)
            G = float(np.abs(  GN(YY,T,i,Mf[i]) * av))
            return 1/G + rstart
    else:
        return 1/float( GN(YY,T,i,Mf[i]) * av)
    
def lnP(R,ys,params):
    '''Function which returns the derivative of the logarithmic probability lnP_i that N_i will reach radius r'''

    '''First extract the parameters'''
    Mf,Yf,i_f,funcsf,aBHf,alpha,Minif,mBHf,GTTback,GNback,Q,epsback = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9],params[11],params[12]
    TBH =   TH(mBHf)
    P = np.exp(ys[0])

    '''Then extract the cosmological parameters'''
    Ma,rada,pbha,Ta,sa = funcsf[0],funcsf[1],funcsf[2],funcsf[3],funcsf[4]
    T = Ta(alpha)

    '''Find the local plasma temperature'''
    r = 10**(R)
    Tr =   profile(Minif,funcsf,aBHf,r,alpha = alpha) #local temperature of the SM plasma

    if Tr == T:
        '''We are in the background'''
        GNi = GNback
        sig = GTTback
        eps = epsback
    else:
        sig =   GammaTT(Y = Yf,T = Tr,M = Mf,i = i_f,mBH = mBHf,method = 'Underwood',inff = 1e10)[0]
        zBH = Mf[i_f]/TBH
        GNi =   GN(Y = Yf,T = Tr,M = Mf,i = i_f) * ME(zBH)
            
        if Q == 'eps':
            '''Calculate the value of epsilon at r'''
            eps = [epsilon(Mf,Yf,i_f,l,Tr,Mvgen(0.051e-9)) for l in [0,1,2]]
            
    Prate = - log(10) * r * (GNi + sig)
    if Q == 'S':
        return [Prate,log(10) * r*sig*P]
    elif Q == 'D':
        return [Prate,log(10) * r*GNi*P]
    elif Q == 'SD' or Q == 'DS':
        return [Prate, log(10) * r*GNi*P,r*sig*P]
    elif Q == 'eps':
        return [Prate, *[log(10) * r * e*P for e in eps]]
    
    

def Pcut(lnr,ys,params):
    pmin = params[10]
    P = np.exp(ys[0])
    return P - pmin



    
    
def HSavg(Q,aBHf,funcsf,Minif,params,debug = False,pmin = 1e-3,method = 'BDF',racc = None,m_h = 0.051e-9):
    '''Function which averages a quantity Q = 'D','S','DS','eps' over the instantaneous profile of a hot-spot for initial PBH mass Mini
    funcsf are the cosmological functions describing the evolution of the universe and the PBH
    aBHf is defined as alpha/alpha_evap for alpha_evap the value of alpha when the PBHs evaporate completely
    Minif is the initial mass of the PBHs
    
    pmin is the minimum probability considered in the integral
    racc is the number of points in r considered, if None they are chosen by the solver
    params contains the necessary parameters for the particle physics model, for vMSM it is [Y,MN,i]. Y is the Yukawa matrix, MN is the RHN mass matrix 
    i_f is the index of the RHN N_i
    when debug is True additional printouts are made
    '''
    
    '''First extract the cosmological variables'''
    Ma,rada,pbha,Ta,sa = funcsf[0],funcsf[1],funcsf[2],funcsf[3],funcsf[4]
    alpha,alphaevap =   paramBH(Ma,aBHf) #instantaneous value of alpha, and the alpha when PBHs evaporate
    alpha,alphaevap =   paramBH(Ma,aBHf,a0 = alphaevap-1,a1 = alphaevap+1) #instantaneous value of alpha, and the alpha when PBHs evaporate
    mBH,T = Ma(alpha),Ta(alpha) #instantaneous mass of the PBH and background temperature
    rBH,TBH = BHradius(mBH),  TH(mBH) #radius and Hawking temperature of the BH
    if debug == True:
        print('Black Hole radius is ' + str(rBH))

    '''Extract the particle physics parameters'''
    Yf,Mf,i_f = params[0],params[1],params[2]

    '''Next calculate the hot-spot parameters'''
    rcore,Tcr = rcrit(mBH,T),Tcore(mBH,T) #radius and temperature of the hot-spot core

    '''Calculate the relevant quantities in the core'''
    GammaTTcore =   GammaTT(Y = Yf,T = Tcr,M = Mf,i = i_f,mBH = mBH,method = 'Underwood',inff = 1e10)[0] #scattering rate at T = Tcr
    if debug == True:
        print('Core scattering rate is' + str(GammaTTcore))
    zBH = Mf[i_f]/TBH
    GNcore =   GN(Y = Yf,T = Tcr,M = Mf,i = i_f) * ME(zBH) #Decay rate at T=Tcr
    dM = Mf[1]- Mf[2]
    epsCore = [epsilon(Mf,Yf,i_f,l,Tcr,Mvgen(m_h)) for l in [0,1,2]]
    
    
    
    
    lnPcore = (- (GammaTTcore + GNcore) * (rcore - rBH)).real #see eqn 121
    Pcore = np.exp(lnPcore)
    if debug == True:
        print('probability to escape the core is ' + str(Pcore))
            
    '''Depending on the quantity being averaged, check if the average should be cut at the core, or set the initial conditions for differential equation solver'''
    if Q == 'S':
        if Pcore <= pmin:
            return GammaTTcore 
        else:
            y0 = [lnPcore,GammaTTcore * Pcore]
    elif Q == 'D':
        if Pcore <= pmin:
            return GNcore
        else:
            y0 = [lnPcore,GNcore * Pcore]
    elif Q == 'SD' or Q == 'DS':
        if Pcore <= pmin:
            return [GNcore,GammaTTcore]
        else:
            y0 = [lnPcore,GNcore * Pcore, GammaTTcore * Pcore]
    elif Q == 'eps':
        if Pcore <= pmin:
            return epsCore
        else:
            y0 = [lnPcore,*[i*Pcore for i in epsCore]]


    '''Scattering and decay rates in the background are calculated, used to over-estimate the mean free path, see Eqn 124'''
    GammaTTback =   GammaTT(Y = Yf,T = T,M = Mf,i = i_f,mBH = mBH,method = 'Underwood',inff = 1e10)[0]
    GNback =   GN(Y = Yf,T = T,M = Mf,i = i_f) * ME(zBH)
    epsback = [epsilon(Mf,Yf,i_f,l,T,Mvgen(m_h)) for l in [0,1,2]]

    mfp = (1/GammaTTback) #overestimate of the scattering length

    if debug == True:
        print('Overestimate of the mean free path is ' + str(mfp))
        print('Background decay rate is ' + str(GNback))
        print('Background scattering rate is ' + str(GammaTTback))
        print('Background temperature is ' + str(T))
    

    '''Now we can solve the coupled differential equations'''
    Pcut.terminal = True
    if racc != None:
        R_eval  = np.linspace(log10(rcore),log10(mfp*10),int(racc)) #I evaluate the integral between the core radius and 10 times the mean free path overestimated above
    else:
        R_eval = None
        
    lnPr = solve_ivp(fun = lnP,t_span = (log10(rcore),log10(mfp*10)),y0 = y0,events = [Pcut],
                     args = [[Mf,Yf,i_f,funcsf,aBHf,alpha,Minif,mBH,GammaTTback,GNback,pmin,Q,epsback]],method = method,t_eval = R_eval)
    
    '''Extract the solutions'''
    Plist,rlist = np.exp(lnPr.y[0]),10**(lnPr.t)
    
    if debug == True:
        print('Number of points returned is ' + str(len(rlist)))
        
    '''Integrate the solution for P to find the denominator for the hot-spot average'''
    Pint = simpson(Plist,rlist)
    
    '''Finally, I evaluate the numerator of the average and return it along with the background rates'''
    if Q == 'D' or Q == 'S':
        numerator = lnPr.y[1][-1]
        result = [numerator/Pint,GNback,GammaTTback]
        return result
    elif Q == 'DS' or Q == 'SD':
        Dnumerator,Snumerator = lnPr.y[1][-1],lnPr.y[2][-1]
        return [Dnumerator/Pint,Snumerator/Pint,GNback,GammaTTback]
    elif Q == 'eps':
        averaged = [lnPr.y[k][-1]/Pint for k in [1,2,3]]
        return [*averaged,*epsback]
    
    
    
    
    
    
    
    
def lsgeneral(Mf,Yf,Mini,funcs,aBH,i,Args= None,particle = 'RHN',a0=10,a1 = 20,accR = 100,output = 0):
    '''Returns the scattering length for a generalised scattering interaction 
    coupling matrix Yf
    mass matrix Mf
    particle'''

    
    '''Extract the cosmological functions'''
    Ma,rada,pbha,Ta,sa,aT = funcs[0],funcs[1],funcs[2],funcs[3],funcs[4],funcs[5]

    alpha,alphaevap = paramBH(Ma,aBH,a0=a0,a1=a1)
    mBH = Ma(alpha)
    T = Ta(alpha)
    
    
    '''Build the BH temperature profile'''
    rcore = rcrit(mBH,T)
    Tcr = profile(Mini,funcs,aBH,rcore/2,M = mBH,alpha = alpha)
    rHS = radiusHS(T,mBH)
    
    def Gamma(MG,YG,iG,mBHG,TG,args = Args):
        '''This generalised, local function should return the scattering rate sigma * N for the particle
        where all diagrams are summed over
        TG is the temperature of the local background'''
        if particle == 'RHN':
            return GammaTT(YG,TG,MG,mBHG,i=iG,accT = args[0],output = args[1],method = args[2],inff = args[3])[0]

        elif particle == 'DMz':
            mZp,gSM = args[0],args[1]

            mSM = 0.1*TG #approximate thermal mass of the SM bath
            mDMT = DMz.thermalMass(TG,'DM',YG[iG],gSM)
            mZpT = DMz.thermalMass(TG,'DM',YG[iG],gSM)
            
            mDM = MG[iG] + mDMT
            args[0] = mZp + mZpT
            
            
            tchannel = thermalAverage(mDM,TG,TH(mBHG),mSM,DMz.sigmat,mDM + mSM,sigArgs = args) * Neqrel(TG)
            '''In the above, if T < TEW it may not make sense to take Neqrel unless photons can scatter and this is dominant'''
            schannel =  thermalAverage(mDM,mDM,TH(mBHG),TG,DMz.sigmas,2*mSM,sigArgs = args) * Neqsa(TG,MG[iG])
            '''For the s channel I am assuming that the DM has an equilibrium abundance always'''
            
            return tchannel + schannel

        
        
    
    '''It will prove useful to first estimate the scattering length as lambdaCore = 1/Gamma(Tcore) and 
    lambdaBack = 1/Gamma(T_plasma)'''
    Gcore = Gamma(Mf,Yf,i,mBH,Tcr,args = Args)
    lambdaCore = 1/Gcore #Estimate of the scattering length assuming T = Tcr everywhere, underestimate
    Gback = Gamma(Mf,Yf,i,mBH,T,args = Args)
    lambdaBack = 1/Gback #Estimate of scattering length assuming that T  = Tplasma, overestimate
    
    '''We are now left with three possible options'''
    
    if lambdaCore < rcore:
        '''If the estimate lambdaCore does not exceed Rcore, it is accurate and should be returned'''
        if output == 1:
            print('Core only')
        return lambdaCore
    else:
        GHSI = GammaHSI(Mf,Yf,Mini,funcs,aBH,i,Gamma,Args= Args)
          
        if log(2) > GHSI:
            case = 1
        else:
            case = 0
            
    if case == 1:
        '''In this case the true scattering length lies outside the hotspot entirely'''
        if output == 1:
            print('Case 1')
        return (log(2) - GHSI )/Gback + rHS
    elif case == 0:
        if output == 1:
            print('Case 0')
        '''In this case the scattering length lies inside the outer profile of the hot-spot'''
        Gint = interp1d(rlistOP,Glist,fill_value = 'extrapolate')
        Gilist = []
        for R in rlistOP:
            tempr = np.logspace(log10(rcore),log10(R),accR)
            Gtemp = [Gint(R) for R in tempr]
            Gilist.append(simpson(Gtemp,tempr))
        Giint = interp1d(Gilist,rlistOP,fill_value = 'extrapolate')
        return Giint(log(2) - rcore * Gcore)    
    
    

def GammaHSI(Mf,Yf,Mini,funcs,aBH,i,Gamma,Args= None,a0=10,a1 = 20,accR = 100,output = 0,tol = 1e-6,Tcrit = None,alpha = None):
    '''Returns the integrated rate over the profile of a hot-spot at alpha = aBH * alpha_evap'''
    '''Gamma is a function with arguments (Mf,Yf,i,mBH,Ttemp,args = Args)'''
    '''tol is the numerical tolerance for finding the hot-spot radius such that r = rHS if T(r) < Tback * (1+tol)'''
    
    '''Extract the cosmological functions'''
    Ma,rada,pbha,Ta,sa,aT = funcs[0],funcs[1],funcs[2],funcs[3],funcs[4],funcs[5]

    mBH = Ma(alpha)
    Tback = Ta(alpha)
    
    
    '''Build the BH temperature profile'''
    rcore = rcrit(mBH,Tback)
    Tcr = profile(Mini,funcs,aBH,rcore/2,M = mBH,alpha = alpha)
    if Tcrit == None:
        '''if critical temperature undefined, assume background temperature'''
        Tcrit = Tback
    
    if Tcrit > Tcr:
        '''The critical temperature is above the core temperature, probability of escape = 1'''
        print('Tcore = ' + str(Tcore) + ' and Tcrit = '  + str(Tcrit))
        return 0
    elif Tcrit < Tback:
        '''The critical temperature is below the background temperature, probability of escape = 0'''
        print('Whole universe above Tcrit')
        return np.infty
    elif Tcrit > Tback and Tcrit < Tcr:
        pass
        
        
    rHS = radiusHS(Tcrit,mBH)
        
    '''In the end stages of the evaporation, the profile may not be analytically found'''
    if aBH > 0.9:
        rlist = np.logspace(log10(rcore),log10(rHS) + 2,accR*10)
        for r in rlist:
            Tr = profile(Mini,funcs,aBH,r,M = mBH,alpha = alpha)
            if Tr <= Tcrit*(1 + tol):
                rHS = r
                break
        if rHS == radiusHS(Tback,mBH):
            print('rHS not found numerically, defualt value assumes r^{-1/3} profile')

    
    Gcore = Gamma(Mf,Yf,i,mBH,Tcr,args = Args)

    rlistOP = np.logspace(log10(rcore),log10(rHS),accR) #list of r values in the outer profile
    Glist,rtemp = [],[]
    for r in rlistOP:
        rtemp.append(r)
        Ttemp = profile(Mini,funcs,aBH,r,M = mBH,alpha = alpha)
        
        G = Gamma(Mf,Yf,i,mBH,Ttemp,args = Args)

        Glist.append(G)
    GintOP = simpson(Glist,rlistOP) #This is the scattering rate integrated over the outer profile
    GHSI = GintOP + Gcore*rcore #This is the total integrated scattering rate inside the hotspot
    return GHSI


def GammaHSI2(Mf,Yf,Mini,mBH,Tback,aBH,i,Gamma,Args= None,a0=10,a1 = 20,accR = 100,output = 0,tol = 1e-6,Tcrit = None,alpha = None):
    '''Returns the integrated rate over the profile of a hot-spot at alpha = aBH * alpha_evap'''
    '''Gamma is a function with arguments (Mf,Yf,i,mBH,Ttemp,args = Args)'''
    '''tol is the numerical tolerance for finding the hot-spot radius such that r = rHS if T(r) < Tback * (1+tol)'''

    '''Build the BH temperature profile'''
    rcore = rcrit(mBH,Tback)
    Tcr = Tcore(mBH,Tback)
    if Tcrit == None:
        '''if critical temperature undefined, assume background temperature'''
        Tcrit = Tback
    
    if Tcrit > Tcr:
        '''The critical temperature is above the core temperature, probability of escape = 1'''
        print('Tcore = ' + str(Tcore) + ' and Tcrit = '  + str(Tcrit))
        return 0
    elif Tcrit < Tback:
        '''The critical temperature is below the background temperature, probability of escape = 0'''
        print('Whole universe above Tcrit')
        return np.infty
    elif Tcrit > Tback and Tcrit < Tcr:
        pass
        
        
    rHS = radiusHS(Tcrit,mBH)
        
    '''In the end stages of the evaporation, the profile may not be analytically found'''
    if aBH > 0.9:
        rlist = np.logspace(log10(rcore),log10(rHS) + 2,accR*10)
        for r in rlist:
            Tr = prof(Mini,Tback,r,mBH)
            if Tr <= Tcrit*(1 + tol):
                rHS = r
                break
        if rHS == radiusHS(Tback,mBH):
            print('rHS not found numerically, defualt value assumes r^{-1/3} profile')

    
    Gcore = Gamma(Mf,Yf,i,mBH,Tcr,args = Args)

    rlistOP = np.logspace(log10(rcore),log10(rHS),accR) #list of r values in the outer profile
    Glist,rtemp = [],[]
    for r in rlistOP:
        
        rtemp.append(r)
        Ttemp = prof(Mini,Tback,r,mBH)
        
        if Ttemp < Tcrit*(1 + tol):
            Glist.append(0)
            
        else:
            G = Gamma(Mf,Yf,i,mBH,Ttemp,args = Args)

            Glist.append(G)
        

    GintOP = simpson(Glist,rlistOP) #This is the scattering rate integrated over the outer profile
    GHSI = GintOP + Gcore*rcore #This is the total integrated scattering rate inside the hotspot
    return GHSI

    
    
    
    
    
    
def mfpgeneral(Mf,Yf,Mini,funcs,aBH,i,Args= None,particle = 'RHN',a0=10,a1 = 20,accR = 100,output = 0):
    '''Returns the scattering length for a generalised scattering interaction 
    coupling matrix Yf
    mass matrix Mf
    particle'''

    
    '''Extract the cosmological functions'''
    Ma,rada,pbha,Ta,sa,aT = funcs[0],funcs[1],funcs[2],funcs[3],funcs[4],funcs[5]

    alpha,alphaevap = paramBH(Ma,aBH,a0=a0,a1=a1)
    alpha,alphaevap = paramBH(Ma,aBH,a0=alphaevap-0.5,a1=alphaevap+0.5)
    mBH = Ma(alpha)
    T = Ta(alpha)
    
    
    '''Build the BH temperature profile'''
    rcore = rcrit(mBH,T)
    Tcr = profile(Mini,funcs,aBH,rcore/2,M = mBH,alpha = alpha)
    rHS = radiusHS(T,mBH)
    
    def Gamma(MG,YG,iG,mBHG,TG,args = Args):
        '''This generalised, local function should return the total interaction rate including the decay for the particle
        where all diagrams are summed over
        
        TG is the temperature of the local background'''
        if particle == 'RHN':
            S = GammaTT(YG,TG,MG,mBHG,i=iG,accT = args[0],output = args[1],method = args[2],inff = args[3])[0]
            D = GN(YG,TG,MG,i=iG)
            return  S + D
        elif particle == 'DMz':
           mZp,gSM = args[0],args[1]

           mSM = 0.1*TG #approximate thermal mass of the SM bath
           mDMT = DMz.thermalMass(TG,'DM',YG[iG],gSM)
           mZpT = DMz.thermalMass(TG,'DM',YG[iG],gSM)
           
           MG[iG] = mDM + mDMT
           args[0] = mZp + mZpT
           
           
           tchannel = thermalAverage(MG[iG],TG,TH(mBHG),mSM,DMz.sigmat,MG[iG] + mSM,sigArgs = args) * Neqrel(TG)
           '''In the above, if T < TEW it may not make sense to take Neqrel unless photons can scatter and this is dominant'''
           schannel =  thermalAverage(MG[iG],MG[iG],TH(mBHG),TG,DMz.sigmas,2*mSM,sigArgs = args) * Neqsa(TG,MG[iG])
           '''For the s channel I am assuming that the DM has an equilibrium abundance always'''
           
           return tchannel + schannel
        
        
    
    '''It will prove useful to first estimate the scattering length as lambdaCore = 1/Gamma(Tcore) and 
    lambdaBack = 1/Gamma(T_plasma)'''
    Gcore = Gamma(Mf,Yf,i,mBH,Tcr,args = Args)
    lambdaCore = 1/Gcore #Estimate of the scattering length assuming T = Tcr everywhere, underestimate
    Gback = Gamma(Mf,Yf,i,mBH,T,args = Args)
    lambdaBack = 1/Gback #Estimate of scattering length assuming that T  = Tplasma, overestimate
    
    '''We are now left with three possible options'''
    
    if lambdaCore < rcore:
        '''If the estimate lambdaCore does not exceed Rcore, it is accurate and should be returned'''
        if output == 1:
            print('Core only')
        return lambdaCore
    else:
        '''If lambdaCore > Rcore then we must determine which regime the calculation lies in'''
        rlistOP = np.logspace(log10(rcore),log10(rHS),accR) #list of r values in the outer profile
        Glist,rtemp = [],[]
        for r in rlistOP:
            rtemp.append(r)
            Ttemp = profile(Mini,funcs,aBH,r,M = mBH,alpha = alpha)
            
            G = Gamma(Mf,Yf,i,mBH,Ttemp,args = Args)
            Glist.append(G)
        GintOP = simpson(Glist,rlistOP) #This is the scattering rate integrated over the outer profile
        GHSI = GintOP + Gcore*rcore #This is the total integrated scattering rate inside the hotspot

        if log(2) > GHSI:
            case = 1
        else:
            case = 0
            
    if case == 1:
        '''In this case the true scattering length lies outside the hotspot entirely'''
        if output == 1:
            print('Case 1')
        return (log(2) - GHSI )/Gback + rHS
    elif case == 0:
        if output == 1:
            print('Case 0')
        '''In this case the scattering length lies inside the outer profile of the hot-spot'''
        Gint = interp1d(rlistOP,Glist,fill_value = 'extrapolate')
        Gilist = []
        for R in rlistOP:
            tempr = np.logspace(log10(rcore),log10(R),accR)
            Gtemp = [Gint(R) for R in tempr]
            Gilist.append(simpson(Gtemp,tempr))
        Giint = interp1d(Gilist,rlistOP,fill_value = 'extrapolate')
        return Giint(log(2) - rcore * Gcore)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def HSeqn(alpha,ys,params):
    '''Function which returns the rates for the coupled set of Boltzmann Equations for Hawking radiation in a hot-spot'''

    '''Unpack the parameters'''
    Mini,funs,alphaevap,M,g,i_f,model,Tcrit,accR = params
    N,Nescape = ys[0],ys[1] 


    '''Extract the cosmological variables'''
    Ma,rada,pbha,Ta,sa,aT = funs
    T,mBH,rhoPBH = Ta(alpha),Ma(alpha),pbha(alpha)
    TBH =  TH(mBH)
    if mBH == 0:
        NPBH = 0
    else:
        NPBH = rhoPBH/mBH
        NPBH = 1 #temporary line for considering 1 PBH in a comoving volume
    H =  Hf(alpha,pbha,rada)
    a = 10**alpha
    aBH = alpha/alphaevap
    
    if model =='DMz':
        '''Unpack the DM parameters'''
        m,mZp = M
        gDM,gSM = g
        Args = None
        '''Calculate the probability of escape'''
        def Gamma(M,g,empty,mBHG,TG,args = Args):
            mDM,mZp = M
            gDM,gSM = g
            args = [mZp,gSM,gDM]

            mSM = 0.1*TG #approximate thermal mass of the SM bath, valid for T > T_EWPT
            mDMT = mDM + DMz.thermalMass(TG,'DM',gSM,gDM)
            mZpT = mZp + DMz.thermalMass(TG,'Zprime',gSM,gDM)
            
            args[0] = mZpT
            
            '''t-channel DM SM -> DM SM'''
            tchannel =  thermalAverage(mDMT,mSM, TH(mBHG),TG,DMz.sigmat,mDMT + mSM,sigArgs = args) *  Neqrel(TG)
            '''t-channel DM DM  -> ZZ'''
            tchannelZ =  thermalAverage(mDMT,mDMT, TH(mBHG),TG,DMz.sigmatZ,2*mZpT,sigArgs = args) *  Neqsa(TG,mDMT)
            '''s-channel DM DM -> SM SM'''
            schannel =   thermalAverage(mDMT,mDMT, TH(mBHG),TG,DMz.sigmas,2*mSM,sigArgs = args) *  Neqsa(TG,mDMT)

            
            return tchannel + schannel + tchannelZ
        
    elif model =='vMSM':
        m = M[i_f]
        Args = [1000,0,'Underwood',1e10]
        def Gamma(MG,YG,iG,mBHG,TG,args = Args):
            S = GammaTT(YG,TG,MG,mBHG,i=iG,accT = args[0],output = args[1],method = args[2],inff = args[3])[0]
            D = GN(YG,TG,MG,i=iG)
            return  S + D
        
    '''Calculate the rate of DM production due to the PBHs'''
    GBH =  Gammaf("RHN",mBH,m)
    
    
    GHSI =  GammaHSI(M,g,Mini,funs,aBH,i_f,Gamma,Args=Args,Tcrit = Tcrit,alpha = alpha,accR = accR) #Integrated scattering rate over the HS
    
    Pescape = np.exp(-GHSI)
    
    '''Calculate the rates'''
    pref = log(10)/(H)
    dNDM = [pref*GBH * NPBH,pref*GBH * NPBH * Pescape]
    return dNDM


def PBHcut(alpha,ys,params):
    '''Unpack the parameters'''
    Mini,funs,alphaevap,M,g,i_f,model,Tcrit,accR = params
    N,Nescape = ys[0],ys[1] 
    
    '''Extract the cosmological variables'''
    Ma,rada,pbha,Ta,sa = funs[0],funs[1],funs[2],funs[3],funs[4]
    T,mBH,rhoPBH = Ta(alpha),Ma(alpha),pbha(alpha)
    indicator = mBH -  Mpl
    return indicator

   
def Nescape(M,Mini,funs,model = 'DMz',gDM = None,mZrange = [2,6],gDMrange = [-3,1],Nalpha = 100,method = 'Radau',debug=False,Yf = None,i_f = 1,Tcrit = None,accR = 100):
    '''Function which solves the differential equation for production and escape of a particle X from a PBH
    Common arguments:
        M is the mass matrix of the model. In the DM Z' model, it takes the form [mDM,mZp]'
        Mini is the initial mass of the PBH in units of GeV
        funs are the interpolated Friedmann equations
        
        Nalpha is the number of points considered in alpha between HS formation and PBH evaporation
        method is the method the differential equation solver uses. Radau by default
        debug if True prints out information during running dor debugging
        
    The 'model' parameter determines what X is and has two possible values:
        
        model == DMz:
            This selects the case of a fermionic dark matter particle X which couples to a Z' boson
            Coupling strength of X to Z' is gDM
            mZp is the mass of Zprime given as M[1].
            Only one of mZp or gDM must be set different from None. If either is None, the other is set to satisfy the relic DM abundance
            gSM  is the coupling of the Z' to the SM sector.  = g_EW by default
            
            mZrange,gDMrange define the search ranges for numerically finding gDM or mZp to satisfy the relic density, in log_10
        
        
        model == vMSM:
            Yf is the Yukawa matrix coupling the RHNs to the Higgs and SM leptons
            i_f is the (matrix) index of the RHN being considered. By default i_f = 1 selects N_2
            
            
    
        '''

    '''First step is to extract the cosmological information'''
    Ma,rada,pbha,Ta,sa,aT = funs
    empty,alphaevap =  paramBH(Ma,0.9,a0=5,a1=25)
    empty,alphaevap =  paramBH(Ma,0.9,a0=alphaevap-0.1,a1=alphaevap+0.1,acc = int(1e6))
    Tform =  T0(Mini)
    THSform =  THS(Mini)
    alphaform = aT(THSform)
    
    
    if model == 'DMz':  
        '''For the DM model, we initialise the couplings and mass matrices and solve for missing parameters'''
        gSM = g_EW
        try:
            mZp = M[1]
        except IndexError() as e:
            mZp = None
            M = [M[0],mZp]
            print(e)
        i_f = 0 #this indexes the mass matrix correctly in the case where there is only one DM species 
        if mZp == None:
            '''Calculate mZp which reproduces the observed DM abundance'''
            mZp = DMz.solvemZp(mDM =  M[i_f],gDM = gDM,gSM=gSM,acc = 1000,mZmin = mZrange[0],mZmax = mZrange[1])  
            if debug == True:
                print('M_Z = ' + str(mZp))
        if gDM == None:
            '''Calculate gDM which reproduces the observed DM abundance'''
            gDM = DMz.solvegDM(mDM= M[i_f],mZp = mZp,gSM=gSM,acc = 1000,gmin = gDMrange[0],gmax = gDMrange[1])
            if debug == True:
                print('g_DM = ' + str(gDM))
        g = [gDM,gSM]
        
    elif model == 'vMSM':
        '''For the vMSM, we initialise the coupling matrix to Yf'''
        g = Yf
        

    '''Now call the Boltzmann Equation solver'''
    if Nalpha == None:
        alpha_eval = None
    else:
        alpha_eval = np.linspace(alphaform+0.1, alphaevap,Nalpha)
    y0 = [0,0] #set the initial conditions
    PBHcut.terminal = True
    sol = solve_ivp(fun = HSeqn,t_span = (alphaform + 0.1, alphaevap),y0 = y0,events = [PBHcut],args = [[Mini,funs,alphaevap,M,g,i_f,model,Tcrit,accR]],method = method,
                    t_eval = alpha_eval)

    return sol
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def DMescape(mDM,Mini,funs,gDM = None,mZp = None,mZrange = [2,6],gDMrange = [-3,1],Nalpha = 100,method = 'Radau',debug=False):
    '''Function which solves the differential equation for production and escape of DM from a PBH'''
    '''gDM is the coupling of the Z prime to DM
       I set gSM = g_EW
       mDM is the mass of the DM
       mZp is the mass of Zprime
       Either gDM or mZp is given as None and the other is provided
       The function ensures that the relic density of DM is satisfied'''
    '''funs are the interpolated Friedmann equations'''
    '''Nalpha is the number of points used to sample the solution'''

    '''First is to extract the cosmological information'''
    Ma,rada,pbha,Ta,sa,aT = funs[0],funs[1],funs[2],funs[3],funs[4], funs[5]
    empty,alphaevap =  paramBH(Ma,0.9,a0=5,a1=25)
    empty,alphaevap =  paramBH(Ma,0.9,a0=alphaevap-0.1,a1=alphaevap+0.1)
    Tform =  T0(Mini)
    THSform =  THS(Mini)
    alphaform = aT(THSform)
    gSM = g_EW
    
    if mZp == None:
        '''Calculate mZp which reproduces the observed DM abundance'''
        mZp = DMz.solvemZp(mDM = mDM,gDM = gDM,gSM=gSM,acc = 1000,mZmin = mZrange[0],mZmax = mZrange[1])  
        if debug == True:
            print('M_Z = ' + str(mZp))
    elif gDM == None:
        '''Calculate gDM which reproduces the observed DM abundance'''
        gDM = DMz.solvegDM(mDM=mDM,mZp = mZp,gSM=gSM,acc = 1000,gmin = gDMrange[0],gmax = gDMrange[1])
        if debug == True:
            print('g_DM = ' + str(gDM))
        

    '''Now call the Boltzmann Equation solver'''
    if Nalpha == None:
        alpha_eval = None
    else:
        alpha_eval = np.linspace(alphaform+0.1, alphaevap,Nalpha)
    y0 = [0,0] #set the initial conditions
    PBHcut.terminal = True
    sol = solve_ivp(fun = HSeqn,t_span = (alphaform+0.1,alphaevap),y0 = y0,events = [PBHcut],args = [[Mini,funs,mDM,mZp,alphaevap,gDM,accR]],method = method,
                    t_eval = alpha_eval)
    return sol
    
   
    
   
def read(path,master= [],output = 0,dtype = float,mi = 10000,length = 9,neg = 1, ref = [],size = None):
    Ys = []
    Mtemp,Mcheck = [],[]
    filetemp,iterable = '',[]
    sort = False 
    if master != []:
        for m in master:
            ms = format(m,f'.{1}e')
            ms = path + ms +'.txt'
            iterable.append([float(m),ms])
    else:
        for filename in glob.glob(path + "*"):
            name = os.path.basename(filename)[:length]
            iterable.append([float(name),filename])
        sort = True
    for it in iterable:
        Ystemp = []
        M,Mstring = it[0],format(it[0],f'.{1}e')
        if ref !=[]:
            Mcheck.append(float(Mstring))  
        data = np.genfromtxt(it[1], dtype=dtype)
        if output == 3:
            print('Reading file ' + Mstring)
        if size == None:
            size = np.size(data)
        for j in range(0,size):
            nan = False
            if np.size(data[j]) != 1:
                for k in range(0,np.size(data[j])):
                    if isnan(data[j][k]) == True:
                        if output != 1:
                            print('Nan detected in ' + Mstring)
                        nan = True
                if j <= mi and nan == False:
                    datareal = [data[j][k].real for k in range(0,np.size(data[j]))]
                    Ystemp.append(neg * datareal)
            else:
                if j <= mi and nan == False and isnan(data[j]) == False:
                    Ystemp.append(neg * data[j].real)
                else:
                    pass
        Mtemp.append(M)
        Ys.append(Ystemp)
    if ref !=[]:
        for m in ref:
            ms = format(m,f'.{1}e')
            if float(ms) in Mcheck:
                pass
            else:
                if output != 1:
                    print('Missing mass ' + str(m) + ' detected')
    if sort == True:    
        sorted_indices = sorted(range(len(Mtemp)), key=lambda i: Mtemp[i])
        Msorted = [Mtemp[i] for i in sorted_indices]
        Ysorted = [Ys[i] for i in sorted_indices]
        return [Msorted,Ysorted]
    else:
        return[Mtemp,Ys]
    
    
    