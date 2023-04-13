'''imports'''
from math import *
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

import Libs.pbhcosmology as cosmo

''''constants'''
alpha = 0.2 #gravitational collapse factor
g = 106.75 #model d.o.f
alpha =0.2
GeV_in_g     = 1.782661907e-24  # 1 GeV in g
GCF = 6.70883E-39
Mpl =1.22e19 # [M_pl] = GeV #Mpl in GeV
h =0.673
Mpls = Mpl * (90/(8*pi**3 *g))**0.5 #the reduced planck mass
rhoc = 1.878E-29 * h**2 / GeV_in_g / (1.9733E14)**3 #critical density in GeV^4
vEW = 246 #Higgs vev in GeV
Msolar = 1.116E57 #Solar mass in GeV
mtau = 1.776
me = 0.511E-3
mmuon = 105.7E-3



'''parameters'''
M1 = 1E13 #mass of the decaying RNH in GeV
MPBH = 1 / GeV_in_g    #mass of the PBHs in GeV where numerical factor is the grams
beta = 1e-15 #beta prime we take 
eps = 1.0 #the CP violating parameter
splitsolar = np.sqrt(7.6*1e-5)*1e-9 #solar neutrino mass mixing
splitatm = np.sqrt(2.47*1e-3)*1e-9 #atmospheric neutrino mass mixing squared in GeV
mstar = 1.08e-12     #m* representing the expansion rate at z = 1




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
    m = [ml(mhf),mm(mhf),mhf]
    
    
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
        '''Exact when m1 = m2 = 0'''
        return mhf * np.abs(cm.sin(xf+1j*yf)**2)
        

def Ygen(x,y,M,mh,alpha23=pi/4,delta=0):
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
    
    m_1 = ml(mh)
    m_2 = mm(mh)                                                   # GeV
    m_3 = mh # GeV
    

    m_nu_hat = [[m_1, 0., 0.],
                [0., m_2, 0.],
                [0., 0., m_3]]
    theta = x + 1.j*y

    U_1 = np.array([[0. + 0.j, 0. + 0.j, 0. + 0.j], 
                [0. + 0.j, 0. + 0.j, 0. + 0.j],
                [0. + 0.j, 0. + 0.j, 0. + 0.j]])

    U_1[0, 0] = c12 * c13
    U_1[0, 1] = s12 * c13
    U_1[0, 2] = s13 * np.exp(- 1j * delta)
    
    U_1[1, 0] = - s12 * c23 - c12 * s23 * s13 * np.exp(1j * delta)
    U_1[1, 1] = c12 * c23 - s12 * s23 * s13 * np.exp(1j * delta)
    U_1[1, 2] = s23 * c13
    
    U_1[2, 0] = s12 * s23 - c12 * c23 * s13 * np.exp(1j * delta)
    U_1[2, 1] = - c12 * s23 - s12 * c23 * s13 * np.exp(1j * delta)
    U_1[2, 2] = c23 * c13
    
    U_2 = [[1., 0.                  , 0.                  ],
           [0., np.exp(1j * alpha21/2), 0.                  ],
           [0., 0.                  , np.exp(1j * alpha31/2)]]

    U = dot(U_1, U_2)
    s_theta = cm.sin(theta)
    c_theta = cm.cos(theta)
    
    O = [[0.,   c_theta, s_theta],
         [0., - s_theta, c_theta],
         [1.,        0.,      0.]]
    
    return 1j * sqrt(2)/vEW * dot(dot(U, np.sqrt(m_nu_hat)), dot(transpose(O), np.sqrt(M_hat)))


def I(iN,jN,l,YY,Y,M):
    '''calculating the quantity Iij,aa from eqn 36 (Petcov)'''
    '''iN and jN are both the indexes for the Yukawa matrix not labels'''
    '''alpha is the index for the flavour'''
    '''YY is the product Y^daggerY, Y is the Yukawa matrix'''
    
    '''first define the hermitian conjugate of Y'''
    YH = np.matrix.getH(Y)
    
    '''calculate I'''
    denom = YY[iN][iN]*YY[jN][jN]
    num = complex(YH[iN][l]*Y[l][jN]*YY[iN][jN]).imag + M[iN]/M[jN]*complex(YH[iN][l]*Y[l][jN]*YY[jN][iN]).imag
    return complex(num/denom).real


def xTz(z,Gamma11,Gamma22,Gamma12):
    '''calculates the thermal corrections to the mass splitting'''
    
    return pi/(4*z**2) * np.sqrt ( (1-Gamma11/Gamma22)**2 + 4*np.abs(Gamma12)**2/Gamma22**2) 
    
def epsilon(dM,M2f,Y,i,l,z):
    '''The flavour dependent CP asymmetry in the decay of N_isuch that i is not an index for the code but the label
    dM is the zero temperature mass splitting, in GeV, z is M1/T, M2 is the heavy neutrino N2's mass 
    Y is the Yukawa matrix, l is the label of the flavour so that l = e,mu,tau (0,1,2)'''
    
    '''first, set the index of i and initialise epsilon'''
    iN = i-1
    eptotal=0
    
    '''then calculate the M matrix'''
    M = [M2-dM,M2f,1e16]
        
    
    '''Next, define the Yukwa matrix products'''
    YY,YH = np.matmul(np.matrix.getH(Y),Y),np.matrix.getH(Y) #
    
    '''calculate the parameter x0, and the total decay widths of N_1,2 
    plus off diagonal term'''    
    Gamma22,Gamma11 = Gammaii(Y,2,M),Gammaii(Y,1,M)
    Gamma12 = Gammaij(YY,0,1,M)
    x0 = dM/Gamma22 #
    
    
    
    '''Then calculate the thermal corrections to the mass splitting'''
    xT = xTz(z,Gamma11,Gamma22,Gamma12) #
    
    '''Now having calculated the necessary constants, perform the sum'''
    '''jN is an index and not a label'''
    for jN in range(0,3):
        if jN != iN:
            
            '''Calculate the masses and decay widths of N_j,i'''
            Mi,Mj = M[iN],M[jN]
            Gammajj = Gammaii(Y,jN+1,M)
            
            '''Calculate the quantity Iij,aa and the sign of Mi-Mj'''
            Iij = I(iN,jN,l,YY,Y,M) #
            sgn = (Mi-Mj)/(np.abs(Mi-Mj)) #
            
            '''calculate the decay asymmetry and add it to the total;='''
            Prefactor = sgn*Iij #
            Numerator = 2*x0*gammaf(z) #
            Denominator1 =  4*Gamma22/Gammajj*(x0 + xT)**2 
            Denominator2 = Gammajj/Gamma22*gammaf(z)**2 
            Denominator = Denominator1 + Denominator2
            eptotal += Prefactor*Numerator/Denominator
    return complex(eptotal).real

gam = np.genfromtxt('Data/gamma.txt')
gammaf = interp1d(gam[:,0],gam[:,1],fill_value="extrapolate")

def Gammaii(Y,i,M):
    '''returns the total decay rate of the heavy neutrino i'''
    '''i is the label not the index'''
    iN = i-1
    YY = np.array(np.matmul(np.matrix(Y).H,Y))[iN][iN]
    return YY*M[iN]/(8*pi)

def Gammaij(YY,i,j,M):
    '''returns the off diagonal decay rate Gamma_ij'''
    '''i and j are labels not indices'''
    '''M is the heavy neutrino mass matrix'''
    '''YY is the Yukawa matrix product Y^daggerY'''
    iN,jN = i-1,j-1
    return YY[iN][jN]*np.sqrt(M[iN]*M[jN])/(8*pi)


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

def Neqsa(alpha,T,species="RHN",Mf = 1):
    '''semi-approximated version of the function Neq'''
    '''uses the asymptotic limits of the integral form to avoid integration except in the narrow range z = 0.1 to  z = 10'''
    '''alpha and T are always provided, species and Mf are optional'''
    a = 10**alpha 
    g = 0
    plus = 1
    if species=="RHN":
        g = 2
        mass = Mf
    if species == "electron":
        g = 2
        mass = me
    if species == "muon":
        g = 2
        mass = mmuon
    if species == "tauon":
        g = 2 
        mass = mtau
    if mass/T < 1E3:
        z = mass/T
        if z < 0.1:
            return Neqrel(T,a,species=species,Mf=Mf)
        if z > 10:
            return Neqnrel(T,a,species = species,Mf=Mf)
        else:
            p_list = np.logspace(float(log10(T/10)), float(log10(T*100)), 100)
            E_list = [sqrt(mass**2 + p**2) for p in p_list]
            int_list = [1/(e**(E_list[i]/T)+plus * 1)*p_list[i]**2 for i in range(0,100)]
            integral = simpson(int_list, p_list)
            return a**3  * 2 * g/(2 * np.pi)**2 * integral
    else:
        return 0.
    
def Neqrel(T,a,species="RHN",Mf = 1):
    '''the relativistic limit of Neq'''
    mass  = Mf
    g = 2
    return 0.75 * g * 1.2 * T ** 3 / pi**2 * a ** 3

def Neqnrel(T,a,species="RHN",Mf = 1):
    '''the non-relativistic limit of Neq'''
    mass  = Mf
    g = 2
    return  g * ( mass * T / (2 * pi) ) ** 1.5 * np.exp(-mass/T) * a ** 3


def BEsNi(alpha,Ni,Mpbhf,pbhf,radf,Tf,i,M,Y,ratio):
    '''Function which solves the BEs for the RHN number density'''
    '''Ns is four dimensional, 0th element is N1 number density and the flavoured densities are the other 3'''
    '''M2 is the right handed neutrino mass M_2, dM is the 0 temp mass splitting, and i is the label of the RHN'''
    
    '''First extract the number densities, and cosmological variables'''
    MBH,rPBH,rRad,T = Mpbhf(alpha),pbhf(alpha),radf(alpha),Tf(alpha)
    H = Hf(alpha,pbhf,radf)
    z=M[1]/T

    '''Calculate the N_i equilibrium density'''
    #Neq = Neqsa(alpha,T,Mf=M[i-1])
    Neq = 3/8 * z ** 2  * kn(2,z) * ratio
    
    '''Calculate the thermally averaged decay rate'''
    Gii = Gammaii(Y,i,M)
    GiiT = mykn(1,z)/mykn(2,z) * Gii
    
    '''Calculate the rate of change'''
    dN = log(10)/H * (Neq - Ni)*GiiT
    return dN
                  
def BEsRBL(alpha,NBL,l,M,Y,P,Nis,Mpbhf,pbhf,radf,Tf,ratio):
    '''Function which solves the BEs for the RHN number density and BL number density'''
    '''Ns is four dimensional, 0th element is N1 number density and the flavoured densities are the other 3'''
    '''M2 is the right handed neutrino mass M_2, dM is the 0 temp mass splitting, and i is the label of the RHN'''

    '''First extract the number densities, and cosmological variables'''
    MBH,rPBH,rRad,T = Mpbhf(alpha),pbhf(alpha),radf(alpha),Tf(alpha)
    H,z = Hf(alpha,pbhf,radf),M[1]/T
    Ni = [Nis[0](alpha),Nis[1](alpha)]
    
    '''Next, extract the RHN parameters'''
    M2f,dMf = M[1],M[1]-M[0]
    
    '''Calculate the decay rate and lepton eq density'''
    
    NeqLe,NeqLmu,NeqLtau = Neqsa(alpha,T,species="electron"),Neqsa(alpha,T,species="muon"),Neqsa(alpha,T,species="tauon")
    NeqL = [NeqLe,NeqLmu,NeqLtau]
    
    '''Calculate the asymmetries'''
    dNBL = 0
    '''j is the label not the index'''
    for j in (1,2): 
        eps = epsilon(dM=dMf,M2f=M2f,Y=Y,i=j,l=l,z=z)
        '''Calculate the N_i equilibrium density'''
        #Neq = Neqsa(alpha,T,Mf=M[i-1])
        Neq = 3/8 * z ** 2  * kn(2,z) * ratio
        '''Calculate the thermally averaged decay rate. The decays, proportional to Ni, give a +ve contribution
        to the asymmetry since they will be subdominant for the whole evolution. Therefore, the -ve contribution
        of the inverse decays generates the asymmetry when epsilon is also negative'''
        Gii = Gammaii(Y,j,M)
        GiiT = mykn(1,z)/mykn(2,z) * Gii
        W = -GiiT/2 * Neq/NeqL[l]  * P[j-1][l] * NBL
        D = eps*(Ni[j-1]-Neq)*GiiT
        dNBL += log(10)/H * (D+W)
    return dNBL

def flavcut(alpha,NBL,l,M,Y,P,Nis,Mpbhf,pbhf,radf,Tf,ratio):
    '''Function which solves the BEs for the RHN number density and BL number density'''
    '''Ns is four dimensional, 0th element is N1 number density and the flavoured densities are the other 3'''
    '''M2 is the right handed neutrino mass M_2, dM is the 0 temp mass splitting, and i is the label of the RHN'''

    '''First extract the number densities, and cosmological variables'''
    MBH,rPBH,rRad,T = Mpbhf(alpha),pbhf(alpha),radf(alpha),Tf(alpha)
    H,z = Hf(alpha,pbhf,radf),M[1]/T
    Ni = [Nis[0](alpha),Nis[1](alpha)]
    NBL = NBL[0]
    '''Next, extract the RHN parameters'''
    M2f,dMf = M[1],M[1]-M[0]
    
    '''Calculate the decay rate and lepton eq density'''
    
    NeqLe,NeqLmu,NeqLtau = Neqsa(alpha,T,species="electron"),Neqsa(alpha,T,species="muon"),Neqsa(alpha,T,species="tauon")
    NeqL = [NeqLe,NeqLmu,NeqLtau]
    
    '''Calculate the asymmetries'''
    W,D = 0,0
    '''j is the label not the index'''
    for j in (1,2): 
        eps = epsilon(dM=dMf,M2f=M2f,Y=Y,i=j,l=l,z=z)
        '''Calculate the N_i equilibrium density'''
        Neq = Neqsa(alpha,T,Mf=M[j-1])
        '''Calculate the thermally averaged decay rate. The decays, proportional to Ni, give a +ve contribution
        to the asymmetry since they will be subdominant for the whole evolution. Therefore, the -ve contribution
        of the inverse decays generates the asymmetry when epsilon is also negative'''
        Gii = Gammaii(Y,j,M)
        GiiT = mykn(1,z)/mykn(2,z) * Gii
        W += -GiiT/2 * Neq/NeqL[l]  * P[j-1][l] * NBL
        D += eps*(Ni[j-1]-Neq)*GiiT
    return complex(W-D).real
    







def solveR(xf,yf,dM,M2f,mhf,functions=None,MBH=None,betaf=None,delta = 0,rtol=1e-10,atol=1e-12):
    '''solving function for the BEs in the resonant leptogenesis (flavoured) case'''
    '''the leptogenesis parameter space spans x,y,dM,M2 and mhf and the PBH parameters are the mass and abundance'''
    '''if functions is not provided, MBH and betaf must be, and viceversa'''
    
    '''Calculate the RHN and active neutrino mass matrices'''
    M,Mv= [M2f-dM,M2f,1e16],[ml(mhf),mm(mhf),mhf]
    
    '''first calculate or extract the cosmological functions'''
    if functions == None and MBH != None and betaf != None:
        functions = cosmo.functions(MBH=MBH,betaf=betaf,MN= M,Mv=Mv)
    Malpha,radalpha,pbhalpha,Talpha,salpha,alphaT = functions[0],functions[1],functions[2],functions[3],functions[4],functions[5]
    ratio = Neqsa(0,Talpha(0),Mf=M2f)/0.75
    
    '''calculate the starting scale factor'''
    Tstart = 1e4
    zstart = M2f/Tstart
    alphai,alphaf = alphaT(Tstart),alphaT(100)
    
    '''Next, calculate the Yukawa matrix and its Hermitian conjugate'''
    Y = Ygen(xf,yf,M,mhf,delta=delta)
    YY = np.array(dot(np.matrix(Y).H,Y))
    
    '''Calculate the flavour projectors'''
    P      = [[np.abs(Y[l][j-1])**2/YY[j-1][j-1] for l in range(0,3)] for j in (1,2)]
    
    '''Check that the flavour projectors are correct'''
    if (P[0][0]+P[0][1]+P[0][2]) < 0.99 or (P[0][0]+P[0][1]+P[0][2]) > 1.01:
        print("Sum of i=1 flavour projectors is not equal to unity, check calculations and inputs")
    if (P[1][0]+P[1][1]+P[1][2]) < 0.99 or (P[1][0]+P[1][1]+P[1][2]) > 1.01:
        print("Sum of i=2 flavour projectors is not equal to unity, check calculations and inputs")    
        
    '''Solve the Ni Boltzmann equation'''
    N1sols = solve_ivp(BEsNi,[alphai,alphaf],[0],method='BDF',args = tuple([Malpha,pbhalpha,radalpha,Talpha,1,M,Y,ratio]),rtol=rtol,atol=atol)
    N2sols = solve_ivp(BEsNi,[alphai,alphaf],[0],method='BDF',args = tuple([Malpha,pbhalpha,radalpha,Talpha,2,M,Y,ratio]),rtol=rtol,atol=atol)
    Nis = [interp1d(N1sols.t,N1sols.y,fill_value="extrapolate"),interp1d(N2sols.t,N2sols.y,fill_value="extrapolate")]
    
    print("Evolution of the number density solved between alpha = " + str(N1sols.t[0]) + " -> alpha = " + str(N1sols.t[-1]))
                                                                           
    '''Solve the asymmetry Boltzmann equations'''
    flavcut.terminal = True
    NBLe = solve_ivp(BEsRBL,[N1sols.t[0],N1sols.t[-1]],[0],method='BDF',args = tuple([0,M,Y,P,Nis,Malpha,pbhalpha,radalpha,Talpha,ratio]),rtol=rtol,atol=atol,events=[flavcut])
    NBLmu = solve_ivp(BEsRBL,[N1sols.t[0],N1sols.t[-1]],[0],method='BDF',args = tuple([1,M,Y,P,Nis,Malpha,pbhalpha,radalpha,Talpha,ratio]),rtol=rtol,atol=atol,events=[flavcut])
    NBLtau = solve_ivp(BEsRBL,[N1sols.t[0],N1sols.t[-1]],[0],method='BDF',args = tuple([2,M,Y,P,Nis,Malpha,pbhalpha,radalpha,Talpha,ratio]),rtol=rtol,atol=atol,events=[flavcut])
    
    '''extract the solutions and append the washout regime solutions'''
    NBLey,NBLmuy,NBLtauy = [N for N in NBLe.y],[N for N in NBLmu.y],[N for N in NBLtau.y]
    '''structure of the solutions array is first index selects flavour, and second as i=0 gives y, i=1 gives t'''
    NBLs = [NBLe,NBLmu,NBLtau]
    BLs = [[NBLey,NBLe.t],[NBLmuy,NBLmu.t],[NBLtauy,NBLtau.t]]
    
    for n in range(0,3):
        if BLs[n][1][-1] < 0.99*alphaf:
            NBL2 = solve_ivp(BEsRBL,[BLs[n][1][-1],alphaf],[BLs[n][0][0][-1]],method='BDF',args = tuple([n,M,Y,P,Nis,Malpha,pbhalpha,radalpha,Talpha,ratio]),rtol=rtol/100,atol=atol/100)
            for j in range(0,np.size(NBL2.t)):
                BLs[n][0][0] = np.append(BLs[n][0][0],NBL2.y[0][j])
                BLs[n][1] = np.append(BLs[n][1],NBL2.t[j])    
    print("Range for mu asymmetry is from " + str(BLs[1][1][0]) + " to " + str(BLs[1][1][-1]))
    
    
    '''Make interpolated functions of the asymmetries'''
    NBLei,NBLmui,NBLtaui = interp1d(BLs[0][1],BLs[0][0],fill_value="extrapolate"),interp1d(BLs[1][1],BLs[1][0],fill_value="extrapolate"),interp1d(BLs[2][1],BLs[2][0],fill_value="extrapolate")
    
    '''Make flavoured summed lists for the yield'''
    YBL = [13/79* (NBLei(a) + NBLmui(a) + NBLtaui(a))/salpha(a) for a in BLs[0][1]]
    YBL = [Y[0] for Y in YBL]

    '''Make list of z for each alphalist and then interpolate the flavour summed asymmetry wrt alpha'''
    YBLa = interp1d(BLs[0][1],YBL,fill_value="extrapolate")
                                  
    return [Nis,YBLa,[NBLei,NBLmui,NBLtaui],alphaT,Talpha]
        