'''imports'''
from math import *
from sympy import polylog
from scipy.integrate import odeint, solve_ivp, quad, simpson
import numpy as np
from numpy import dot,transpose
import random
from scipy.special import kn
from scipy.interpolate import interp1d, interp2d
import cmath as cm
from scipy.special import *



''''constants'''

g = 106.75 #model d.o.f
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
MPBH = 1 / GeV_in_g    #mass of the PBHs in GeV where numerical factor is the grams
beta = 1e-15 #beta prime we take 
eps = 1.0 #the CP violating parameter
splitsolar = np.sqrt(7.6*1e-5)*1e-9 #solar neutrino mass mixing
splitatm = np.sqrt(2.47*1e-3)*1e-9 #atmospheric neutrino mass mixing squared in GeV
mstar = 1.08e-12     #m* representing the expansion rate at z = 1


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
        return [analytic[0],radalpha,analytic[1],Talpha,salpha,alphaT,alphatotal,Halpha]
    else:
        return funs1


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
    m = [Mlep(mhf),mm(mhf),mhf]
    
    
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
        

def Ygen(x,y,M,mh,alpha23=pi/4,delta=0,param = 'Petcov',massless= False,degen = False):
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

    
    m_1 = Mlep(mh)
    m_2 = mm(mh)                                                   # GeV
    m_3 = mh # GeV
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
    U = U_1
    s_theta = complex(cm.sin(theta))
    c_theta = complex(cm.cos(theta))
    if param == 'Petcov':
        O = [[0. + 0j,   c_theta, s_theta],
             [0. + 0j, - s_theta, c_theta],
             [1. + 0j,        0. + 0j,      0. + 0j]]
    elif param == '2HNL':
             O = [[1.,    0.,    0.],
             [0.,   c_theta, s_theta],
             [0., - s_theta, c_theta]] 
    elif param == 'Strumina':
        O = [[c_theta , 0, s_theta],
             [ 0,   1,    0],
             [-s_theta, 0, c_theta]]
        
    
    return 1.j * sqrt(2.)/vEW * dot(dot(U, np.sqrt(m_nu_hat)), dot(transpose(O), np.sqrt(M_hat)))

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
    
def epsilon(dM,M2f,Y,i,l,z,Mv,x,y):
    '''The flavour dependent CP asymmetry in the decay of N_isuch that i is not an index for the code but the label
    dM is the zero temperature mass splitting, in GeV, z is M1/T, M2 is the heavy neutrino N2's mass 
    Y is the Yukawa matrix, l is the label of the flavour so that l = e,mu,tau (0,1,2)'''
    
    '''first, set the index of i and initialise epsilon'''
    iN = i-1
    eptotal=0
    
    '''then calculate the M matrix'''
    M = [M2f-dM,M2f,1e16]
        
    
    '''Next, define the Yukwa matrix products'''
    YY,YH = np.matmul(np.matrix.getH(Y),Y),np.matrix.getH(Y) #
    
    '''calculate the parameter x0, and the total decay widths of N_1,2 
    plus off diagonal term'''    
    Gamma22,Gamma11 = Gii(M,Mv,x,y,1),Gii(M,Mv,x,y,0)
    Gamma12 = np.sqrt(Gij2(M,Mv,x,y))

    x0 = dM/Gamma11 #
    
    '''Then calculate the thermal corrections to the mass splitting'''
    xT = xTz(z,Gamma11,Gamma22,Gamma12) #
    
    '''Now having calculated the necessary constants, perform the sum'''
    '''jN is an index and not a label'''
    for jN in range(0,3):
        if jN != iN:
            
            '''Calculate the masses and decay widths of N_j,i'''
            Mi,Mj = M[iN],M[jN]
            if Mi-Mj == 0:
                print(i)
            Gammajj = Gii(M,Mv,x,y,jN)
            
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




def GN(YY,T,i,M):
    Mh,Ml= MH(T),Mlep(T,me)
    aH,aL = (Mh**2/M**2),(Ml**2/M**2)
    G = M/(8*pi) * YY[i][i] * (1-aH + aL) * np.sqrt(Lambda(1,aH,aL)) 
    if M > Mh + Ml:
        return M/(8*pi) * YY[i][i] * (1- aH + aL) * np.sqrt(Lambda(1,aH,aL))
    elif Mh > Ml + M:
        return Mh/(16*pi) * YY[i][i] * (-1+ aH - aL) * np.sqrt(Lambda(aH,1,aL)) / (aH**2)
    else:
        return 0

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

def Neqsa(alpha,T,species="RHN",Mf = 1,physical = False):
    '''semi-approximated version of the function Neq'''
    '''uses the asymptotic limits of the integral form to avoid integration except in the narrow range z = 0.1 to  z = 10'''
    '''alpha and T are always provided, species and Mf are optional'''
    a = 10**alpha 
    g = 0
    plus = 1
    if physical == True:
        fac = 1
    elif physical == False:
        fac = a**3
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
            return Neqrel(T,a,species=species,Mf=Mf,physical=physical)  
        if z > 10:
            return Neqnrel(T,a,species = species,Mf=Mf,physical=physical) 
        else:
            p_list = np.logspace(float(log10(T/10)), float(log10(T*100)), 100)
            E_list = [sqrt(mass**2 + p**2) for p in p_list]
            int_list = [1/(e**(E_list[i]/T)+plus * 1)*p_list[i]**2 for i in range(0,100)]
            integral = simpson(int_list, p_list)
            return 2 * g/(2 * np.pi)**2 * integral * fac
    else:
        return 0.
    
def Neqrel(T,a,species="RHN",Mf = 1,physical = False):
    '''the relativistic limit of Neq'''
    mass  = Mf
    g = 2
    if physical == False:
        return 0.75 * g * 1.2 * T ** 3 / pi**2 * a ** 3
    else:
        return 0.75 * g * 1.2 * T ** 3 / pi**2 

def Neqnrel(T,a,species="RHN",Mf = 1,physical = False):
    '''the non-relativistic limit of Neq'''
    mass  = Mf
    g = 2
    if physical == False:
        return  g * ( mass * T / (2 * pi) ) ** 1.5 * np.exp(-mass/T) * a ** 3
    else:
        return  g * ( mass * T / (2 * pi) ) ** 1.5 * np.exp(-mass/T) 

def TH(M):
    '''returns the Hawking temperature in GeV'''
    if M == 0:
        print("uh oh")
        return 1E-20
    else:
        return float( 1 / ( 8 * pi * GCF * M))




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

def U2a(Mv,M,x,y):
    '''Returns the value of U^2 for active neutrino mass matrix Mv, RHN mass matrix M and x and y the real and imaginary parts of mixing angle'''
    m1,m2,m3 = Mv[0],Mv[1],Mv[2]
    M1,M2,dM,M3= M[0],M[1],M[1]-M[0],M[2]
    return ( 2*m1 * M1*(dM + M1) + dM*(m2-m3)*M3*cos(2*x) + (dM + 2*M1) * (m2+m3)*M3*cosh(2*y))/(2*M1*M2*M3)



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
        NPBH = pbhalpha(alpha)/MBH / S
    else:
        NPBH = 0
    '''define the entropy normalised quantities Yl, Yleq'''
    Mlep = [ me, mmuon, mtau]
    Yleq = [ Neqrel(T,a,Mf = ml,physical = True)/s for ml in Mlep]
    YN,YL = [N[0],N[1]],[N[2],N[3],N[4]]
    total = 0
    '''Calculate dTda'''
    epSM,epRHN =  epN(MBH,Mvf,M), epN(MBH,Mvf,M,species = "RHN")
    ep = epSM+epRHN
    total,kappa = 0, 416.3/(30720*pi) *  Mpl**4
    dTda = T*(1 - 0.25*epSM*10**alpha /(H*MBH**3)*pbhalpha(alpha)/radalpha(alpha))
    rates = []
    for k in (0,1,2):
        '''summing over N = N1,N2'''
        for i in (0,1):
            '''scattering rates'''
            gSt,gSs,gDi = scatter[0][i],scatter[1][i],scatter[2]
            '''RHN number densities'''
            Yeq = 3/8*z**2*kn(2,z) * ratio / S
            '''calculate the rate'''
            prefactor = log(10)*dTda/(T*H*s)
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
            W = -P[i][k]*YL[k]/Yleq[k]*( gD/2 + 2*gSt(T) + YN[i]/Yeq * gSs(T) )
            dY = complex(prefactor*( D + W )).real
            if k == 0:
                DN = (1-YN[i]/Yeq)*(gD+gSs(T)*2 + gSt(T)*4)
                Nrate = prefactor * (DN + NPBH*GPBH)
                rates.append(Nrate)
            total += dY 
        rates.append(total)
    return rates



def solvecoupled(xf,yf,dM,M2f,mhf,funs=None,ic=0,MBH=None,betaf=None,delta = 0,zstart=1e-5,output = 3,rtol2=1e-10,atol2=1e-10,rtol3 = 1e-5,atol3 = 1e-20,inst = False,gammaS = None,negative = False,Scatter = False,Acc = 100,interp = False):
    '''solving function for the BEs in the resonant leptogenesis (flavoured) case'''
    '''the leptogenesis parameter space spans x,y,dM,M2 and mhf and the PBH parameters are the mass and abundance'''
    '''if functions is not provided, MBH and betaf must be, and viceversa'''
    
    '''Calculate the RHN and active neutrino mass matrices'''
    Mv = [ ml(mhf), mm(mhf),mhf]
    M = [M2f-dM,M2f,1e16]
    
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
    ratio =  Neqsa(alphai,Talpha(alphai),Mf= me)/0.75
    Tsphal =  sphal(funs)
    NL =  Neqsa(10,Talpha(10),Mf= me) 
    
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
        gAs1,gAs2     = [ geq('As',T,M,0,YY,y=True,accuracy=acc*10)+ geq('As',T,M,0,YY,y=False,accuracy=acc*10) for T in Tlist],[ geq('As',T,M,1,YY,y=True,accuracy=acc*10)+ geq('As',T,M,1,YY,y=False,accuracy=acc*10) for T in Tlist]
        gAtLH1,gAtLH2 = [ geq('AtLH',T,M,0,YY,y=False,accuracy=acc)+ geq('AtLH',T,M,0,YY,y=True,accuracy=acc) for T in Tlist],[ geq('AtLH',T,M,1,YY,y=False,accuracy=acc)+ geq('AtLH',T,M,1,YY,y=True,accuracy=acc) for T in Tlist]
        gAtLA1,gAtLA2 = [ geq('AtLA',T,M,0,YY,y=True,accuracy=acc) +  geq('AtLA',T,M,0,YY,y=False,accuracy=acc) for T in Tlist],[ geq('AtLA',T,M,1,YY,y=True,accuracy=acc) +  geq('AtLA',T,M,1,YY,y=False,accuracy=acc) for T in Tlist]
        
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
    span = [alphai,alphaT(Tsphal)] 
    Nic =  Neqsa(alphai,Tstart,Mf=M2f,physical=True)/(salpha(alphai)*10**(-3*alphai))
    args = [funs,M,Y,Mv,xf,yf,YY,P,ratio,scatter,negative,interp] 
    ic = [Nic,Nic,0,0,0]

    '''Solve the asymmetry Boltzmann equations'''               
    NBL = solve_ivp(coupled,span,ic,args = tuple(args),**options2)

    '''extract the solutions'''
    Ny1,Ny2 = [N for N in NBL.y[0]],[N for N in NBL.y[1]]

    NBLe,NBLmu,NBLtau = NBL.y[2],NBL.y[3],NBL.y[4]
    '''structure of the solutions array is first index selects flavour, and second as i=0 gives y, i=1 gives t'''
    NBLs = [NBLe,NBLmu,NBLtau]

    '''Make interpolated functions of the asymmetries'''
    NBLei,NBLmui,NBLtaui = interp1d(NBL.t,NBLe,fill_value="extrapolate"),interp1d(NBL.t,NBLmu,fill_value="extrapolate"),interp1d(NBL.t,NBLtau,fill_value="extrapolate")

    '''Make flavoured summed lists for the asymmetry density'''
    Sratio = salpha(5) / salpha(25)
    NBLT= [(NBLei(a) + NBLmui(a) + NBLtaui(a))*Sratio for a in NBL.t]

    NBLi = interp1d(NBL.t,NBLT,fill_value = 'extrapolate')
    
    YBL = [NBLi(a)* Chi(M2f/Talpha(a)) for a in NBL.t]

    '''Make list of z for each alphalist and then interpolate the flavour summed asymmetry wrt alpha'''
    YBLa = interp1d(NBL.t,YBL,fill_value="extrapolate")
    
    if inst == True:      
        return [[Ny1,Ny2],[YBLa,NBL.t],[NBLei,NBLmui,NBLtaui],alphaT,Talpha]
    elif inst == False:
        a150 = alphaT(150)
        Bsol = solve_ivp(dB,[a150,alphaT(120)],[NBLi(a150) *  Chi(150)],args=tuple([NBLi,Talpha,pbhalpha,radalpha])
                                 ,method = 'BDF',rtol = rtol3,atol = atol3)               

        Bint =  interp1d(Bsol.t,Bsol.y,fill_value = 'extrapolate')
   
        Tlist = np.linspace(150,120,1000)
        alphalist = [alphaT(T) for T in Tlist]
        YBL = [(Bint(a)) for a in alphalist]
        YBL = [YBL[i][0] for i in range(0,np.size(alphalist))]
        YBLi = interp1d(alphalist,YBL,fill_value = 'extrapolate')
    
        Yinst = [np.abs(NBLi(a))* Chi(Talpha(a)) for a in alphalist]
        
        return [YBLi,alphalist]
                                  




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

def GN(YY,T,i,M):
    Mh,Ml=    MH(T),   Mlep(T,   me)
    aH,aL = (Mh**2/M**2),(Ml**2/M**2)
    if M > Mh + Ml:
        G = M/(8*pi) * YY[i][i] * (1- aH + aL) * np.sqrt(Lambda(1,aH,aL))
        return G
    elif Mh > Ml + M:
        G = Mh/(16*pi) * YY[i][i] * (-1+ aH - aL) * np.sqrt(Lambda(aH,1,aL)) / (aH**2)
        return G
    else:
        return 0
    
def Lambda(a,b,c):
    L = (a-b-c)**2 - 4*b*c
    return L

def gammaD(YY,i,z,M):
    '''thermal averaged rate as given in Hambye'''
    T = M/z
    Mh,Ml=    MH(T),   Mlep(T,   me)
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
        return kn(1,np.sqrt(s)/T)*sigma(s,YY,i,M,T,eps,y)*s**0.5
    
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



def mQ(T):
    C = np.sqrt(1/6*g3**2 + gY**2/288 + 3/32*g2**2 + ht**2/16)
    return C*T

def mW(T):
    return np.sqrt(11/12)*g2*T

def mB(T):
    return np.sqrt(11/12*gY**2*T**2)

def Mlep(T,m=me):
    return np.sqrt( 3/32 * g2**2 + 1/32 * gY**2)*T

def MH(T,EWPT = True,source = 'Riotto'):
    D = dmh(T,source)**2
    Tc = 160
    return np.sqrt( v(T)**2*np.heaviside(Tc-T,1) + D*T**2*(1-Tc**2/T**2)*np.heaviside(-Tc+T,0.))

    
def MHR(T):
    D = dmh(T,'Riotto')**2
    T_c = 160
    m_h = 125
    temp = m_h**2 * (1-T**2/T_c**2) * np.heaviside(T_c-T,1) +  D * (T**2-T_c**2)* np.heaviside(-T_c+T,0.)
    return np.sqrt(temp)
    
def dmh(T,source):
    return np.sqrt(3/16*g2**2 + 1/16*gY**2 + 1/4*ht**2 + 1/2*HQ) 



    


def t(m1,m2,m3,m4,s,pm):
    
    term1 = (m1**2-m2**2-m3**2+m4**2)**2/(4*s)
    
    term2 = (s + m1**2 - m2**2)**2/(4*s) - m1**2 

    term3 =  (s + m3**2 - m4**2)**2/(4*s) - m3**2

    return term1 - (np.sqrt(term2) + pm*np.sqrt(term3))**2
    
def tbracket(func,m1,m2,m3,m4,s,mN):
    tplus,tminus = t(m1,m2,m3,m4,s,1),t(m1,m2,m3,m4,s,-1)
    return func(tplus/mN**2) - func(tminus/mN**2)



''' ################################################################################################################'''
'                                                      ''Cross sections'''
''' ################################################################################################################'''
 

def sigmaAs(s,YY,i,M,T,eps,y = False):
    if y == True:
        aB = (mB(T)/M[i])**2
        G = gY / np.sqrt(6)
        mBoson = mB
    else:
        aB = (mW(T)/M[i])**2
        G = g2
        mBoson = mW
    x = s/M[i]**2
    aQ,aL = (mQ(T)/M[i])**2,( Mlep(T)/M[i])**2
    pref = 3/(16*pi*x**2)*G**2
    def func(t):
        term = 2*(x*(aL-t)*(aL + aL*x - aB) + eps*(2-2*x+x**2))/((aL-t)**2 + eps)
        return 2*t*(x-2) + (2-2*x+x**2)*log((aL-t)**2 + eps) + term
    sig = pref * tbracket(func,M[i], Mlep(T), MH(T),mBoson(T),s,M[i])
    return sig


def sigmaAtLH(s,YY,i,M,T,eps,y = False):
    aH,aL = ( MH(T)/M[i])**2,( Mlep(T)/M[i])**2
    if y == True:
        aB = (mB(T)/M[i])**2
        G = gY/np.sqrt(6)
        mBoson = mB
    else:
        aB = (mW(T)/M[i])**2
        G = g2
        mBoson = mW
    x = s/M[i]**2
    prefLH = np.abs(3*G**2/(8*pi*x*(1-x)))
    def funcLH(t):
        term1 = 2*x*log(np.abs(t-aH))
        term2 = (1+x**2)*log(np.abs(t+x-1-aB-aH))
        return term1 - term2
    tb = tbracket(funcLH, Mlep(T), MH(T),M[i],mBoson(T),s,M[i])
    sigLH = prefLH * tb
    return sigLH

def sigmaAtLA(s,YY,i,M,T,eps,y = False):
    aH,aL = ( MH(T)/M[i])**2,( Mlep(T)/M[i])**2
    if y == True:
        aB = (mB(T)/M[i])**2
        G = gY/np.sqrt(6)
        mBoson = mB
    else:
        aB = (mW(T)/M[i])**2
        G = g2
        mBoson = mW
    x = s/M[i]**2
    prefLA = 3*G**2/(16*pi*x**2)
    def funcLA(t):
        return t**2 + 2*t*(x-2) - 4*(x-1)*log(np.abs(t-aH)) + x*(aB-4*aH)/(aH-t)
    sigLA = prefLA * tbracket(funcLA, Mlep(T),mBoson(T),M[i], MH(T),s,M[i])
    return sigLA


def sigmaHs(s,YY,i,M,T,eps,smin):
    if s < smin:
        return 0.
    aH,aL,aQ = ( MH(T)/M[i])**2,( Mlep(T)/M[i])**2,(mQ(T)/M[i])**2
    pref = 3/(4*pi)*ht**2
    x = s/M[i]**2
    term1 = (x-1-aL)*(x-2*aQ)/(x*(x-aH)**2)
    t2s = ((1+aL-x)**2 - 4*aL)*(1-4*aQ/x)
    term2 = np.sqrt(t2s)
    sig = pref * term1 * term2
    return sig

def sigmaHt(s,YY,i,M,T,eps,y):
    x = s/M[i]**2
    aH,aL,aQ = ( MH(T)/M[i])**2,( Mlep(T)/M[i])**2,(mQ(T)/M[i])**2
    pref = 3/(4*pi*x)*ht**2
    def tpm(pm):
        term2 =  np.sqrt((aQ**2 + (x-1)**2 - 2*aQ*(1+x))*(aL**2 + (x-aQ)**2 - 2*aL*(x+aQ)))
        term1 =   aQ + x - (aQ-x)**2 + aL*(x+aQ-1)
        return 1/(2*x) * (term1 + pm*term2)
    tp,tm = tpm(1),tpm(-1)
    term1 = tp -tm - (1-aH+ aL)*(aH-2*aQ)*(1/(aH-tp) - 1/(aH-tm)) - ( 1-2*aH + aL + 2*aQ)*log(np.abs((tp-aH)/(tm-aH)))
    sig = pref*term1
    return pref*term1



'''Legacy functions'''


def coupledLegacy(alpha,N,funs,M,Yf,Mvf,xf,yf,YY,P,ratio,scatter = [],negative=False):
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
        NPBH = pbhalpha(alpha)/MBH / S
    else:
        NPBH = 0
    '''define the entropy normalised quantities Yl, Yleq'''
    Mlep = [ me, mmuon, mtau]
    Yleq = [ Neqrel(T,a,Mf = ml,physical = True)/s for ml in Mlep]
    YN,YL = [N[0],N[1]],[N[2],N[3],N[4]]
    total = 0
    '''Calculate dTda'''
    epSM,epRHN =  epN(MBH,Mvf,M), epN(MBH,Mvf,M,species = "RHN")
    ep = epSM+epRHN
    total,kappa = 0, 416.3/(30720*pi) *  Mpl**4
    dTda = T*(1 - 0.25*epSM*10**alpha /(H*MBH**3)*pbhalpha(alpha)/radalpha(alpha))
    rates = []
    for k in (0,1,2):
        '''summing over N = N1,N2'''
        for i in (0,1):
            '''scattering rates'''
            gSt,gSs = scatter[i][0],scatter[i][1]
            '''RHN number densities'''
            Yeq = 3/8*z**2*kn(2,z) * ratio / S
            '''calculate the rate'''
            prefactor = log(10)*dTda/(T*H*s)
            decay =  gammaD(YY,i,z,M[i])
            gD = complex(decay[0]).real   #rate of production gammaD
            '''CP asymmetry parameter'''
            eps =  epsilon(dM=dMf,M2f=M[1],Y=Yf,i=i+1,l=k,z=z,Mv=Mvf,x=xf,y=yf)
            if negative == True:
                eps  =  eps*decay[1]
            '''PBH production rate'''
            GPBH =  Gammaf("RHN",MBH,M[i])
            D = gD*((YN[i]/Yeq-1)*eps)
            W = -P[i][k]*YL[k]/Yleq[k]*(gD/2)
            if gD == 0:
                W = -P[i][k]*YL[k]/Yleq[k]*( 2*gSt(T) + YN[i]/Yeq * gSs(T) )
            dY = complex(prefactor*( D + W )).real
            if k == 0:
                DN = (1-YN[i]/Yeq)*(gD)
                if gD == 0:
                    dN =+ (1-YN[i]/Yeq)*(gSs(T)*2 + gSt(T)*4)
                Nrate = prefactor * (DN + NPBH*GPBH)
                rates.append(Nrate)
            total += dY 
        rates.append(total)
    return rates

