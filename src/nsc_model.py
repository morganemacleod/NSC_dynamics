import matplotlib.pyplot as plt
import numpy as np
from Constants import Constants
import scipy.special
from scipy.integrate import quad



class nsc_model:
    def __init__(self,Mbh,ms,gamma=1.5,
                 rm_o_rh=1.,mrm_o_mbh=2.):
        """ Simple power law NSC model, 
        following Merritt sec 3.2.1 (p74)"""
        self.c=Constants()
        self.Mbh = Mbh
        self.ms = ms
        self.gamma = gamma
        self.sigma_h = 2.3e5*(self.Mbh/self.c.msun)**(1./4.38)  # Kormendy, MacLeod 2014
        self.r_h = self.c.G * self.Mbh/self.sigma_h**2 # sphere of influence
        self.r_m = rm_o_rh*self.r_h  # encloses 2Mbh
        self.n_m = (mrm_o_mbh/2.)*(3-self.gamma)/(2*np.pi) *(self.Mbh/self.ms)*self.r_m**-3  # Merritt eq 3.48
        self.mrm_o_mbh = mrm_o_mbh
        self.phi0 = (self.c.G*self.Mbh/self.r_m)
        self.f0 = ((2*np.pi)**-1.5 * self.n_m * self.phi0**-self.gamma 
                   *scipy.special.gamma(self.gamma+1.)/scipy.special.gamma(self.gamma-0.5) )  # vasiliev & merritt 2013
        
    def rho(self,r):
        """ Stellar mass density as a function of radius  """
        rho_m = self.n_m*self.ms 
        return rho_m * (r/self.r_m)**-self.gamma # eq 3.48
       
    def sigma(self,r):
        """ Stellar velocity dispersion as a function of radius  """
        #return np.sqrt(self.c.G * self.Mbh /((1+self.gamma)*r) + self.sigma_h**2) # eq 3.63a, includes flattening outside r_h
        return np.sqrt(self.c.G * self.Mbh /((1+self.gamma)*r) )   # keplerian regime
  
    def t_r(self,r):
        """ two-body relaxation timescale (r)  """
        lnC = np.log(self.Mbh/self.ms)
        return 0.34*self.sigma(r)**3/(self.c.G**2*self.ms*self.rho(r)*lnC)
    
    def P(self,r):
        """ orbital period given SMA """
        return 2*np.pi*np.sqrt( r**3/(self.c.G*(self.Mbh)) )
    
    def E(self,r):
        """ orbital energy given SMA  """
        return self.c.G*(self.Mbh)/(2.*r)
    
    def a(self,E):
        """ orbital SMA given energy """
        return self.c.G*(self.Mbh)/(2.*E)
    
    def Jc(self,r):
        """ circular angular momentum given SMA """
        return np.sqrt(self.c.G*self.Mbh*r)
        
    def DeltaErms(self,r):
        """ RMS Delta Energy over one orbital period from two-body relaxation """
        return self.E(r)*np.sqrt(self.P(r)/self.t_r(r))
    
    def DeltaJrms(self,r):
        """ RMS Delta Ang Momentum over one orbital period from two-body relaxation """
        return self.Jc(r)*np.sqrt(self.P(r)/self.t_r(r)) 

    def fE(self,E):
        """Distribution function (E); Vasiliev & Merritt 2013"""
        return self.f0*E**(self.gamma-1.5)
    
    def Jlc(self,E,rlc):
        """Loss cone angular momentum given Energy, periapse radius of loss cone
           approximately equal to sqrt(2*G*Mbh*rlc)"""
        return np.sqrt(2*rlc**2 * (self.c.G*self.Mbh/rlc - E) )
    
    def qE(self,E,rlc):
        """ ratio of per orbit scatter to loss cone angular momentum; MacLeod2012"""
        return self.DeltaJrms(self.a(E))**2 / self.Jlc(E,rlc)**2
    
    def Rlc(self,E,rlc):
        """ Dimensionless loss cone angular momentum; MacLeod2012"""
        return self.Jlc(E,rlc)**2/self.Jc(self.a(E))**2
    
    def R0(self,E,rlc):
        """ Dimensionless Cohn & Kulsrud minimum populated Ang momentum; Merritt eq 6.66 p 304"""
        q = self.qE(E,rlc)
        alpha = (q**4 + q**2)**0.25
        return self.Rlc(E,rlc)*np.exp(-alpha)
    
    
    def lnR0Inv(self,E,rlc):
        """Dimensionless Cohn & Kulsrud minimum populated Ang momentum --> log(1/R0), based on eq 6.66 p 304"""
        q = self.qE(E,rlc)
        alpha = (q**4 + q**2)**0.25
        return -np.log(self.Rlc(E,rlc)) + alpha
    
    
    def flux_flc(self,E,rlc):
        """Full loss cone flux as a function of Energy, radius of loss cone
           similar to Merritt eq 6.10b, p293, doesn't assume Jlc^2 = 2GMrp """
        return 4.*np.pi**2 * self.fE(E) * self.Jlc(E,rlc)**2
  
    
    def flux_lc(self,E,rlc):
        """loss cone flux as a function of Energy, radius of loss cone; Merritt eq 6.71 (p304)"""
        return  self.qE(E,rlc)*self.flux_flc(E,rlc)/self.lnR0Inv(E,rlc)


    def flux_lc_disk(self,E,rlc_tidal,rlc_disk):
        """loss cone flux as a function of Energy, radius of loss cone; Merritt eq 6.71 (p304)"""
        if rlc_disk > rlc_tidal:
            return  self.qE(E,rlc_disk)*self.flux_flc(E,rlc_tidal)/self.lnR0Inv(E,rlc_disk) 
        else:
            return self.flux_lc(E,rlc_tidal)
  
    
    def test_plot_TDE_rate(self):
        """TO RUN TEST:
        
        # COMPARE TO WM2004 SETTINGS
        n=nsc_model(Mbh=1.e6*c.msun,
                    ms=c.msun,
                    gamma=2.,
                    rm_o_rh=1.,
                    mrm_o_mbh=2.)

        n.test_plot_TDE_rate()
        
        """
        
        
        rt = 1*(self.Mbh/self.ms)**(1./3.)*1*self.c.rsun

        alist = np.logspace(np.log10(self.r_h/self.c.pc)+2,np.log10(self.r_h/self.c.pc)-4)*self.c.pc

        plt.plot(self.E(alist)/self.E(self.r_h),self.flux_flc(self.E(alist),rt)*self.E(alist)*self.c.yr,color='grey',ls="--")
        plt.plot(self.E(alist)/self.E(self.r_h),self.flux_lc(self.E(alist),rt)*self.E(alist)*self.c.yr)

        plt.loglog()

        plt.xlabel(r'${\cal E} / {\cal E}_{\rm h}$')
        plt.ylabel(r"${\cal E} F_{\rm lc}(\cal E)$ [yr$^{-1}$]")
        #plt.ylim(1.e-8,1.e-3)
        plt.xticks([0.01,0.1,1,10,100,1000,1e4])

        plt.grid()

        rate = quad(self.flux_lc,self.E(alist[0]),self.E(alist[-1]),args=(rt))[0]*self.c.yr

        # Comparison to Wang & Merritt 2004
        # https://arxiv.org/pdf/1307.3268.pdf EQ 33
        wmrate = 4.3e-4*(self.sigma_h/9.e6)**(7./2.)*(self.Mbh/(4.e6*self.c.msun))**(-1.) 

        print ("======================================================" )
        print ("Mbh =",self.Mbh/self.c.msun,"msun" )
        print ("sigma =",self.sigma_h/1.e5,"km/s" )
        print ("total rate =",rate, "   log10(rate) =",np.log10(rate) )
        print ("WM2004 scaled =",wmrate,"   log10(WMrate) =",np.log10(wmrate) )
        print ("ratio rate/WMrate =",rate/wmrate )
        print ("======================================================" )


    
