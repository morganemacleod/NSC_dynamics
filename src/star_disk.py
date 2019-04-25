import matplotlib.pyplot as plt
import numpy as np
from Constants import Constants
from nsc_model import nsc_model

    
class disk:
    """After Art. & Lin 1993 and Accretion power Ch 5.3"""
    def __init__(self,Mbh,ms,rs,Mdot_disk,alpha_disk,HoR_disk):
        self.c=Constants()
        self.Mbh = Mbh
        self.ms = ms
        self.rs = rs
        self.Mdot = Mdot_disk
        self.alpha = alpha_disk
        self.HoR = HoR_disk
        
    def sigma(self,r):
        vk = np.sqrt(self.c.G*self.Mbh/r)
        vr = self.alpha*self.HoR**2 * vk
        return self.Mdot/(2*np.pi*r*vr)
    
    def rho(self,r):
        return self.sigma(r) / (self.HoR*r) # rho = sigma/H
            
    def Fdrag(self,r,v):
        # A&L eq 2 (IS THIS A FACTOR OF 4 too high in geometric limit?)
        Cd = 1.0
        vesc = np.sqrt(self.c.G*self.ms/self.rs)
        vc = Cd**0.25 * vesc # A&L eq 3
        f_pre = 4.0*np.pi*(self.c.G * self.ms)**2 * self.rho(r) * Cd
        return f_pre*np.where(v<vc, v**-2, v**2 * vc**-4 )
    
    def VDISK(self,POS):
        R = np.linalg.norm(POS)
        vk = np.sqrt(self.c.G*self.Mbh/R)
        phi_rot = np.arcsin(POS[1]/R)
        VDISK = vk*np.array([-np.sin(phi_rot),np.cos(phi_rot),0.0]) 
        return VDISK
    
    
class disk_Q:
    """After Doug's Notes, constant Mdot, Q, disk"""
    def __init__(self,Mbh,ms,rs,lambda_disk,alpha_disk=1.0,Q_disk=1.0):
        self.c=Constants()
        self.Mbh = Mbh
        self.ms = ms
        self.rs = rs       
        Mdot_Edd = (4.*np.pi*self.c.G*Mbh*self.c.mp/(0.1 *self.c.sigmaT*self.c.c))
        self.Mdot_disk = lambda_disk*Mdot_Edd
        print ("Mdot_Edd = ",Mdot_Edd/c.msun*c.yr, "(msun/yr) Mdot_disk =",self.Mdot_disk/c.msun*c.yr,"(msun/yr)")
        self.alpha = alpha_disk
        self.Q = Q_disk
        
    def h(self,r):
        omega = np.sqrt((self.c.G*self.Mbh)/r**3)
        return (self.Mdot_disk/(omega*self.Mbh)*(self.Q/(2.*self.alpha)))**(1./3.)
        
        
    def sigma(self,r):
        vk = np.sqrt(self.c.G*self.Mbh/r)
        vr = self.alpha*self.h(r)**2 * vk
        return self.Mdot_disk/(2.*np.pi*r*vr)
    
    def rho(self,r):
        return self.sigma(r) / (2*self.h(r)*r) # rho = sigma/2H
        
    
    def Fdrag(self,r,v):
        Fdrag_geo = np.pi*self.rs**2 * self.rho(r) * v**2 
        Fdrag_grav = 4.0*np.pi*(self.c.G * self.ms)**2 * self.rho(r) / v**2 
        return max(Fdrag_geo,Fdrag_grav)
    
    def VDISK(self,POS):
        R = np.linalg.norm(POS)
        vk = np.sqrt(self.c.G*self.Mbh/R)
        phi_rot = np.arctan2(POS[1],POS[0]) 
        VDISK = vk*np.array([-np.sin(phi_rot),np.cos(phi_rot),0.0]) 
        return VDISK
    

    

class orbit:
    def __init__(self,Mbh):
        self.c=Constants()
        self.Mbh = Mbh
    
    def elements_to_pos_vel_orb_plane(self,elements):
        a,e,OMEGA,omega,I,f = elements
        r = a*(1-e**2)/(1+e*np.cos(f))
        x = r*np.cos(f)
        y = r*np.sin(f)
        z = 0.0
        n=np.sqrt(self.c.G*self.Mbh/a**3) # mean motion
        vx = - n*a/np.sqrt(1-e**2) * np.sin(f)
        vy = n*a/np.sqrt(1-e**2) * (e + np.cos(f))
        vz = 0.0
        pos = np.array([x,y,z])
        vel = np.array([vx,vy,vz])
        return pos,vel
    
    
    def P1(self,omega):
        return np.array([[np.cos(omega),-np.sin(omega),0],
                       [np.sin(omega),np.cos(omega),0],
                       [0,0,1]])
    
    def P2(self,I):
        return np.array([[1,0,0],
                       [0,np.cos(I),-np.sin(I)],
                       [0,np.sin(I),np.cos(I)]])
    
    def P3(self,OMEGA):
        return np.array([[np.cos(OMEGA),-np.sin(OMEGA),0],
                       [np.sin(OMEGA),np.cos(OMEGA),0],
                       [0,0,1]])
        
    def xyz_to_XYZ(self,pos,vel,elements):
        a,e,OMEGA,omega,I,f = elements
        P1 = self.P1(omega)
        P2 = self.P2(I)
        P3 = self.P3(OMEGA)
        POS = np.matmul(P3,np.matmul(P2,np.matmul(P1,pos)))
        VEL = np.matmul(P3,np.matmul(P2,np.matmul(P1,vel)))
        
        return POS,VEL
    
    def XYZ_to_xyz(self,POS,VEL,elements):
        a,e,OMEGA,omega,I,f = elements
        P1 = self.P1(omega)
        P2 = self.P2(I)
        P3 = self.P3(OMEGA)
        pos = np.matmul(P1.T,np.matmul(P2.T,np.matmul(P3.T,POS)))
        vel = np.matmul(P1.T,np.matmul(P2.T,np.matmul(P3.T,VEL)))
        
        return pos,vel
    
    def h_vec(self,pos,vel):
        return np.cross(pos,vel)
    
    def elements_to_POS_VEL(self,elements):
        pos,vel = self.elements_to_pos_vel_orb_plane(elements)
        return self.xyz_to_XYZ(pos,vel,elements)
    
    def POS_VEL_to_elements(self,POS,VEL):
        """Murray&Dermott p52-53, eq 2.126 - 2.139"""
        R = np.sqrt(POS[0]**2 + POS[1]**2 + POS[2]**2)
        V = np.sqrt(VEL[0]**2 + VEL[1]**2 + VEL[2]**2)
        a = (2/R - V**2/(self.c.G*self.Mbh))**-1
        hv = self.h_vec(POS,VEL)
        h = np.sqrt(hv[0]**2 + hv[1]**2 + hv[2]**2)
        e = np.sqrt(1.-h**2/(self.c.G*self.Mbh*a))
        I = np.arccos(hv[2]/h)
        #OMEGA=np.where(hv[2]>0,np.arctan2(hv[0],-hv[1]),np.arctan2(hv[0],-hv[1]))
        OMEGA=np.arctan2(hv[0],-hv[1])
        #omega_plus_f = np.arcsin(POS[2]/(R*np.sin(I)))
        sin_omega_plus_f = POS[2]/(R*np.sin(I))
        cos_omega_plus_f = 1./np.cos(OMEGA)*(POS[0]/R + np.sin(OMEGA)*sin_omega_plus_f*np.cos(I))
        omega_plus_f = np.arctan2(sin_omega_plus_f,cos_omega_plus_f)
        Rdot = (POS[0]*VEL[0] + POS[1]*VEL[1] + POS[2]*VEL[2])/R
        #f = np.arcsin(a*(1-e**2)/(h*e)*Rdot)
        f=np.arctan2( a*(1-e**2)*Rdot*R, h*(a*(1-e**2)-R))
        omega = omega_plus_f-f
        omega += np.where(omega>np.pi,-2*np.pi,0.0)
        elements = (a,e,OMEGA,omega,I,f)
        return elements
        
    def f_node_crossings(self,omega):
        f1=-omega
        f2=np.where(omega<0,-omega-np.pi,-omega+np.pi) 
        return np.where(omega>0,(f1,f2),(f2,f1)) 
    
        

class star_disk:
    def __init__(self,Mbh,ms,rs,lambda_disk=1,alpha_disk=1,Q_disk=1,gamma=1.5):
        self.c=Constants()
        self.Mbh = Mbh
        self.ms = ms
        self.rs = rs
        self.orb = orbit(Mbh)
        #self.d = disk(Mbh,ms,rs,Mdot_disk,alpha_disk,HoR_disk)
        
        self.d = disk_Q(Mbh,ms,rs,lambda_disk,alpha_disk,Q_disk)
        self.n= nsc_model(Mbh,ms,gamma)
        
    def disk_cross(self,elements,direction,debug=False):
        a,e,OMEGA,omega,I = elements
        f_in,f_out = self.orb.f_node_crossings(omega)
        if direction=="in":
            f_cross = f_in
        if direction=="out":
            f_cross = f_out
        if debug:
            print ("f=",f_in,f_out,f_cross)
        my_ele = (a,e,OMEGA,omega,I, f_cross)
        POS,VEL = self.orb.elements_to_POS_VEL(my_ele)
        if debug:
            print ("p,v=",POS,VEL)
        VREL = VEL-self.d.VDISK(POS)
        if debug:
            print ("vrel=",VREL)
        r = np.linalg.norm(POS)
        if debug:
            print ("r=",r)
        vrel = np.linalg.norm(VREL)
        tcross = (2*self.d.h(r)*r/np.abs(VEL[2]))
        dV = -self.d.Fdrag(r,vrel)/self.d.ms *(VREL/vrel)*tcross
        if debug:
            print ("tcross",tcross)
            print ("dV=",dV)
        VEL += dV
        new_ele = self.orb.POS_VEL_to_elements(POS,VEL)
        return new_ele[:-1]
        
    
    def orb_delta_EJ(self,elements):
        a,e,OMEGA,omega,I = elements
        E0 = -self.c.G*self.Mbh/(2*a)
        POS,VEL = self.orb.elements_to_POS_VEL((a,e,OMEGA,omega,I,np.pi/2))
        hv = self.orb.h_vec(POS,VEL)
        h0 = np.linalg.norm(hv)
        
        my_ele = (a,e,OMEGA,omega,I)
        c1_ele = s.disk_cross(my_ele,"in")
        c2_ele = s.disk_cross(c1_ele,"out")
        
        a,e,OMEGA,omega,I = c2_ele
        Ef = -self.c.G*self.Mbh/(2*a)
        POS,VEL = self.orb.elements_to_POS_VEL((a,e,OMEGA,omega,I,np.pi/2))
        hv = self.orb.h_vec(POS,VEL)
        hf = np.linalg.norm(hv)
        
        dE = Ef-E0
        dh = hf-h0
        return dE,E0,dh,h0
    


        
