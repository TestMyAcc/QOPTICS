from numpy import pi
from numpy import gradient
from numpy import conjugate

def oam(data,dx,dy,dz,length):
    """Calculate orbital angular momentum of the light
    
        Args:
            Data: Distribution of light (3D, with unit)
            dx,dy,dz: Difference along 3 dimensions
            length: Wavelength of the light

        Output:
            (Lx, Ly, Lz): angular momentum of the light 
                of 3 dimensions
        """

    epsilon  = 8.854187817e-12 
    c = 299792458
    angFreq = 2*pi*c/length
    [gray,grax,graz] = gradient(data,dy,dx,dz)
    
    # TODO: the direction may be wrong.....
    Lx = 1j*angFreq*epsilon/2*(data*conjugate(grax) - conjugate(data)*grax) 
    Ly = 1j*angFreq*epsilon/2*(data*conjugate(gray) - conjugate(data)*gray)
    Lz = 1j*angFreq*epsilon/2*(data*conjugate(graz) - conjugate(data)*graz) +   \
         angFreq*2*pi/length*epsilon*abs(data)**2*1  
    
    return (Lx, Ly, Lz)