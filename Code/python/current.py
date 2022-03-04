from numpy import pi
from numpy import gradient
from numpy import conjugate
    
def current(data,dx: float,dy,dz,m):
    """Calculating probability density current

    Args:
        data: Wavefunction (3D, with unit)
        dx,dy,dz: Difference along 3 dimensions
        m: mass of ato
    Returns:
        Jx, Jy, Jz: Probability density current 
            of three dimensions
    """

    hbar = 6.62607004e-34/(2*pi)
    
    [VxData,VyData,VzData] = gradient(data,dx,dy,dz)
    [VxData_C,VyData_C,VzData_C] = gradient(conjugate(data),dx,dy,dz)

    Jx = (hbar/(2j*m))*(conjugate(data)*VxData - data*VxData_C) 
    Jy = (hbar/(2j*m))*(conjugate(data)*VyData - data*VyData_C)
    Jz = (hbar/(2j*m))*(conjugate(data)*VzData - data*VzData_C)
    
# Equal to first one
#     Jx = (hbar/(m))*imag(conj(data).*VxData) 
#     Jy = (hbar/(m))*imag(conj(data).*VyData)
#     Jz = (hbar/(m))*imag(conj(data).*VzData)
    
#Dimensionless equaltion, has slightly difference.
#         Jx = omegaZ*unit^-2*imag(conj(data).*VxData) 
#         Jy = omegaZ*unit^-2*imag(conj(data).*VyData)
#         Jz = omegaZ*unit^-2*imag(conj(data).*VzData)
    return (Jx, Jy, Jz)