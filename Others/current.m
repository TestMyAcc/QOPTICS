function [Jx,Jy,Jz] = current(Data,dx,dy,dz,m)
%Current of three direction
%Data: Wavefunction
%dx,dy,dz: resolution
%m: mass

    hbar = 6.62607004e-34/(2*pi);    
    
    [VxData,VyData,VzData] = gradient(Data,dx,dy,dz);
    [VxData_C,VyData_C,VzData_C] = gradient(conj(Data),dx,dy,dz); 

    Jx = (hbar/(2i*m))*(conj(Data) .*VxData - Data.*VxData_C); 
    Jy = (hbar/(2i*m))*(conj(Data).*VyData - Data.*VyData_C);
    Jz = (hbar/(2i*m))*(conj(Data).*VzData - Data.*VzData_C);
    
% Equal to first one
%     Jx = (hbar/(m))*imag(conj(Data).*VxData); 
%     Jy = (hbar/(m))*imag(conj(Data).*VyData);
%     Jz = (hbar/(m))*imag(conj(Data).*VzData);
    
%Dimensionless equaltion, has slightly difference.
%         Jx = omegaZ*unit^-2*imag(conj(Data).*VxData); 
%         Jy = omegaZ*unit^-2*imag(conj(Data).*VyData);
%         Jz = omegaZ*unit^-2*imag(conj(Data).*VzData);
end