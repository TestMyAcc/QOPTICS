function [Lx,Ly,Lz] = oam(Data,dx,dy,dz,Lambda)
%Orbital angular momentum of the light(with Dimension)
%Data: the light
%dx,dy,dz: resolution
%Lambda: wavelengh

    epsilon  = 8.854187817e-12; 
    c = 299792458;
    angFreq = 2*pi*c/Lambda;
    [grax,gray,graz] = gradient(Data,dx,dy,dz);
    % [grax_C,gray_C,graz_C] = gradient(conj(LGdata),dx,dy,dz);
    Lx =  1i*angFreq*epsilon/2.*(Data.*conj(grax) - conj(Data).*grax); 
    Ly = 1i*angFreq*epsilon/2.*(Data.*conj(gray) - conj(Data).*gray);
    Lz = 1i*angFreq*epsilon/2.*(Data.*conj(graz) - conj(Data).*graz) + angFreq*2*pi/Lambda*epsilon*abs(Data).^2*1;  % unit vector along z = 1
end