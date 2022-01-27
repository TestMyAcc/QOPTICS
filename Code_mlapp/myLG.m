function myLG(A,W0,Lambda,Gridz,Gridxy,L,P)
%MYLG parallel loop version drawing the LG beam 
% spatial profile with determined waist and wavelength 
% along z-axis. Data is stored in specified path.
%
% LGdata = MYLG(A,W0,Lambda,Gridz,Gridxy,L,P)
% Save data at displayed path.
% A: Amplitude of field
% W0 : Beam Radius at z=0
% Lambda: Wavelength of LG beam.
% Gridz : the coordinate point of  the z point.
% Gridxy: 1-D array for x and y cooridnate
% L: the azumithal mode number. 
% P: the radial mode number.
% See also LaguerreGauss

Nx = size(Gridxy,2);
Ny = size(Gridxy,2);
Nz = size(Gridz,2);
[X,Y] = meshgrid(Gridxy);
LGdata = zeros(Nx,Ny,Nz);

%Beam parameters
                                           
Zrl = pi*W0.^2/Lambda;                                        %Rayleigh length
W= W0*sqrt(1+(Gridz./Zrl).^2);  
R = Gridz + Zrl^2./Gridz;
Guoy = (abs(L)+2*P+1).*atan2(Gridz,Zrl); 


tstart = tic;
ticBytes(gcp)
parfor k = 1:Nz
%     tloop = tic
    LGdata(:,:,k) = computeLG(X,Y,Gridz(k),W(k),R(k),Guoy(k),Lambda,L,P,W0)
%     time = toc(tloop)
end
LGdata = A*LGdata/max(abs(LGdata),[],'all'); 


tocBytes(gcp);
times = toc(tstart);
fprintf('Beam waist W:%g, Wavelength lambda:%g  Simulation time:%.00005f(s)\n', ...
    [W0,Lambda, times])

datadir = '~/Documents/Lab/Projects/LGBeamdata/';
dlmt = '_';
fname = strcat(num2str(Nx),dlmt,num2str(Ny),dlmt,num2str(Nz), ... 
    'W0',num2str(W0),dlmt,'Lambda',num2str(Lambda),dlmt,'L&P',num2str(L),num2str(P),'.mat');
save(strcat(datadir,fname),'LGdata','W0','Lambda','Gridxy','Gridz','L','P');
     

end

function LGdata = computeLG(X,Y,z,W,R,Guoy,Lambda,L,P,W0)
    r = squeeze(sqrt(X.^2 + Y.^2)); 
    Phi = atan2(Y,X);
    AL =((sqrt(2)*r./W)).^abs(L);
    ALpoly =laguerreL(P,abs(L),2*(r./W).^2);
    AGauss = exp(-(r./W).^2);
    Ptrans1 = exp(-1i*(2*pi/Lambda)*r.^2./(2*R));
    Ptrans2 = exp(-1i*L.*Phi);
    PGuoy = exp(1i.*Guoy);
    LGdata = (W0/W).*AL.*ALpoly.*AGauss.*Ptrans1.*Ptrans2.*PGuoy;
    if L == 0 && P == 0
        Plong = exp(-1i*((2*pi/Lambda)*z - Guoy));
        LGdata = (W0/W).*AGauss.*Ptrans1.*Ptrans2.*Plong;
    end
end




