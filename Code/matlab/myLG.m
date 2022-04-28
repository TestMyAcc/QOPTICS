function LG = myLG(A,W0,Lambda,z,x,y,L,P)
%MYLG parallel loop version drawing the LG beam 
% spatial profile with determined waist and wavelength 
% along z-axis. Data is stored in specified path.
%
% MYLG(A,W0,Lambda,z,x,y,L,P)
% Save data at ~.
% A: Amplitude of field
% W0 : Beam Radius at z=0
% Lambda: Wavelength of LG beam.
% z,x,y: coordinate array
% L: the azumithal mode number. 
% P: the radial mode number.
% See also LaguerreGauss



%Beam parameters

                                           
Zrl = pi*W0.^2/Lambda;                                        %Rayleigh length
W= W0*sqrt(1+(z./Zrl).^2);  
R = z + Zrl^2./z;
Guoy = (abs(L)+2*P+1).*atan2(z,Zrl); 

Nx = size(x,2);
Ny = size(y,2);
Nz = size(z,2);
[X,Y] = meshgrid(x,y);
LG = zeros(Nx,Ny,Nz);

tstart = tic;
ticBytes(gcp)
parfor k = 1:Nz
%     tloop = tic
    LG(:,:,k) = computeLG(X,Y,z(k),W(k),R(k),Guoy(k),Lambda,L,P,W0)
%     time = toc(tloop)
end
LG = A*LG/max(abs(LG),[],'all'); 


tocBytes(gcp);
times = toc(tstart);
fprintf('Beam waist W:%g, Wavelength lambda:%g  Simulation time:%.00005f(s)\n', ...
    [W0,Lambda, times])

if ispc
    userDir = winqueryreg('HKEY_CURRENT_USER',...
        ['Software\Microsoft\Windows\CurrentVersion\' ...
         'Explorer\Shell Folders'],'Personal');
else
    userDir = char(java.lang.System.getProperty('user.home'));
end


dlmt = '_';
fname = strcat(num2str(Nx),dlmt,num2str(Ny),dlmt,num2str(Nz), ... 
    'W0',num2str(W0),dlmt,'Lambda',num2str(Lambda),dlmt,'L&P',num2str(L),num2str(P),'.mat');
fprintf('save at %s\n', fullfile(userDir,fname))
save(fullfile(userDir,fname));

     

end

function LG = computeLG(X,Y,z,W,R,Guoy,Lambda,L,P,W0)
    r = squeeze(sqrt(X.^2 + Y.^2)); 
    Phi = atan2(Y,X);
    AL =((sqrt(2)*r./W)).^abs(L);
    ALpoly =laguerreL(P,abs(L),2*(r./W).^2);
    AGauss = exp(-(r./W).^2);
    Ptrans1 = exp(-1i*(2*pi/Lambda)*r.^2./(2*R));
    Ptrans2 = exp(-1i*L.*Phi);
    PGuoy = exp(1i.*Guoy);
    LG = (W0/W).*AL.*ALpoly.*AGauss.*Ptrans1.*Ptrans2.*PGuoy;
    if L == 0 && P == 0
        Plong = exp(-1i*((2*pi/Lambda)*z - Guoy));
        LG = (W0/W).*AGauss.*Ptrans1.*Ptrans2.*Plong;
    end
end




