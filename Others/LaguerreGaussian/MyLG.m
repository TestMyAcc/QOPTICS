
function [Z,Grid]= MyLG(P,L,W0,Zaxis,plotornot)
%%Description%%
% Revised version based on LG beam.m in the same directory, drawing the LG
% beam of determined waist and wavelength on specified z plane. 
% P = P mode
% L = L mode
% W0 = FWHM at z = 0
% Zaxis = Z*axis (mu_m)
% plotornot = 'yes' or 'no'

switch plotornot
    case 'yes'
        isplot = 1;
    case 'no'
        isplot = 0;
end

%Define beam parameters(spatial:mu_m)
A = 30;          %Amplitude 
lambda = 0.78;    %Wavelength
Zrl = pi*W0.^2/lambda; %Rayleigh length
W = W0*sqrt(1+(Zaxis/Zrl).^2);  %FWHM
R = Zaxis+Zrl^2./Zaxis;  %Radius of Curvature
Guoy = (abs(L)+2*P+1)*atan2(Zaxis,Zrl); %Guoy phase

tic
%Define Working grid size and resolution.
Grid= -4:0.01:4;
[X,Y] = meshgrid(Grid);

%Laguerre-Gauss equaiton: 
r = sqrt(X.^2 + Y.^2); Phi = atan2(Y,X);
t = (r./W).^2; LPhi = L.*Phi;
Term1 =((sqrt(2)*r./W)).^abs(L);
Term2 =laguerreL(P,abs(L),2.*t);
Term3 = exp(-t);
% C = sqrt(2*factorial(P)/(pi*factorial((P+abs(L))))); %Normalize
%Phase term
PTerm1 = exp(-1i*(2*pi/lambda)*r.^2./(2*R));
PTerm2 = exp(-1i*LPhi);
PTerm3 = exp(1i.*Guoy);
Z = A*(W0/W).*Term1.*Term2.*Term3.*PTerm2.*PTerm1.*PTerm3;

RealZ = real(Z);
ImagZ = imag(Z);
Phase = angle(Z);
Intensity = abs(Z);
times = toc;
fprintf('W:%.5f,  Simulation time:%.5f(s)\n',[W, times])

%Plots
%%
if isplot
tic
figure('Name',strcat('LG',num2str(P),',',num2str(L),'Z',num2str(Zaxis)),'Units', 'normalized', 'Position', [0.1 0.1 0.7 0.7])
subplot(2,2,1)
xlabel('\itrealE(x,y)')
surface(X,Y,RealZ)
colormap(jet)
colorbar()
shading interp

subplot(2,2,2)
xlabel('\itimagE(x,y)')
surface(X,Y,ImagZ)
shading interp

subplot(2,2,3)
xlabel('|E(x,y)|\^{2}')
% xlim([Grid(1) Grid(end)]);ylim([Grid(1) Grid(end)]);
surface(X,Y,Intensity)
colorbar()
shading interp 

subplot(2,2,4)
xlabel('Phase')
% xlim([Grid(1) Grid(end)]);ylim([Grid(1) Grid(end)]);
surface(X,Y,Phase)
colorbar('Ticks',[-pi -pi/2 0 pi/2 pi], 'Ticklabels',{'-\pi', '-\pi/2', '0', '\pi/2', '\pi'})
shading interp
timep = toc;
fprintf('Plotting time:%.5f(s)\n',timep)
end
end

