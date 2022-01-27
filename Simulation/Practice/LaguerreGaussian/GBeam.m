function [Z,x,Grid] = GBeam(A,W0,lambda)
%%Description%%
%Gaussian beam at rl plane. Because it is rotation symmetric, the r ruduces
%to x or y here(choose x)
% A = Amplitude
% W0 = FWHM at z = 0
% lambda = the wavelength of incedent light

%Define Working grid size and resolution.
x = -200:1:200;
Grid = -400:1:400;
%parameters 
Zrl = pi*W0.^2/lambda; %Rayleigh length
W = W0*sqrt(1+(Grid/Zrl).^2);  %FWHM
R = Grid+Zrl^2./Grid;  %Radius of Curvature
Guoy = atan2(Grid,Zrl); %Guoy phase
k = 2*pi/lambda;

%Gauss equaiton: 
idx = 1;
Z = zeros(length(x),length(Grid));
tic
for i = 1:length(Grid)
    t = (x./W(i)).^2; 
    Term1 = exp(-t);
    PTerm1 = exp(-1i*k*Grid(i));
    PTerm2 = exp(-1i*k*(x.^2/(2*R(i))));
    PTerm3 = exp(1i*Guoy(i));
    Z(:,idx) = A*(W0/W(i)).*Term1.*PTerm1.*PTerm2.*PTerm3;
idx = idx + 1
end
time = toc
end