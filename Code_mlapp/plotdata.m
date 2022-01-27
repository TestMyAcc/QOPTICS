%% load data to plotting

clear
%%
disp('loading....')
load(fullfile('~','Documents','Lab','Projects','BECdata','161220_noLightnoUpdate_2000000.mat'));

% load(fullfile('~','Documents','Lab','Projects','LGBeamdata','251_251_131W01e-06_Lambda2e-06_L&P00.mat'));

x = mydata.x;
y = mydata.y;
z = mydata.z;

Nx = size(x,2); Ny = size(y,2); Nz = size(z,2);
psiG = mydata.psiG;
psiE = mydata.psiE;
LG = mydata.LG;
Lambda = mydata.Lambda;
m = mydata.m;

% load(fullfile('~','Documents','Lab','Projects','BECdata','Parameters','Light_BEC.mat')); 
% psiG = psiG*unit.^-(3/2);
% psiE = psiE*unit.^-(3/2);
% LG = omegaZ*LG;
% x = x*unit;
% y = y*unit;
% z = z*unit;

dx = diff(x(round(Nx/2):round(Nx/2)+1),1);
dy = diff(y(round(Ny/2):round(Ny/2)+1),1);
dz = diff(z(round(Nz/2):round(Nz/2)+1),1);

%%
% data = struct('x',x,'y',y,'z',z(99:121),'psiE',psiE(:,:,99:121));
data = struct('x',x,'y',y,'z', z, 'LG',LG);
profile on 
scanning(data, 'phase', 'z', 1, 'Margin',60,'Inter',1,'Cmode','auto','Quiversize',3)
profile viewer

%%
% %% test direction 
% % clear
% Nx = 300;
% Ny = 300;
% Nz = 100;
% Lx = 5;
% Ly = 5;     
% Lz = 5;
% x = linspace(-Lx,Lx,Nx);
% y = linspace(-Ly,Ly,Ny);
% z = linspace(-Lz,Lz,Nz);
% dx = diff([x(1),x(2)]);
% dy = diff([y(1),y(2)]);
% dz = diff([z(1),z(2)]);
% 
% [X,Y,Z] = meshgrid(x,y,z);
% 
% % U =  exp(-(X.^2+Y.^2)/100);
% U =  X.^2 + Y.^2  + Z.^3;
% 
% LU = gather(mydel2(gpuArray(U),dx,dy,dz));
% % LU = 6*del2(U,dx,dy,dz);
% LUdiff = LU-4;
% error = max(LUdiff,[],'all');
% errorAvg = sum(abs(LU-4),'all')/(Nx*Ny*Nz);
% LUerror = LUdiff>error/5;
% 
% fprintf('Error Max: %e\n', error);
% fprintf('Error Avg: %e\n', errorAvg);
% 
% % plot(z,squeeze(LU(200,200,:)))
