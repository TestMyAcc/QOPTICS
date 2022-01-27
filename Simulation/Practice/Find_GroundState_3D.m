clc,clear
%%Initialize video%%
% OutputVideo();
%=paremeters=%
dx = 0.1; dy = dx; dz = 0.1;
dw = 0.01*dx^2;  % 虛數時間  % dw < dx^2 3order  
Lxy = 20; Lz = 6;
x = -Lxy:dx:Lxy; y = x;   z = -Lz:dz:Lz; 
Nx = 2*Lxy/dx +1; Ny = Nx; Nz = 2*Lz/dz +1; 
zidx = 1:Nz;  xidx = 1:Nx;

[X,Y,Z] = meshgrid(x,y,z);
E = 10; EU = 0.5*(X.^2+Y.^2+Z.^2);
psi = rand(length(x),length(y),length(z));

% use one Rb(85amu), omega = particle energy/hbar, I just intuietively
% choose these quantity. 
hbar = 6.62607004e-34/(2*pi); m = 85*1.66e-27; omega = 1.38e-23/hbar; 
XX = sqrt(hbar/(m*omega))*X;  % applied dimension variable
YY = sqrt(hbar/(m*omega))*Y;
ZZ = sqrt(hbar/(m*omega))*Z;
%=Prabability density can only be positive numbers=%
P_rho = E - EU;  
minusIndex = (P_rho < 0);
P_rho(minusIndex) = 0;  

G = sum(P_rho*dx*dy*dz,'all');   % Interaction constant 




%=近似解=%
solution = (E - EU)/G;
minusIndex = (solution < 0);
solution(minusIndex) = 0;  

% yyaxis right
% plot(x,EU)
% ylabel('Potential')
% xlabel('Position')
% yyaxis left

%%
%=Create matrix=%
%方程動能項的離散部份決定是tridiagonal matrix
M = gallery('tridiag',Nx,1,-2,1);   %gallery 叫出的是sparce matrix
M(1,1) = -1; M(end,end) = -1; 
M1 = dw/(2*dx.^2)*M; M2 = dw/(2*dy.^2)*M; M3 = dw/(2*dz.^2)*M;

w = 0; j = 0; jj = 1;  %j = loop idx, jj = frame idx
Cdata = zeros(size(psi));
P = 10;

tstart = tic;
tdata = zeros(1,1000);
%%
while j < 10000
 
    if ~1
%         surface(X,Y,psi.^2,'EdgeColor','none'); %不會刷新
        subplot(1,2,1)
        plot3(XX,YY,psi.^2,'*','MarkerSize',0.05);
        hold on
        surface(XX,YY,solution,Cdata,'Edgecolor','none');
        xlabel('position(m)'); ylabel('position(m)'); zlabel('Probability density')
        drawnow  
        hold off
        subplot(1,2,2)
        plot(XX(201,1:end),psi(201,1:end).^2);
        hold on
        plot(XX(201,1:end),solution(201,1:end));
        drawnow  
        hold off
%         subplot(1,3,3)
%         plot3(XX,YY,psi.^2);
        OutputVideo(jj)
        jj = jj+1;
    end
   
    tic
    psiOld = psi;
    psiXY = zeros(Nx,Ny,Nz); psiZ =psiXY;
    for kz = zidx
        psiXY(:,:,kz) = M1*psiOld(:,:,kz) + psiOld(:,:,kz)*M2;    % Ek term (for y-direction and x-direction)       
    end
    for kx = xidx
        psiZslice = squeeze(psiOld(kx,:,:)); %remove useless dimension
        psiZ(kx,:,:) = M1*psiZslice;    % Ek term (for z direction)       
    end
    psi = psiXY + psiZ - psiOld.*(EU + G*(psiOld.^2) - E)*dw + psiOld;  %leftover term
    w = w+dw;
    j = j + 1;
   P = sum(psi.^2*dx*dy*dz,'all')
   tdata(j) = toc;
end
tdata(end+1) = toc(tstart);
% OutputVideo(jj,2)

