clc,clear
%%Initialize video%%
OutputVideo();
%=paremeters=%
dx = 0.1; dy = dx; dw = 0.0001*dx^2;  % 虛數時間  % dw < dx^2 3order  
L = 20;
x = -L:dx:L; y = x;     
Nx = 2*L/dx +1; 

[X,Y] = meshgrid(x,y);
E = 100; EU = 0.5*(X.^2+Y.^2);
psi = rand(length(x),length(y));

% use one Rb(85amu), omega = particle energy/hbar, I just intuietively
% choose these quantity. 
hbar = 6.62607004e-34/(2*pi); m = 85*1.66e-27; omega = 1.38e-23/hbar; 
XX = sqrt(hbar/(m*omega))*X;  % applied dimension variable
YY = sqrt(hbar/(m*omega))*Y;
%=Prabability density can only be positive numbers=%
P_rho = E - EU;  
minusIndex = (P_rho < 0);
P_rho(minusIndex) = 0;  

G = sum(P_rho*dx*dy,'all');   % Interaction constant 




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
M1 = dw/(2*dx.^2)*M; M2 = dw/(2*dy.^2)*M;
%%
w = 0; j = 0; jj = 1;
Cdata = zeros(size(psi));
view(3)
P = 10;

while jj<100
 
    if ~mod(j,101)
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
    psiOld = psi;
    psi = M1*psi + psi*M2;    % Ek term (x-direction and y -direction)
    psi = psi - psiOld.*(EU + G*(psiOld.^2) - E)*dw + psiOld;  %leftover term
    w = w+dw;
    j = j + 1;
    P = sum(psi.^2*dx*dy,'all')

end
OutputVideo(jj,2)
