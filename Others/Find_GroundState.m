clc,clear
%=====paremeters=====
dx = 0.1; L = 20;
dw = 0.01*dx^2;   % 虛數時間       
Nx = 2*L/dx +1; x = -L:dx:L;

% use one Rb(85amu), omega = particle energy/hbar, i just intuietively
% choose the quantity. 
hbar = 6.62607004e-34/(2*pi); m = 85*1.66e-27; omega = 1.38e-23/hbar; 
xx = sqrt(hbar/(m*omega))*x;  % applied dimension variable

%%
w = 0;   %虛數時間
psi = rand(1,length(x));
E = 150; EU = 0.5*x.^2; 
% Prabability density can only be positive numbers
P_rho = E - EU;  
minusIndex = (P_rho < 0);
P_rho(minusIndex) = 0;  
G = sum((P_rho))*dx;   % Interaction constant 
% 近似解
solution = (E - EU)/G;
minusIndex = (solution < 0);
solution(minusIndex) = 0;  

figure(1)
yyaxis right
plot(xx,0.5*m*omega.^2*xx.^2)
ylabel('Potential')
xlabel('Position')
yyaxis left

% % ============Initialize video===========
% % loops = 500;
% % F(loops) = struct('cdata',[],'colormap',[]);  % Preallocate frame structre array
% % myVideo = VideoWriter('findGS.avi'); %open video file
% % myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
% % open(myVideo)
% % j = 1;
j = 0;
tic
while j < 10000
    scatter(xx,psi.^2,'.','SizeData',50)
    hold on
    plot(xx,solution)
    ylabel('Probability density')
%     plot(xx,psi.^2)
    drawnow
    hold off
%     pause(0.001)
    psiOld = psi;
    psi = FTCS(dx,dw,psi,x,G, E);
    w = w+dw;
    j = j +1;    
    sum(psi.^2)*dx
    % Animation part
%     ax = gca;
%     ax.Units = 'pixels';
%     pos = ax.Position;
%     ti = ax.TightInset;
%     rect = [-ti(1), -ti(2), pos(3)+ti(1)+ti(3), pos(4)+ti(2)+ti(4)];
%     try
%         F(j) = getframe(gcf);
%         writeVideo(myVideo, F(j));
%     catch
%         fprintf('miss%d\n',j)
%     end
%     j = j+1;
end
% close(myVideo);
toc

function psiNew = FTCS(dx,dt,psiOld,x,G,E_mu)
    n = length(psiOld);
    j = 2:n-1;
    psiNew(j) = -(dt/(-2*dx^2))* (psiOld(j+1) - 2*psiOld(j) + psiOld(j-1))...
    + (0.5 * x(j).^2 .* psiOld(j) + G*abs(psiOld(j)).^2.*psiOld(j) - E_mu*psiOld(j)) * dt/(-1) + psiOld(j);
    % Boundary
    psiNew(1) = -(dt/(-2*dx^2)) * (psiOld(1+1) - 1*psiOld(1) + 0)...
    +(0.5 * x(1).^2 .* psiOld(1) + G*abs(psiOld(1)).^2.*psiOld(1)- E_mu*psiOld(1)) * dt/(-1) + psiOld(1);

    psiNew(n) = -(dt/(-2*dx^2)) * (0 - 1*psiOld(n) + psiOld(n-1))...
    + (0.5 * x(n).^2 .* psiOld(n) + G*abs(psiOld(n)).^2.*psiOld(n)- E_mu*psiOld(n)) * dt/(-1) + psiOld(n);

end