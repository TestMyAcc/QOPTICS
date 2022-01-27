clc,clear
% % ============Initialize video===========
% loops = 5000;
% F(loops) = struct('cdata',[],'colormap',[]);  % Preallocate frame structre array
% myVideo = VideoWriter('findGS.avi'); %open video file
% myVideo.FrameRate = 8;  %can adjust this, 5 - 10 works well for me
% open(myVideo)

%=====paremeters=====
dx = 0.1; dy = dx; dw = 0.00001*dx^2;  % 虛數時間 w = -it % dw < dx^2 3order  
L = 20;
x = -L:dx:L; y = x;     
Nx = 2*L/dx +1; Ny = Nx; 
[X,Y] = meshgrid(x,y);
E = 100; EU = 0.5*(X.^2+Y.^2);


% =====Initialize wave function with phase=====
psi = zeros(Nx,Ny);
perimeter = size(psi,1);
for iii =  1:perimeter
    for jjj = 1:perimeter
        Amp = rand(1,1);
        corx = jjj-(1+perimeter)/2; cory = iii-(1+perimeter)/2;
        psi(iii,jjj) = Amp*exp(1i*atan2(cory,corx));
%             if corx > 0 && cory > 0
%                 psi(iii,jjj) = Amp*abs(cos(atan(cory/corx))) + Amp*1i*abs(sin(atan(cory/corx)));
%             elseif corx < 0 && cory > 0
%                 psi(iii,jjj) = -Amp*abs(cos(atan(cory/corx))) + Amp*1i*abs(sin(atan(cory/corx)));
%             elseif corx < 0 && cory < 0
%                 psi(iii,jjj) = -Amp*abs(cos(atan(cory/corx))) - Amp*1i*abs(sin(atan(cory/corx)));
%             elseif corx > 0 && cory < 0
%                 psi(iii,jjj) = Amp*abs(cos(atan(cory/corx))) - Amp*1i*abs(sin(atan(cory/corx))); 
%             elseif corx == 0 && cory == 0
%                 psi(iii,jjj) = Amp;
%             end
    end
end

figure(1)
theta = angle(psi);
surface(X,Y,theta,'EdgeColor','none'); 
view([0 90]);colorbar;


%=====Dimensionless variables=====
% use one Rb(85amu), omega = particle energy/hbar, I just intuietively
% choose these quantities. 
hbar = 6.62607004e-34/(2*pi); m = 85*1.66e-27; omega = 1.38e-23/hbar; 
XX = sqrt(hbar/(m*omega))*X;  % applied dimension variable
YY = sqrt(hbar/(m*omega))*Y;

% Prabability density can only be positive numbers
P_rho = E - EU;  
minusIndex = (P_rho < 0);
P_rho(minusIndex) = 0;  
G = sum(P_rho*dx*dy,'all');   % Interaction constant 

% TF近似解
solutionP_rho = (E - EU)/G;
minusIndex = (solutionP_rho < 0);
solutionP_rho(minusIndex) = 0;
% TF with phase
% for iii =  1:perimeter
%     for jjj = 1:perimeter
%         corx = jjj-(1+perimeter)/2; cory = iii-(1+perimeter)/2;
%         if solutionP_rho(iii,jjj) == 0
%             psi(iii,jjj) = 0;
%         else         
%             psi(iii,jjj) = sqrt(solutionP_rho(iii,jjj))*exp(1i*atan2(cory,corx));
%         end
%     end
% end



%%
% yyaxis right
% plot(x,EU)
% ylabel('Potential')
% xlabel('Position')
% yyaxis left

%%
% =====Create matrix=======
% 方程動能項的離散部份決定是tridiagonal matrix
M = gallery('tridiag',Nx,1,-2,1);   %gallery 叫出的是sparce matrix
M(1,1) = -1; M(end,end) = -1;
M1 = dw/(2*dx.^2)*M; M2 = dw/(2*dy.^2)*M;
%%
figure(2)
w = 0; j = 0; jj = 1;
Cdata = zeros(size(psi));
hsurf = surface(XX,YY,abs(psi).^2,'EdgeColor','none');
view(3)
P = 10;
tic

%%
while ~isnan(P)
    tic
    psiOld = psi;
    psi = M1*psi + psi*M2;    % Ek term (x-direction and y -direction)
    psi = psi - psiOld.*(EU + G*(abs(psiOld).^2) - E)*dw + psiOld;  %leftover term
    w = w+dw; j = j + 1
    P = sum(abs(psi).^2*dx*dy,'all');
    tdata = toc;

    if ~mod(j,300)
          set(hsurf, 'ZData',abs(psi).^2);
%         subplot(1,2,1)
%         plot3(XX,YY,abs(psi).^2,'*','MarkerSize',0.05);
%         hold on
%         surface(XX,YY,solution,Cdata,'Edgecolor','none');
%         xlabel('position(m)'); ylabel('position(m)'); zlabel('Probability density')
        drawnow  
%         hold off
%         subplot(1,2,2)
% 
%         plot(XX(201,1:end),psi(201,1:end).^2);
%         hold on
%         plot(XX(201,1:end),solution(201,1:end));
%         drawnow  
%         hold off
%         subplot(1,3,3)
%         plot3(XX,YY,psi.^2);

    %        ======Animation part=====
%         ax = gca;
%         ax.Units = 'pixels';
%         pos = ax.Position;
%         ti = ax.TightInset;
%         rect = [-ti(1), -ti(2), pos(3)+ti(1)+ti(3), pos(4)+ti(2)+ti(4)];
% %         try
%             F(jj) = getframe(gcf);
%             writeVideo(myVideo, F(jj));
%             jj = jj + 1;
%             %         catch ME
% %             fprintf('miss%d\n',j)
% %             throw(ME)
% %         end
   
    end
end
Tdata = toc;
% close(myVideo)