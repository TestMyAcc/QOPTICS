% Disgusting scrpit file to see the property of BEC state 
% and the spartial profile of Laguerre-Gaussian beam
% as quickly as possible.

%!the surface plot represents Amplitude.^2
%% Probability current of BEC & linear momentum density of LG Comparison
clear
dataName = '161220_noLight_400000.mat';
load(fullfile('~','Documents','Lab','Projects','BECdata',dataName));

%% interface

x = mydata.x;
y = mydata.y;
z = mydata.z;
[X,Y,~] = meshgrid(x,y,z);
Nx = size(x,2); Ny = size(y,2); Nz = size(z,2);
dx = diff(x(round(Nx/2):round(Nx/2)+1),1);
dy = diff(y(round(Ny/2):round(Ny/2)+1),1);
dz = diff(z(round(Nz/2):round(Nz/2)+1),1);
psiG = mydata.psiG;
psiE = mydata.psiE;
TFsolG = mydata.TFsolG;
psiGmuArray = mydata.psiGmuArray;
psiEmuArray = mydata.psiEmuArray;
LG = mydata.LG;
Lambda = mydata.Lambda;
m = mydata.m;
%% 


plotwhat = 'density';                
idx_z = [1,21,41,61,81,101,131];     


% plotting parameters
Quiversize = 2;                                                               % how big the arrow
inter = 8;                                                                         % arrow density
margin = 0 ;                                                                      %narrower
range1 = 1+margin:inter:Nx-margin;                               %data range for quiver plot
range2 = 1+margin:Nx-margin;                                        %data range for surface plot



Ncoloum = size(idx_z,2); 
Nrow = 3;
Nplot = Ncoloum*Nrow;  
    
left = 0.02;                                                                   %left margin
bottom = 0.02;                                                             %bottom margin


Row = 0:Nrow-1;
Coloum = 0: Ncoloum-1;
leftPos = round(Coloum*(1-left*(Ncoloum+1))/Ncoloum+left*(Coloum+1),5);
bottomPos = round(Row(end:-1:1)*(1-bottom*(Nrow+1))/Nrow+bottom*(Row(end:-1:1)+1),5);
Width = (1-left*(Ncoloum+1))/Ncoloum ;
Height = (1-bottom*(Nrow+1))/Nrow;

% the center of PsiG along z-axis
figure()  
plot(z,abs(squeeze(psiG(round(Ny/2),round(Nx/2),:))).^2);
hf1 = gcf;
hf1.Units = 'normalized';
hf1.Position = [leftPos(1)  (bottomPos(1)+Height)  (leftPos(end)-leftPos(1)+Width) Height/2];
ax = gca;
ax.Position = [0.05 0.25 0.9 0.7];
axis([z(idx_z(1)),z(idx_z(end)),-inf,inf])

figure('Name',dataName)
hf2 = gcf;
hf2.Units = 'normalized';
hf2.Position = [0 0 1 1];
view([0 90])
k = 0; 
cmode = 'auto';             %% set to manual for the same colorbar limits

zerosmtx = zeros(size(range1,2),size(range1,2));
for i = Row+1

    zi = 0;

    switch i
        %plotting two BEC states plus the light profile subplots
        case 1
            Data = LG;   
%             Data = zeros(Nx,Ny,Nz);
            [Jx,Jy,Jz] = oam(Data,dx,dy,dz,Lambda);
            
        case 2
            Data = psiG;
            [Jx,Jy,Jz] =current(Data,dx,dy,dz,m);

        case 3
            Data = psiE;
            [Jx,Jy,Jz] = current(Data,dx,dy,dz,m);

    end
    
    for j = Coloum+1
        k = k +1;
        zi = zi + 1;
        
        if k > Nplot
            break
        end
        subplot(Nrow,Ncoloum,k)     
        axis([-4 4 -4 4]*3)
      
        
        hold on
        if isequal(d,'phase')
       
            hp = pcolor(X(range2,range2,idx_z(zi)),    Y(range2,range2,idx_z(zi)) , ...  
                angle(squeeze(Data(range2,range2,idx_z(zi)))));   
            set(hp,'ZData',-1+zeros(size(Data(range2,range2,idx_z(zi)))))          %offset of 2Dplot on z
            
        elseif isequal(plotwhat,'density')
            
            hp = pcolor(X(range2,range2,idx_z(zi)),    Y(range2,range2,idx_z(zi)) , ...  
                abs(squeeze(Data(range2,range2,idx_z(zi)))).^2);   
            set(hp,'ZData',-1+zeros(size(Data(range2,range2,idx_z(zi)))))           %offset of 2Dplot on z
        end
      
        quiver3(X(range1,range1,idx_z(zi)),    Y(range1,range1,idx_z(zi)),  zerosmtx, ...   % flux vector
            Jx(range1,range1,idx_z(zi)),    Jy(range1,range1,idx_z(zi)),  Jz(range1,range1,idx_z(zi)),Quiversize,'r')
        
        ax = gca;
        ax.Position = [leftPos(j) bottomPos(i) Width Height];
        ax.View = ([0 90]);
        cmin = min(abs(Data).^2,[],'all');
        cmax = max(abs(Data).^2,[],'all');
        
        if isequal(cmode,'manual')
            caxis manual
            caxis([cmin,cmax]);       
        end
        colorbar()
       
        shading flat
        title(sprintf('z = %.e(m)',z(idx_z(zi))))
        axis tight
       
        if i == Nrow
            xlabel('x(m)'); ylabel('y(m)')
        end 
        hold off
    end
end




%% plotting command reference
% plotting parameters
[X,Y,Z] = meshgrid(x,y,z);
Nx = size(x,2); Ny = size(y,2); Nz = size(z,2);
Quiversize = 2;                                                               % how big the arrow
inter = 6;                                                                         % arrow density
interz = 8;               
margin = 0 ;                                                                      %narrower
rangexy = 1+margin:inter:Nx-margin;                                      
rangez = 1+margin : interz:Nz-margin;
centerX = ceil(Nx/2);
centerY = ceil(Ny/2);
centerZ = ceil(Nz/2);
%%




plot(x,abs(squeeze(TFsolG(centerY,:,centerZ))).^2*dx*dy*dz,'-*','LineWidth',1);  
plot(x,abs(squeeze(psiG(centerY,:,centerZ))).^2*dx*dy*dz,'-*','MarkerSize',1);              
plot(x,abs(squeeze(psiE(centerY,:,centerZ))).^2*dx*dy*dz,'-*','MarkerSize',2);              
plot(x,abs(squeeze(psiG(centerY,:,centerZ))).^2,'-*','MarkerSize',2);              

plot(x,angle(squeeze(psiE(centerY,:,centerZ))),'-*','MarkerSize',2);              

surf(x,y,squeeze(psiE(:,:,centerZ)).^2*dx*dy*dz,'EdgeColor','none')
surf(Gridxy,Gridxy,abs(squeeze(LGdata(:,:,61)).^2),'EdgeColor','none')
surf(Gridxy,Gridxy,angle(squeeze(LGdata(:,:,61))),'EdgeColor','none')

plot(Gridxy,abs(squeeze(LGdata(121,:,61))).^2,'-*','MarkerSize',1);    

[Jx,Jy,Jz] =current(psiG,dx,dy,dz,m); 
quiver3(X(rangexy,rangexy,rangez),... 
    Y(rangexy,rangexy,rangez), ... 
    Z(rangexy,rangexy,rangez), ... 
    Jx(rangexy,rangexy,rangez), ... 
    Jy(rangexy,rangexy,rangez), ... 
    Jz(rangexy,rangexy,rangez),1.5)
