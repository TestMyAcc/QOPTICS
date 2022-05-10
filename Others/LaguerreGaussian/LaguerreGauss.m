
function Z = LaguerreGauss(P,L,A,W)
%%Description%%
% A simple function which plots the spatial, phase and
% intensity profiles of Laguerre-Gauss(P,L) modes. 
% P = P mode
% L = L mode
% A = Amplitude
% W = FWHM

%Define working grid size and resolution.
Grid= -4:0.1:4;
[X,Y] = meshgrid(Grid);

%Laguerre-Gauss equaiton: 
%(ref: N. Hodgson, 'Laser Resonators and Beam Propagation'.(Pg 222)) 
t = (X.^2 + Y.^2)/(W^2);
Phi = L.*atan2(Y,X);
Term1 =((sqrt(2)*sqrt(X.^2 + Y.^2)/W)).^L;
Term2 =laguerreL(P,L,2.*t);
Term3 = exp(-t);
Term4 = exp(1i*Phi);
Z = A.*Term1.*Term2.*Term3.*Term4;
Spatial = real(Z);
Phase = angle(Z);
Intensity = abs(Z);


%Plots
figure('Name',strcat('LG',num2str(P),',',num2str(L)),'Renderer', 'painters', 'Position', [125 125 1300 300])
subplot(1,3,1)
xlabel('E(x,y)')
xlim([1 length(Grid)]);ylim([1 length(Grid)]);
surface(Spatial)
colormap(jet)
colorbar()
shading interp 

subplot(1,3,2)
xlabel('|E(x,y)|\^{2}')
xlim([1 length(Grid)]);ylim([1 length(Grid)]);
surface(Intensity)
colormap(hot)
colorbar()
shading interp 

subplot(1,3,3)
xlabel('Phase')
xlim([1 length(Grid)]);ylim([1 length(Grid)]);
surface(Phase)
colormap(hot)
colorbar('Ticks',[-pi -pi/2 0 pi/2 pi], 'Ticklabels',{'-\pi', '-\pi/2', '0', '\pi/2', '\pi'})
shading interp 

end

