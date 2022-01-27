clc,clear
%Simulation parameters
dz = 0.1; Lz = 10; z = -Lz:dz:Lz; Nz = size(z,2);
Grid= -4:0.1:4; NGrid = size(Grid,2);
E = zeros(NGrid,Nz);  
idxaxis = find(Grid==0); %Find the x axis of each z slice
W0 = 1;
tic
for i = 1:Nz
% assemble each z slice with respect to x-axis
     temp = MyLG(0,0,W0,z(i));
     E(:,i) = temp(idxaxis,:);
end
time = toc;
%%

figure('Name',sprintf('real part'))
xlabel('z(\it\mum)'); ylabel('r(\it\mum)')
surface(z,Grid,real(E))
colormap(jet)
colorbar(); view([0 0 90]);
shading interp 

figure('Name',sprintf('imag part'))
xlabel('z(\it\mum)'); ylabel('r(\it\mum)')
% xlim([Grid(1) Grid(end)]);ylim([Grid(1) Grid(end)]);
surface(z,Grid,imag(E))
colormap(jet)
colorbar(); view([0 0 90]);
shading interp 