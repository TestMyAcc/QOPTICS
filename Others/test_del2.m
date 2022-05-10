%% test my mydel2.m
% approximately 2x speedup on lab machine.
Lx = 3;
Ly = 2;
Lz = 2 ;                               
Nx = 300;
Nz = 100;
Ny = 300;
x = linspace(-Lx,Lx,Nx);
y = linspace(-Ly,Ly,Ny);
z = linspace(-Lz,Lz,Nz);
dx = 2*Lx/(Nx-1);
dy = 2*Ly/(Ny-1);
dz = 2*Lz/(Nz-1);


[X,Y,Z] = meshgrid(x,y,z);


% F1 = sin(Z);
% del2_F1 = mydel2(F1,dx,dy,dz);
% F1_sol = -sin(Z);

% F2 = X.^2 + Y.^2 + Z.^2;
% del2_F2 = mydel2(F2,dx,dy,dz);
% F2_sol = 6*ones(Ny,Nx,Nz);

F3 = (cos(X) + sin(Y)).^2 + Y.^2  + Z.^2;
tic_start = tic;
cpu_start = cputime;
del2_F3 = mydel2(F3,dx,dy,dz);
tic_elapsed = toc(tic_start) 
cpu_elapsed = cputime - cpu_start;

btic_start = tic;
bcpu_start = cputime;
builtin_del2_F3 = 6*del2(F3,dx,dy,dz);
btic_elapsed = toc(btic_start);
bcpu_elapsed = cputime - bcpu_start;


% F3_sol = -2*( 2*cos(X).*sin(Y)+cos(2*X)+sin(Y).^2 - cos(Y).^2-1 );
% subplot(2,1,1)
% surf(del2_F3(:,:,round(Nz/2)), 'EdgeColor','none');
% xlabel('x');
% ylabel('y');
% grid off
% axis tight
% % zlim([0,10])
% subplot(2,1,2)
% surf(builtin_del2_F3(:,:,round(Nz/2)), 'EdgeColor','none');
% xlabel('x');
% ylabel('y');
% grid off
% axis tight
% % zlim([0,10])

%%
x = 0:2:6;
y = 0:1:6;
z = 0:3:6;
[X,Y,Z] = meshgrid(x,y,z);
F = X.^2 + Y.^2 + Z.^2;

%%
F = magic(2);
subplot(2,1,1)
surf(F)
subplot(2,1,2)
surf(peaks)