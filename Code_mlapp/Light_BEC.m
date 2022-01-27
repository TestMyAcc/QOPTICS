%Setting parameters

%% Grid paremeters 

Nx = 251; Nz = 131;                                            
Ny = Nx;
Lx = 4; Ly = 4; Lz = 5;                                               %single side length                           

dx = 2*Lx/(Nx-1); 
dy = 2*Ly/(Ny-1); dz = 2*Lz/(Nz-1);  
x = -Lx:dx:Lx; y = -Ly:dy:Ly;   z = -Lz:dz:Lz; 

dw = 1e-6;                                                                   %condition for converge : <1e-3*dx^2                                               
nj = 2000000; stepJ = 100000;                                    % times of runs and steps between updates normalized energy 

%% BEC parameters

As = 5.82e-09;
Nbec = 10000;
Rabi = 1000;
hbar = 1.054571800139113e-34;
m = 1.411000000000000e-25;
omegaXY = 2000;
omegaZ = 2000;
unit = 1.222614572474304e-06;

parfile = '161020_Light_BEC.mat';                                      
parPath = fullfile('~','Documents','Lab','Projects','Parameters',parfile);

% save(parPath)                          %Save parameters

%%  LG parameters 

LGfile = '251_251_131W01e-06_Lambda2e-06_L&P10_Half.mat';
lightPath = fullfile('~','Documents','Lab','Projects','LGBeamdata',LGfile);

%%

gpuDevice(1);
computeBEC('211220_noLightnoUpdate', parPath, lightPath);
