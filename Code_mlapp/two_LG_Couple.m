%% Grid paremeters 

Nx = 251; Nz = 131;                                            
Ny = Nx;
Lx = 4; Ly = 4; Lz = 5;                                               %single side length                           

dx = 2*Lx/(Nx-1); 
dy = 2*Ly/(Ny-1); dz = 2*Lz/(Nz-1);  
x = -Lx:dx:Lx; y = -Ly:dy:Ly;   z = -Lz:dz:Lz; 

dw = 1e-6;                                                                   %condition for converge : <1e-3*dx^2                                               

%% BEC parameters
As = 5.82e-09;
Nbec = 10000;
Rabi = 1000;
hbar = 1.054571800139113e-34;
m = 1.411000000000000e-25;
omegaXY = 2000;
omegaZ = 2000;



%%
[X,Y,Z] = meshgrid(x,y,z);

%Lagrangian multiplier
psiGmu = (15*Ggg  /  (  16*pi*sqrt(2))  ).^  (2/5);         
psiEmu = (15*Ggg  /  (  16*pi*sqrt(2))  ).^  (2/5);                                                  
init_psiGmu = psiGmu;
init_psiEmu = psiEmu;


TFsolG = (psiGmu-Epot)/Ggg;                               
whereMinus = (TFsolG < 0);
TFsolG(whereMinus) = 0;  
TFsolG = sqrt(TFsolG);



%% Array prelocation 

j = 0; J = 0;                   
nj = 2000000; stepJ = 100000 ;   %number of runs & steps to update bound.

fprintf(" Total runs %d steps , update every %d steps\n",...
    nj, stepJ)

psiGmuArray = zeros(1, ceil(nj/stepJ) + 1);                 %Energy of groundstate every cetain step, 
psiGmuArray(1) = init_psiGmu;                                   %Fitst is the energy of T.F.          
psiEmuArray = zeros(1, ceil(nj/stepJ) + 1);
psiEmuArray(1) = init_psiEmu;

LG = gpuArray(LG);
psiE = gpuArray(zeros(Nx,Ny,Nz));  
psiG = gpuArray(TFsolG);

mydata = struct();                                                    

%% main part

storingData =  ...
    {... 
    'x', 'y', 'z', 'LG','Lambda','W0', 'psiG', 'psiE', 'm', 'TFsolG',  ...
    'init_psiGmu', 'init_psiEmu',  'psiGmuArray' , 'psiEmuArray', ...
    'time', 'j', 'J','stepJ'};

datapath = fullfile('~','Documents','Lab','Projects','BECdata',[filename,'_%d.mat']); 

tstart = tic;
while j ~= nj  
    
    j = j+1;

    if mod(j, 4*stepJ) == 0 || j == nj
        temp = ws2struct();                     %data of current workspace
        varname = fieldnames(temp);
        for k = 1:size(varname,1)
            if any(strcmp(varname{k}, storingData))
            %converted to GPU
                mydata.(varname{k}) = gather(temp.(varname{k})); 
            end
        end
        disp( ['save as ',  sprintf(datapath, j)] );
        save( sprintf(datapath, j), 'mydata' );    
        clear temp
    end

    if mod(j,stepJ) == 0    
    % chemical potential convergent condition
        Nfactor = Normalize(psiG,psiE,dx,dy,dz);
        J = J + 1 ;
        psiGmuArray(J+1) = gather(sum(conj(psiG).* ... 
            (- 0.5 * 2*ndims(psiG)*del2(psiG, dx, dy, dz) + ...                   %Hamiltonian of Ground state
            (Epot + Ggg*abs(psiG).^2 + Gge*abs(psiE).^2) .* psiG ) ... 
            *dx*dy*dz,'all'));
        psiEmuArray(J+1) = gather(sum(conj(psiE).* ... 
            (- 0.5 * 2*ndims(psiE)*del2(psiE, dx, dy, dz) + ...                     %Hamiltonian of Excited state
            ( Epot  + Gee*abs(psiE).^2 + Geg*abs(psiG).^2).*psiE)... 
            *dx*dy*dz,'all'));
        time(J) = toc(tstart);
    end
    
end

function Nfactor = Normalize(psiG,psiE,dx,dy,dz)
        SumPsiG = sum( abs(psiG).^2*dx*dy*dz,  'all' );
        SumPsiE = sum( abs(psiE).^2*dx*dy*dz,  'all' );
        Nfactor = SumPsiG +  SumPsiE;  
end