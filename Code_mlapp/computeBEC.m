function computeBEC(filename, parPath, lightPath)
%COMPUTEBEC produces the result of BEC
%   interacting  with Laguerre-Gaussian beams.
% 
%   It currently produces converged steady state.
% 
%   COMPUTEBEC(filename, parPath, lightPath)
%   Wavefunction and other quantities are stored 
%   as a scalar structure  in the path with given filename
%   every  certian time steps.
% 
%   parPath is a string of path storing the parameters 
%   needed for sinulation, including grid points and 
%   physics constant for BEC.
% 
%   The parameters of  beam are stored in lightPath
% 
%   Last revised on 21/12/2020.  LG = 0; no update chemical energy

    %% Physical system parameters
    
load(parPath);
    
% names = fieldnames(mydata);

% % discouraged use
%     for k = 1:length(names)
%         eval([names{k} '=mydata.' names{k} ]);  
%     end

    
    Nx = size(x,2);
    Ny = size(y,2);
    Nz = size(z,2);
    dx = mode(diff(x));                        %in case of error in computing differential
    dy = mode(diff(y));
    dz = mode(diff(z));

    %Dimensionlesslize 
    Rabi = Rabi/omegaZ;                                                                               
    Ggg = (4*pi*hbar^2*As*Nbec/m)*unit^-3*(hbar*omegaZ)^-1;
    Gee = Ggg;  
    Gge = 0;
    Geg = 0;
    %% LG and T.F approximation
    [X,Y,Z] = meshgrid(x,y,z);
    
    %Lagrangian multiplier
    psiGmu = (15*Ggg  /  (  16*pi*sqrt(2))  ).^  (2/5);          %T.F. chemical energy (treat psiE0 as zeros)
    psiEmu = (15*Ggg  /  (  16*pi*sqrt(2))  ).^  (2/5);                                                  
    init_psiGmu = psiGmu;
    init_psiEmu = psiEmu;

    Epot = 1/2 * (  (omegaXY/omegaZ)^2 *(X.^2 + Y.^2) +  Z.^2);
    TFsolG = (psiGmu-Epot)/Ggg;                               
    whereMinus = (TFsolG < 0);
    TFsolG(whereMinus) = 0;  
    TFsolG = sqrt(TFsolG);

    
    LGdata =  load(lightPath);
    Lambda = LGdata.Lambda;
    L = LGdata.L;
    P = LGdata.P;
    W0 = LGdata.W0;
   
    LGmsg = ['Using LG beam stored in:\n%s\nwith\n', ...
        'l,p=%d,%d Wavelength=%e BeamWasit=%e\n'];
    fprintf(LGmsg,lightPath,L,P,Lambda,W0); %dimension needs care
    fprintf('\n');
    


    LG = 0.5*Rabi*LGdata.LGdata;
    
    LG = 0;


    %%  prelocation of data
    j = 0; J = 0;                   %%%runs

    fprintf(" Total runs %d steps , update every %d steps\n",...
        nj, stepJ)
    
    psiGmuArray = zeros(1, ceil(nj/stepJ) + 1);                 % Energy of groundstate every cetain step, 
    psiGmuArray(1) = init_psiGmu;                                   % At the first step is, the energy is the energy of T.F.          
    psiEmuArray = zeros(1, ceil(nj/stepJ) + 1);
    psiEmuArray(1) = init_psiEmu;

    LG = gpuArray(LG);
    psiE = gpuArray(zeros(Nx,Ny,Nz));  
    psiG = gpuArray(TFsolG);
    
    mydata = struct();                                                     
    %% main part
    %data interface
    %psi to gpuArray
    %converge condition

    storingData =  ...
        {... 
        'x', 'y', 'z', 'LG','Lambda','W0', 'psiG', 'psiE', 'm', 'TFsolG',  ...
        'init_psiGmu', 'init_psiEmu',  'psiGmuArray' , 'psiEmuArray', ...
        'time', 'j', 'J','stepJ'};

    datapath = fullfile('~','Documents','Lab','Projects','BECdata',[filename,'_%d.mat']); 

    tstart = tic;
    while j ~= nj  
%         tic;
        j = j+1;

        if mod(j, 4*stepJ) == 0 || j == nj
            data = ws2struct();                     %data of current workspace
            % put the dimension back to data.
            varname = fieldnames(data);
            data.('x') = x*unit;
            data.('y') = y*unit;
            data.('z') = z*unit;
            data.('LG') = LG*omegaZ;
            data.('psiG') = psiG*unit.^(-3/2);
            data.('psiE') = psiE*unit.^(-3/2);
            data.('psiGmu') = psiGmu*hbar*omegaZ;
            data.('psiEmu') = psiGmu*hbar*omegaZ;
            data.('TFsolG') = TFsolG*unit.^(-3/2);
            data.('psiGmuArray') = psiGmuArray*hbar*omegaZ;
            data.('psiEmuArray') = psiEmuArray*hbar*omegaZ;

            for k = 1:size(varname,1)
                if any(strcmp(varname{k}, storingData))
                %data to CPU to be stored as a scalar structure array    
                    mydata.(varname{k}) = gather(data.(varname{k})); 
                end
            end

            disp( ['save as ',  sprintf(datapath, j)] );
            save( sprintf(datapath, j), 'mydata' );    
            clear data

        end

%         psiGK = - 0.5 * mydel2(psiG, dx, dy, dz);        
%         psiEK = - 0.5 * mydel2(psiE, dx, dy, dz);
% - 0.5 * 2*ndims(psiE)*del2(psi, dx, dy, dz)
     


        psiG = -dw*( ... 
            - 0.5 * mydel2(psiG, dx, dy, dz) + ...                              %kinectic energy
            ( Epot + Ggg*abs(psiG).^2 + Gge*abs(psiE).^2) .* psiG  -  ...   %potential and collision energy
            psiGmu*psiG +  ...                                                           %energy condition
            conj(LG).*psiE...                                                                 % light
            ) + psiG;    
        psiE = -dw*( ... 
            - 0.5 * mydel2(psiE, dx, dy, dz) + ... 
            ( Epot  + Gee*abs(psiE).^2 + Geg*abs(psiG).^2).*psiE - ... 
            psiEmu*psiE + ... 
            LG.*psiG ... 
            ) + psiE;

        
        

        if mod(j,stepJ) == 0    
        % chemical potential convergent condition
            Nfactor = Normalize(psiG,psiE,dx,dy,dz);
            J = J + 1 ;
%             psiGmu = psiGmu/(Nfactor);
%             psiEmu = psiEmu/(Nfactor);
            psiGmuArray(J+1) = gather(sum(conj(psiG).* ... 
                (- 0.5 * 2*ndims(psiG)*del2(psiG, dx, dy, dz) + ...                   %Operated psiG
                (Epot + Ggg*abs(psiG).^2 + Gge*abs(psiE).^2) .* psiG ) ... 
                *dx*dy*dz,'all'));
            psiEmuArray(J+1) = gather(sum(conj(psiE).* ... 
                (- 0.5 * 2*ndims(psiE)*del2(psiE, dx, dy, dz) + ...                     %Operated psiE
                ( Epot  + Gee*abs(psiE).^2 + Geg*abs(psiG).^2).*psiE)... 
                *dx*dy*dz,'all'));
            
            time(J) = toc(tstart);
        end

%         toc;
    end

end

function Nfactor = Normalize(psiG,psiE,dx,dy,dz)
        SumPsiG = sum( abs(psiG).^2*dx*dy*dz,  'all' );
        SumPsiE = sum( abs(psiE).^2*dx*dy*dz,  'all' );
        Nfactor = SumPsiG +  SumPsiE;  
end
