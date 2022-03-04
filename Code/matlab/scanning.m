function scanning(spatialProfile,plotwhat,direction,varargin)
%SCANNING scanning spatial spatialProfile of the experiment data
%   The experiment data consists wavefunction of
%   BEC ground stsate,BEC excieted stata, and 
%   electrical field of LG light beam.
% 
%   SCANNING(spatialProfile,plotwhat,direction)
%   spatialProfile is a 1x1 structure array of the data 
%   key: value pairs
%   {
%   x: x cooridnate, y: y cooridnate , z: z cooridnate, 
%   psiG | psiE | LG: data in 3D arrays
%   plotwhat: properties to be plotted
%   direction: direction to be scanned
%   }
% 
%   SCANNING(spatialProfile,plotwhat,direction,*OPTIONAL)
%   OPTIONAL: name-value pairs of plotting parameters
%   Quiversize: size of arrow in quiverplot
%   Inter: Space between data points plotted
%   Margin: How much area does plotting windows shrinked 
%   Color: Color of arrows in quiver plot
%   Cmode: auto or manual colorbar setting  

%   LG needs wavelength, wavefunction needs mass

%   see also Light_BEC
%   11/30/2020 by daydream 
%

% datapath = '/home/GuoJinHao/Documents/Lab/Projects/BECdata/Miscellaneous/testcode2_30000.mat';
% load(datapath)         
%
% load( '/home/GuoJinHao/Documents/Lab/Projects/BECdata/Paremeters/Light_BEC.mat')
%interface


p = inputParser;

dataType = fieldnames(spatialProfile);
whichdata = dataType{end};

defaultQuiversize = 3;
defaultInter = 3;
defaultMargin = 40 ;
defaultColor = 'blue' ;
defaultMode = 'auto' ;
defaultLength =  'notgiven';                                    %User must provide length. 
defaultMass =  'notgiven';                                    %If  user doesn't give the length, than it leads to error.                                                                                
% check3D = @(x) validateattributes(x, "double", {'ndims',3});  %not% strucutre

addRequired(p, 'spatialProfile');
addRequired(p, 'plotwhat', @ischar);
addRequired(p, 'direction', @ischar);

if strcmp(whichdata, 'LG')
    addOptional(p,'Length', defaultLength);
    
end

if any(strcmp(whichdata, {'psiG','psiE'}))
    addOptional(p,'Mass', defaultMass);
end

addParameter(p,'Margin', defaultMargin, @isscalar);
addParameter(p,'Inter', defaultInter, @isscalar);
addParameter(p,'Quiversize', defaultQuiversize, @isscalar);
addParameter(p,'Color', defaultColor, @ischar);
addParameter(p,'Cmode', defaultMode, @ischar);


parse(p, spatialProfile, plotwhat, direction, varargin{:});

if strcmp(whichdata, 'LG') 
    if strcmp(p.Results.Length, 'notgiven')
        error("You didn't give the lenght of light beam")
        return
    else
        Lambda = p.Results.Length;
    end
end

if any(strcmp(whichdata, {'psiG','psiE'}))
    if any(strcmp(p.Results.Mass, 'notgiven'))
        error("You didn't give the mass of atoms")
        return
    else
        Mass = p.Results.Mass;
    end
end

Color = p.Results.Color;
direction = p.Results.direction;
Inter = p.Results.Inter;
Margin = p.Results.Margin;
plotwhat =    p.Results.plotwhat;
spatialProfile = p.Results.spatialProfile;
Quiversize = p.Results.Quiversize;
Cmode = p.Results.Cmode;



%%
x = spatialProfile.x;
y = spatialProfile.y;
z = spatialProfile.z;

Nx = size(x,2); Ny = size(y,2); Nz = size(z,2);
dx = diff(x(round(Nx/2):round(Nx/2)+1),1);
dy = diff(y(round(Ny/2):round(Ny/2)+1),1);
dz = diff(z(round(Nz/2):round(Nz/2)+1),1);
Nx = size(x,2); Ny = size(y,2);Nz = size(z,2);

                                                                      
rangex = 1+Margin:Inter:Nx-Margin;     
rangey = 1+Margin:Inter:Ny-Margin;   
rangez = 1:Inter:Nz;
crangex  = 1+Margin:Nx-Margin;     
crangey  = 1+Margin:Ny-Margin;  
crangez = 1+Margin:Nz-Margin;

hf = figure();
hf.Position = [430,10, 1060 , 1060];    
dim = [.4 .7 .3 .3];
str = 'left-click to stop';
han = annotation('textbox',dim,'String',str,'FitBoxToText','on');

phase = angle(spatialProfile.(whichdata));



if strcmp(plotwhat,'current')
% % % % %     
% Wavefunction needs prabability current.
% Light beam needs OAM of poynting vectors 
hold on

    switch whichdata
        case {'psiG', 'psiE'}            
            [Jx,Jy,Jz] =current(spatialProfile.(whichdata),dx,dy,dz,Mass);          
        case 'LG'         
            [Jx,Jy,Jz] = oam(spatialProfile.(whichdata),dx,dy,dz,Lambda);                
    end
    
    switch direction
       
    % initialize quiver plot for different directions
        case 'z'
            zerosmtx =  zeros(size(rangey,2),size(rangex,2));   
            hq = quiver3(x(rangex), y(rangey) , zerosmtx, zerosmtx, zerosmtx, zerosmtx, Quiversize,Color);

            xlabel('x')
            ylabel('y')
            Nlim = Nz;
            idx = 1:Nz;
        case 'x'                                                                   
            zerosmtx =  zeros(size(rangez,2),size(rangey,2));   
            hq = quiver3(y(rangey), z(rangez) , zerosmtx,zerosmtx,zerosmtx,zerosmtx,Quiversize,Color);

            xlabel('y')
            ylabel('z')
            Nlim = Nx;
            idx = 1:Nx;
            Jx = shiftdim(Jx,2);                                      % y-x-z circular shifted
            Jy = shiftdim(Jy,2);
            Jz = shiftdim(Jz,2);
            phase = shiftdim(phase,2);

        case 'y'
            zerosmtx =  zeros(size(rangex,2),size(rangez,2));  
            hq = quiver3(z(rangez), x(rangex), zerosmtx,zerosmtx,zerosmtx,zerosmtx,Quiversize,Color);

            xlabel('z')
            ylabel('x')
            Nlim = Ny;    
            idx = 1:Ny;
            Jx = shiftdim(Jx,1);
            Jy = shiftdim(Jy,1);
            Jz = shiftdim(Jz,1);
            phase = shiftdim(phase,1);

    end

    hax = ancestor(hq,'axes');  
    
    
    
    
    
elseif any(strcmp(plotwhat, {'density','phase'}))
% % % % %     
% Data <--- wavefunction or light beam  to 
% plot denstiy or phase   

    Data = spatialProfile.(whichdata); 
    
    switch direction
        % initialize surface plot for different directions
        case 'z'
            zerosmtx =  zeros(size(rangey,2),size(rangex,2));   
            hs = surf(x(rangex), y(rangey) , zerosmtx,'EdgeColor','none');
            xlabel('x')
            ylabel('y')
            idx = 1:Nz;
            Nlim = Nz;
        case 'x'
            zerosmtx =  zeros(size(rangez,2),size(rangey,2));   
            hs = surf(y(rangey), z(rangez) , zerosmtx,'EdgeColor','none');
            xlabel('y')
            ylabel('z')
            Nlim = Nx;
            idx = 1:Nx;
            Data = shiftdim(Data,2);
        case 'y'
            zerosmtx =  zeros(size(rangex,2),size(rangez,2));   
            hs = surf(z(rangez), x(rangex) , zerosmtx,'EdgeColor','none');
            xlabel('z')
            ylabel('x')
            Nlim = Ny;        
            idx = 1:Ny;
            Data = shiftdim(Data,1);
    end
    
    hax = ancestor(hs,'axes');   
    
end
    
% colorbar setting
    if strcmp(plotwhat,'density')
        switch whichdata
            case {'psiG', 'psiE'}
                cmax = max(abs(Data.^2)*dx*dy*dz,[],'all');
                cmin = min(abs(Data.^2)*dx*dy*dz,[],'all');
                caxis([cmin,cmax])
            case 'LG'
                cmax = max(abs(Data.^2),[],'all');
                cmin = min(abs(Data.^2),[],'all');
                caxis([cmin,cmax])
        end
    elseif any(strcmp(plotwhat, {'current','phase'})) 
        cmax = max(phase,[],'all');
        cmin = min(phase,[],'all');
        caxis([cmin,cmax])
    end
        
        caxis(Cmode)    




colorbar; axis tight;
view([0 90])
k = 0;



while true
    
    k = k + 1; 

    if strcmp(plotwhat,'current')
    
        switch direction
            case 'z'
                set(hq,'Udata',Jx(rangey,rangex,idx(k)))
                set(hq,'Vdata',Jy(rangey,rangex,idx(k)))
                set(hq,'Wdata',Jz(rangey,rangex,idx(k)))                      
                % object method doesn't work(need to set contour maxtric)
                % plot directly                
                contour(hax,x(crangex), y(crangey),phase(crangey,crangex,idx(k)),8)

                set(hax.Title,'String',sprintf('z=%.2e(m)',z(idx(k))))
            case 'y'
                set(hq,'Udata',Jz(rangex,rangez,idx(k)))
                set(hq,'Vdata',Jx(rangex,rangez,idx(k)))
                set(hq,'Wdata',Jy(rangex,rangez,idx(k)))
                
                contour(hax,z(crangez), x(crangex),phase(crangex,crangez,idx(k)),8)
                
                set(hax.Title,'String',sprintf('y=%.2e(m)',y(idx(k))))
            case 'x'
                set(hq,'Udata',Jy(rangez,rangey,idx(k)))
                set(hq,'Vdata',Jz(rangez,rangey,idx(k)))
                set(hq,'Wdata',Jx(rangez,rangey,idx(k)))

                contour(hax,y(crangey), z(crangez),phase(crangez,crangey,idx(k)),8)

                set(hax.Title,'String',sprintf('x=%.2e(m)',x(idx(k))))
        end
        


    elseif strcmp(plotwhat,'density')


        if any(strcmp(whichdata, {'psiG','psiE'}))
            switch direction
                case 'z'
                    set(hs,'Zdata',abs(squeeze(Data(rangey,rangex,idx(k))).^2)*dx*dy*dz)
                    set(hax.Title,'String',sprintf('z=%.2e(m)',z(idx(k))))
                case 'y'
                    set(hs,'Zdata',abs(squeeze(Data(rangex,rangez,idx(k))).^2)*dx*dy*dz)
                    set(hax.Title,'String',sprintf('y=%.2e(m)',y(idx(k))))
                case 'x'
                    set(hs,'Zdata',abs(squeeze(Data(rangez,rangey,idx(k))).^2)*dx*dy*dz)
                    set(hax.Title,'String',sprintf('x=%.2e(m)',x(idx(k))))
            end
        elseif strcmp(whichdata, 'LG')
            switch direction
                case 'z'
                    set(hs,'Zdata',abs(squeeze(Data(rangey,rangex,idx(k))).^2))
                    set(hax.Title,'String',sprintf('z=%.2e(m)',z(idx(k))))
                case 'y'
                    set(hs,'Zdata',abs(squeeze(Data(rangex,rangez,idx(k))).^2))
                    set(hax.Title,'String',sprintf('y=%.2e(m)',y(idx(k))))
                case 'x'
                    set(hs,'Zdata',abs(squeeze(Data(rangez,rangey,idx(k))).^2))
                    set(hax.Title,'String',sprintf('x=%.2e(m)',x(idx(k))))
             end
        end


    elseif strcmp(plotwhat,'phase')

        switch direction
            case 'z'
                set(hs,'Zdata',angle(squeeze(Data(rangey,rangex,idx(k)))))
                set(hax.Title,'String',sprintf('z=%.2e(m)',z(idx(k))))
            case 'y'
                set(hs,'Zdata',angle(squeeze(Data(rangex,rangez,idx(k)))))
                set(hax.Title,'String',sprintf('y=%.2e(m)',y(idx(k))))
            case 'x'
                set(hs,'Zdata',angle(squeeze(Data(rangez,rangey,idx(k)))))
                set(hax.Title,'String',sprintf('x=%.2e(m)',x(idx(k))))
        end

    else

        disp('Noplot')
        break

    end


        drawnow
        pause(0.01)

        pause on
    if k == Nlim
        k = 0;
        idx(:) = idx(end:-1:1);
        pause(1)
    elseif k == Nlim && idx(1) == Nlim
        k = 0;
        idx(:) = idx(end:-1:1);
        pause(1)
    end

    %   user  interacitve stuff
    mouseState = get(hf,'SelectionType');                 

    if strcmpi(mouseState,'alt')       
        set(hf,'SelectionType','normal')
        set(han,'String', ... 
            'Press q to quick, other button to pause, and right-click to continue')
        keystate = waitforbuttonpress;
        value = double(get(gcf,'CurrentCharacter'));

        if keystate == 1
            if value == 113                                             %q
                close(hf)
            else
                pause on
                set(han,'String', ... 
            'Pausing. Any keybutton to continue')
                pause
                value = double(get(gcf,'CurrentCharacter'));
                if value == 113
                    close(hf)
                end
            end
        end
        set(han,'String','left-click to stop')
    end
    
    if strcmpi(plotwhat,'current') 
    %clear previous contour plot
        hca = gca;
        delete(hca.Children(1));
    end
    
end
    
       
    
end
