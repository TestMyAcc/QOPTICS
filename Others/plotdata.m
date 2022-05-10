%% load data to plotting

clear
LGmatlab = load( 'C:\Users\Lab\Documents\121_121_121W01_Lambda1_L&P10.mat').LG;
LGdata = h5read("LG10_121-121-121.h5",'/LGdata');
LGpython = LGdata.r + 1i*LGdata.i;
% error from orientation of 3d Array(row major or column major?);
LGpython = shiftdim(pagetranspose(LGpython),2);
x = h5read("LG10_121-121-121(4).h5",'/Coordinates/x');
y = h5read("LG10_121-121-121(4).h5",'/Coordinates/y');
z = h5read("LG10_121-121-121(4).h5",'/Coordinates/z');
x = x';
y = y';
z = z';

% compare LG produced by matlab and python
(121*121*121 - nnz(ismembertol(real(LGmatlab),real(LGpython),1e-5)))/(121*121*121) * 100
(121*121*121 - nnz(ismembertol(imag(LGmatlab),imag(LGpython),1e-5)))/(121*121*121) * 100

%%
data = struct('x',x,'y',y,'z',z, 'LG',LGpython);
scanning(data, 'current', 'z', 1,'Inter',3,'Quiversize',5)
