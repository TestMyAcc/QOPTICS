%% load data to plotting

clear
% load( 'C:\Users\Lab\Documents\121_121_121W01_Lambda1_L&P10.mat');
 LGdata = h5read("LG10_121-121-121(4).h5",'/LGdata');
 LG = LGdata.r + 1i*LGdata.i;
 x = h5read("LG10_121-121-121(4).h5",'/Coordinates/x');
 y = h5read("LG10_121-121-121(4).h5",'/Coordinates/y');
 z = h5read("LG10_121-121-121(4).h5",'/Coordinates/z');
 x = x';
 y = y';
 z = z';

data = struct('x',x,'y',y,'z',z, 'LG',LG);
scanning(data, 'current', 'x', 1,'Inter',3,'Quiversize',5)
