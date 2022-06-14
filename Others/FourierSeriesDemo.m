% check fourier series(determined) of a step function

dx = 0.001;
x = -2:dx:2;
y = step(x,2,4);


fseries = zeros(1, length(x));
fseries2 = zeros(1, length(x)); %phase form
n = 100;
for i = 1:n
    fseries = fseries + sin( (2*i-1)*pi.*x )./(2*i-1);
    fseries2 = fseries2 + cos( (2*i-1)*pi.*x -pi/2)./(2*i-1);
end

a0 = 1;
a02 = 2;
    
fseries = a0/2 + 2/pi*fseries;
fseries2 = a02/2 + 2/pi*fseries2;

plot(x,y,'--','LineWidth',4)
hold on
plot(x,fseries);
plot(x,fseries2);




function y = step(x,p,len)
    tmp = mod(x,p);
    y = tmp<p/2;
    y(abs(x)>len) = 0;
end


