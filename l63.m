

%----------StartChua.m----------

x0 = -0.01 + 0.02*rand;
y0 = -0.01 + 0.02*rand;
z0 = -0.01 + 0.02*rand;

x0 = 1;
y0 = 1;
z0 = 1;

tspan = 0:0.001:100;
[t,yOUT] = ode45(@fl63,tspan,[x0 y0 z0]);
x = yOUT(:,1);
y = yOUT(:,2);
z = yOUT(:,3);

figure;
c=0;
for i=1:3
    for j=1:3
        c = c + 1;
        subplot(3,3,c)
        if i==j
            ksdensity(yOUT(:,i))
        else
            ksdensity([yOUT(:,i),yOUT(:,j)],'PlotFcn','contour');
%             scatter(y(:,i), y(:,j))
        end
    end
end

figure;
for i=1:3
    subplot(3,1,i)
    plot(t, yOUT(:,i))
end

function out = fl63(t,in)

x = in(1);
y = in(2);
z = in(3);

sig = 10; %a
r = 28; 
b = 1; %c

xdot = sig*(y-x);
ydot = r*x - y - x*z;
zdot = x*y -b*z;

out = [xdot ydot zdot]';
end