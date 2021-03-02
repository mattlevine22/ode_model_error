

%----------StartChua.m----------
x0 = -0.01 + 0.02*rand;
y0 = -0.01 + 0.02*rand;
z0 = -0.01 + 0.02*rand;

[t,y] = ode45(@fchua,[0 100],[x0 y0 z0]);
figure;
c=0;
for i=1:3
    for j=1:3
        c = c + 1;
        subplot(3,3,c)
        if i==j
            ksdensity(y(:,i))
        else
            ksdensity([y(:,i),y(:,j)],'PlotFcn','contour');
%             scatter(y(:,i), y(:,j))
        end
    end
end

figure;
for i=1:3
    subplot(3,1,i)
    plot(t, y(:,i))
end

function out = fchua(t,in)

x = in(1);
y = in(2);
z = in(3);

alpha  = 15.6;
beta   = 28; 
m0     = -1.143;
m1     = -0.714;

h = m1*x+0.5*(m0-m1)*(abs(x+1)-abs(x-1));

xdot = alpha*(y-x-h);
ydot = x - y+ z;
zdot  = -beta*y;

out = [xdot ydot zdot]';
end