

%----------StartChua.m----------
% L63 params we desire
sig = 10;
rbar = 28;

% waterwheel parameters
K_leak = 1; % leakage rate
I = 1; %moment of inertia of the wheel
g = 1; %gravity
r = 1; %radius of the wheel
nu = sig*K_leak*I; %rotational damping rate
q1 = rbar*K_leak^2*nu / (pi*g*r); %first fourier coefficient for Q, the rate at which water is pumped into the chambers

% a10 = -0.01 + 0.02*rand;
% b10 = -0.01 + 0.02*rand;
% omega0 = -0.01 + 0.02*rand;

x0 = 1;
y0 = 1;
z0 = 1;

a10 = y0 * K_leak*nu/(pi*g*r);
b10 = q1/K_leak - z0*K_leak*nu/(pi*g*r);
omega0 = K_leak*x0;

tspan = 0:0.001:100;
[t,yOUT] = ode45(@fwheel,tspan,[a10 b10 omega0], [], K_leak, I, g, r, nu, q1);
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

a1 = yOUT(:,1);
b1 = yOUT(:,2);
omega = yOUT(:,3);

figure;
subplot(3,1,1)
plot(t, a1)
title('a1')

subplot(3,1,2)
plot(t, b1)
title('b1')

subplot(3,1,3)
plot(t, omega)
title('omega')

sgtitle('water wheel coordinates')

% now convert to L63 coordinates
x_tran = omega / K_leak;
y_tran = a1 * (pi*g*r) / (K_leak*nu);
z_tran = (b1 - q1/K_leak) * (-pi*g*r) / (K_leak*nu);
T = K_leak*t;


figure;
subplot(3,1,1)
hold on;
plot(T, x_tran);
plot(T, x); % from L63 script (run that first)
title('x')

subplot(3,1,2)
hold on;
plot(T, y_tran)
plot(T, y) % from L63 script (run that first)
title('y')

subplot(3,1,3)
hold on;
plot(T, z_tran)
plot(T, z) % from L63 script (run that first)
title('z')

sgtitle('L63 coordinates')



function out = fwheel(t,in, K_leak, I, g, r, nu, q1)

a1 = in(1);
b1 = in(2);
omega = in(3); %angular velocity of the wheel

adot = omega*b1 - K_leak*a1;
bdot = -omega*a1 - K_leak*b1 + q1;
omegadot = (-nu*omega + pi*g*r*a1) / I;



out = [adot bdot omegadot]';
end