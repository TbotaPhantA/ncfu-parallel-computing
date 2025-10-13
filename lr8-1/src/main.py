from numpy import empty, linspace, sin, pi
import time

def u_init(x) :
    u_init = sin(3*pi*(x - 1/6))
    return u_init

def u_left(t) :
    u_left = -1.
    return u_left

def u_right(t) :
    u_right = 1.
    return u_right

start_time = time.time()

a = 0.; b = 1.
t_0 = 0.; T = 6.0

eps = 10**(-1.5)

N = 200; M = 20000

h = (b - a)/N; x = linspace(a, b, N+1)
tau = (T - t_0)/M; t = linspace(t_0, T, M+1)

u = empty((M + 1, N + 1))

for n in range(N + 1) :
    u[0, n] = u_init(x[n])
    
for m in range(1, M + 1) :
    u[m, 0] = u_left(t[m])
    u[m, N] = u_right(t[m])
    
for m in range(M) :
    for n in range(1, N) :
        u[m+1, n] = u[m,n] + eps*tau/h**2*(u[m,n+1] - 2*u[m,n] + u[m,n-1]) + \
            tau/(2*h)*u[m,n]*(u[m,n+1] - u[m,n-1]) + tau*u[m,n]**3

end_time = time.time()

print('Elapsed time is {:.4f} sec'.format(end_time-start_time))

from numpy import savez
savez('results_of_calculations', x=x, u=u)