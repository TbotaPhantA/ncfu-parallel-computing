from numpy import empty, linspace, tanh
import time

def u_init(x, y) :
    u_init = 0.5*tanh(1/eps*((x - 0.5)**2 + (y - 0.5)**2 - 0.35**2)) - 0.17
    return u_init

def u_left(y, t) :
    u_left = 0.33
    return u_left

def u_right(y, t) :
    u_right = 0.33
    return u_right

def u_top(x, t) :
    u_top = 0.33
    return u_top

def u_bottom(x, t) :
    u_bottom = 0.33
    return u_bottom

start_time = time.time()

a = -2.; b = 2.
c = -2.; d = 2.
t_0 = 0.; T = 5.

eps = 10**(-1.0)

N_x = 50; N_y = 50; M = 500

h_x = (b - a)/N_x; x = linspace(a, b, N_x+1)
h_y = (d - c)/N_y; y = linspace(c, d, N_y+1)

tau = (T - t_0)/M; t = linspace(t_0, T, M+1)

u = empty((M + 1, N_x + 1, N_y + 1))

for i in range(N_x + 1) :
    for j in range(N_y + 1) :
        u[0, i, j] = u_init(x[i], y[j])

for m in range(1, M + 1) :
    for j in range(1, N_y) :
        u[m, 0, j] = u_left(y[j], t[m])
        u[m, N_x, j] = u_right(y[j], t[m])
    for i in range(0, N_x + 1) :
        u[m, i, N_y] = u_top(x[i], t[m])
        u[m, i, 0] = u_bottom(x[i], t[m])
    
for m in range(M) :
    # print('m=',m)
    
    for i in range(1, N_x) :
        for j in range(1, N_y) :
            u[m+1, i, j] =  u[m,i,j] + \
                tau*(eps*((u[m,i+1,j] - 2*u[m,i,j] + u[m,i-1,j])/h_x**2 +
                          (u[m,i,j+1] - 2*u[m,i,j] + u[m,i,j-1])/h_y**2) +
                     u[m,i,j]*((u[m,i+1,j] - u[m,i-1,j])/(2*h_x) +
                               (u[m,i,j+1] - u[m,i,j-1])/(2*h_y)) + 
                     u[m,i,j]**3)

end_time = time.time()

print('Elapsed time is {:.4f} sec'.format(end_time-start_time)) # 2.5

# from numpy import savez
# savez('results_of_calculations', x=x, y=y, t=t, u=u)