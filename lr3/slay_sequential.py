from numpy import zeros, empty, dot, arange, linalg, eye
from matplotlib.pyplot import style,  figure , axes, show

def conjugate_gradient_method(A, b, x, N):
  s = 1
  p = zeros(N)
  while s <= N:
    if s == 1:
      r = dot(A.T, dot(A,x) - b)
    else:
      r = r - q/dot(p, q)
    p = p + r/dot(r, r)
    q = dot(A.T, dot(A, p))
    x = x - p/dot(p,q)
    s = s + 1
  return x


f1 = open('in.dat', 'r')
N = int(f1.readline())
M = int(f1.readline())
f1.close()

A = empty((M,N));
b = empty(M)

f2 = open('AData.dat', 'r')
for j in range(M):
  for i in range(N):
    A[j, i] = float(f2.readline())
f2.close()

f3 = open('bData.dat', 'r')
for j in range(M):
  b[j] = float(f3.readline())
f3.close()

x = zeros(N)

x = conjugate_gradient_method(A, b, x, N)

fig = figure();
ax = axes(xlim=(0, N), ylim=(-1.5, 1.5))
ax.set_xlabel('i');
ax.set_ylabel('x[i]')
ax.plot(arange(N), x, '-y', lw=3)

show()