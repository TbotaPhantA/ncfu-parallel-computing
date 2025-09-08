from numpy import dot, empty

f1 = open('in.dat', 'r')
N = int(f1.readline())
M = int(f1.readline())
f1.close()

A = empty((M, N))
x = empty(N)
b = empty(M)

f2 = open('AData.dat', 'r')
for j in range(M):
  for i in range(N):
    A[j, i] = float(f2.readline())
f2.close()

f3 = open('xData.dat', 'r')
for i in range(N):
  x[i] = float(f3.readline())
f3.close()

b = dot(A.T, x)

f4 = open('Results_sequential.dat', 'w')
for j in range(M):
  print(b[j], file=f4)
f4.close()
