import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

x0 = [-1, -0.7, -0.43, -0.14, -0.14, 0.43, 0.71, 1, 1.29, 1.57, 1.86, 2.14, 2.43, 2.71, 3]
y0 = [-2.25, -0.77, 0.21, 0.44, 0.64, 0.03, -0.22, -0.84, -1.2, -1.03, -0.37, 0.61, 2.67, 5.04, 8.90]
n = len(x0)

# I.I
def design_mat(x, m_size):
    # set design matrix A
    n_size = len(x)
    A = np.zeros((n_size, m_size + 1))  # m_size = 0, 1, 2,..., n-1
    for i in range(n_size):
        row = [x[i] ** k for k in range(m_size + 1)]
        A[i] = row
    return A
# дальше было удобно сделать фукнцию, которая сразу собирает design matrix A и ищет коэфы в полиноме
def lls_coeff(x, y, m_size):
    A = design_mat(x, m_size)
    # construct the equations to find coeff
    # A.T @ A @ b = A.T @ y, b - coeff vector
    L = A.T @ A
    R = A.T @ y
    b = LA.solve(L, R)
    return b

# I.II
m_ls = np.arange(0, n, 1)
t_ls = []
for m in m_ls:
    p = lambda z: np.dot(lls_coeff(x0, y0, m_size=m), [z ** k for k in range(m + 1)])
    t = np.array(list(map(p, x0))) - np.array(y0)
    t_ls.append(t)
sigma_sq_ls = []
sigma_sq = lambda m, t: 1 / (n - m) * (LA.norm(t)) ** 2
for m, t in zip(m_ls, t_ls):
    sigma_sq_ls.append(sigma_sq(m, t))

plt.figure()
plt.plot(m_ls, sigma_sq_ls, '.')
plt.grid()
plt.title('Find the optimal value of m')
plt.show()
# (sigma_m)**2 становится приблизительно константой начиная с m = 3 => 3 - оптимальная степень полинома
plt.close()

z = np.linspace(-1.1, 3.1, 1000)
plt.figure(figsize=(10, 10))
p = lambda z, m: np.dot(lls_coeff(x0, y0, m_size=m), [z ** k for k in range(m + 1)])
for m in range(6):
    if m == 3:
        plt.plot(z, p(z, m), label='m = ' + str(m) + ' - optimal value')
    else:
        plt.plot(z, p(z, m), label='m = '+str(m))
plt.plot(x0, y0, '.', label='Input data points')
plt.grid()
plt.legend()
plt.title('Polynomials with diff m')
plt.show()
# начиная с m = 3 (оптимального значения из начала I.II) аппроксимирующие полиномы P_m визуально совпадают
plt.close()

# I.III
def lls_coeff_qr(x):
    A = design_mat(x, m_size=3)
    Q, R = LA.qr(A)
    b = LA.inv(R) @ Q.T @ y0
    return b
print('Норма разности векторов из коэффициентов полинома, найденных разными способами:')
print(LA.norm(lls_coeff(x0, y0, 3) - lls_coeff_qr(x0)))
print('Результаты совпадают!')

# II.1
def f(x):
    return x**2 * np.cos(x)

z = np.linspace(0.5 * math.pi, math.pi, 1000)

def lagrange_pol(z, x, y, m):
    L = 0
    for j in range(len(y)):
        l = 1
        for p in range(m + 1):
            if p == j:
                continue
            l *= (z - x[p]) / (x[j] - x[p])
        L += y[j] * l
    return L

for m in [1, 2, 3, 4, 5]:
    x0 = np.linspace(0.5 * math.pi, math.pi, m + 1)
    y0 = f(x0)
    plt.figure(figsize=(5, 5))
    plt.plot(z, f(z), label='Original function')
    plt.plot(z, lagrange_pol(z, x0, y0, m), label='m = ' + str(m))
    plt.plot(x0, y0, '.', label='Input data points')
    plt.title('Lagrange interpolation using the uniform mesh')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()

# II.2
# Chebyshev nodes

for m in [1, 2, 3, 4, 5]:
    pi = math.pi
    a = 0.5 * pi
    b = pi
    x0 = [0.5*(a+b) + 0.5*(a-b) * math.cos(pi * (2*k-1) / (2*(m+1))) for k in range(1, m+2)]
    y0 = f(np.array(x0))
    plt.figure()
    plt.plot(z, f(z), label='Original function')
    plt.plot(z, lagrange_pol(z, x0, y0, m=m), label='m = ' + str(m))
    plt.plot(x0, y0, '.', label='Given data')
    plt.grid()
    plt.legend()
    plt.title('Interpolation using Chebyshev nodes')
    plt.show()
    plt.close()

#interpolation error

m = 3
pi = math.pi
a = 0.5 * pi
b = pi
x_unif = np.linspace(a, b, m+1)
x_cheb = [0.5*(a+b) + 0.5*(a-b) * math.cos(pi * (2*k-1) / (2*(m+1))) for k in range(1, (m+2))]
def error_pol(x, x_node):
    w = 1
    for point in x_node:
        w *= x - point
    return w
z = np.linspace(a, b, 1000)
er_unif = error_pol(z, x_unif)
er_cheb = error_pol(z, x_cheb)

plt.figure()
x0 = [0.5*(a+b) + 0.5*(a-b) * math.cos(pi * (2*k-1) / (2*(m+1))) for k in range(1, m+2)]
y0 = f(np.array(x0))
plt.plot(z, f(z), label='Original function')
plt.plot(z, lagrange_pol(z, x0, y0, m=m), label='Chebyshev interpolation nodes')
x0 = np.linspace(0.5 * math.pi, math.pi, m + 1)
y0 = f(x0)
plt.plot(z, lagrange_pol(z, x0, y0, m), label='The uniform mesh')
plt.legend()
plt.grid()
plt.title('Comparing interpolation results')
plt.show()
plt.close()

print('Чтобы уменьшить ошибку интерполяции, нужны узлы которые минимизируют многочлен (x-x0)*...*(x-x3) на заданном интервале, x0, ..., x3 - узлы. \n'
      'Сравним значения этих многочленов на данном отрезке в случае равномерной сетки и узлов Чебышева.')

plt.figure()
plt.plot(z, abs(er_unif), label='The uniform mesh')
plt.plot(z, abs(er_cheb), label='Chebyshev interpolation nodes')
plt.title('(x-x0)*...*(x-x3) polynomial behaviour')
plt.grid()
plt.legend()
plt.plot()
plt.show()
plt.close()