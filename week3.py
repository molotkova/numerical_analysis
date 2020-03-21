import math
import matplotlib.pyplot as plt
import numpy as np

# Part I
print('Part I')
def newton_iteration(f, fder, x0, eps=1e-5, maxiter=1000):
    x = x0
    niter = 0
    while niter < maxiter:
        niter += 1
        x0 = x
        x = x - f(x) / fder(x)
        if abs(x0 - x) < eps:
            break
        if niter == maxiter:
            # достигнуто максимальное число итераций
            return None
    return x, niter

# Test I.1
print('Test I.1')
f = lambda x: x**2 - 1
fder = lambda x: 2 * x
print('В зависимости от начального приближения находится один из корней уравнения.')
print(newton_iteration(f, fder, x0=-0.5))
print(newton_iteration(f, fder, x0=0.5))

def newton_iteration_mod(f, fder, x0, m, eps=1e-5, maxiter=1000):
    x = x0
    niter = 0
    while niter <= maxiter:
        niter += 1
        x0 = x
        x = x - m * f(x) / fder(x)
        if abs(x0 - x) < eps:
            break
        if niter == maxiter:
            # достигнуто максимальное число итераций
            return None
    return x, niter

# Test I.2
print('Test I.2')
f = lambda x: (x**2 - 1)**2
fder = lambda x: 4 * (x**2 - 1) * x
m = [1, 2, 3, 4, 5]
# посмотрим, что происходит для разных m
for value in m:
    print(newton_iteration_mod(f, fder, x0=0.5, m=value), 'm =', value)
print('Видно, что верна гипотеза о квадратичной сходимости к одному из решений при m = 2.')

# Part II
print('Part II')
f_l = lambda x: math.sqrt(x)
f_r = lambda x: math.cos(x)
z = np.linspace(0, 2, num=100)
y1 = np.array([f_l(x) for x in z])
y2 = np.array([f_r(x) for x in z])
plt.figure()
plt.plot(z, y1, label='sqrt(x)')
plt.plot(z, y2, label='cos(x)')
plt.legend()
plt.grid()
plt.show()
# оценка на корень из графика
x_est = 0.6
print('Оценка на корень из графика:', x_est)

def fixed_point_iter(g, x0, eps=1e-5, maxiter=1000):
    """Метод для нахождения корня уравнения f(x) = 0,
    приведённого к виду x = g(x).
    g - функция (см. выше),
    смысл остальных параметров тот же, что и раньше.
        """
    x = x0
    niter = 0
    while niter < maxiter:
        x0 = x
        niter += 1
        x = g(x)
        if abs(x0 - x) < eps:
            break
        if niter == maxiter:
            # достигнуто максимальное число итераций
            return None
    return x, niter

# Test
# f(x) = sqrt(x) - cos(x)
g = lambda x: math.cos(x) ** 2
print('Результат метода итераций')
print('Корень:', fixed_point_iter(g, x0=0.6)[0], 'Число итераций:', fixed_point_iter(g, x0=0.6)[1])

def fixed_point_iter_mod(f, x0, a, eps=1e-5, maxiter=1000):
    x = x0
    niter = 0
    while niter < maxiter:
        x0 = x
        niter += 1
        x = x - a * f(x)
        if abs(x0 - x) < eps:
            break
        if niter == maxiter:
            # достигнуто максимальное число итераций
            return None
    return x, niter

# Test
f = lambda x: math.sqrt(x) - math.cos(x)
print('Результат модифицированного метода итераций для разных значений параметра a')
a = np.linspace(0.1, 1.5, num=20)
for value in a: print(fixed_point_iter_mod(f, x0=0.6, a=value), 'a =', value)
print('Видно, что лучшая сходимость достигается при a ~ 0.8')
# корень локализован на отрезке [0.5, 0.7]
fder = lambda x: 0.5 / math.sqrt(x) + math.sin(x)
# посмотрим значения производной f(x) на отрезке [0.5, 0.7]
z = np.linspace(0.5, 0.7, 10)
fder_val = [fder(x) for x in z]
fder_max = max(fder_val)
fder_min = min(fder_val)
print('Проверим гипотезу об оптимальном значении параметра a:')
print('a_opt =', 2 / (fder_max + fder_min))
print('Корень:', fixed_point_iter_mod(f, x0=0.6, a=2 / (fder_max + fder_min))[0],
      'Число итераций:', fixed_point_iter_mod(f, x0=0.6, a=2 / (fder_max + fder_min))[1])



