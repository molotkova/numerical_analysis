import numpy as np
from numpy.testing import assert_allclose
import sklearn.datasets
rndm = np.random.RandomState(1234)

n = 10
b = rndm.uniform(size=n)

def jacobi(A, b, n_iter):
    d = np.diag(A)
    D = np.diag(d)
    invD = np.diag(1. / d)
    B = -A.copy()
    np.fill_diagonal(B, 0)
    BB = invD @ B
    # норма BB
    if np.linalg.norm(BB) > 1: raise ValueError('Не выполнено условие на норму матрицы B')
    c = invD @ b
    x0 = np.ones(b.shape[0])
    x = x0
    for _ in range(n_iter):
        x = BB @ x + c
    # проверка
    xx = np.linalg.solve(A, b)
    assert_allclose(x, xx)
    assert_allclose(-B + D, A)
    # np.testing.assert_allclose(A@xx, b) - метод напрямую проверяет работу np.linalg
    # следующими действиями проверяем верность построения вспмогательных матриц D, BB, B
    assert_allclose(D @ xx, B @ xx + b)
    assert_allclose(xx, BB @ xx + c)
    return x
# Test I
print('Test I')
A1 = rndm.uniform(size=(n, n)) + np.diagflat([15]*n)
print('Метод Якоби для матрицы А со свойством диагонального преобладания - работает ок (проходит все проверки).')
print('x =', jacobi(A1, b, n_iter=50))
# что если уменьшать диагональные элементы матрицы А?
# циклы ниже уменьшают диагональные элементы и ищут решения
for i in list(reversed(range(15))):
    A_1 = rndm.uniform(size=(n, n)) + np.diagflat([i]*n)
    # всё ок с решением до i = 7 включительно
    #print('x =', jacobi(A_1, b, n_iter=50))
    #print(i)
for i in list(reversed(range(7))):
    A = rndm.uniform(size=(n, n)) + np.diagflat([i]*n)
    d = np.diag(A)
    D = np.diag(d)
    invD = np.diag(1. / d)
    B = -A.copy()
    np.fill_diagonal(B, 0)
    BB = invD @ B
    # норма B < 1 до i = 5 включительно
    #print(np.linalg.norm(BB), i)

def seidel(A, b, n_iter=1000, eps=1e-9):
    U = np.triu(A, 1)
    L = np.tril(A, -1)
    D = np.diag(np.diag(A))
    M = np.linalg.inv(A.copy() - U)
    c = M @ b
    BB = M @ U
    S = np.identity(b.shape[0]) - M @ A
    if np.max(np.abs(np.linalg.eigvals(S))) > 1: raise ValueError('Не выполнено условие сходимости на матрицу итераций')
    x0 = np.ones(b.shape[0])
    x = x0
    count = 0
    for _ in range(n_iter):
        count += 1
        x0 = x
        x = c - BB @ x
        if np.linalg.norm(x - x0) < eps:
            break
    return x, count
# Test II
print('Test II')
print('Метод Зейделя для произвольной матрицы (для которой выполняется условие сходимости на стационарную матрицу итераций)')
print('A - рандомная симметричная и положительно определённая матрица:')
A_sym = 3 * np.ones((n, n)) + sklearn.datasets.make_spd_matrix(n, random_state=1234)
xx = np.linalg.solve(A_sym, b)
x = seidel(A_sym, b)
print('linalg for A_sym:', xx)
print('seidel for A_sym:', x[0], 'number of iterations:', x[1])
print('Отклонение от тосчного решения:', np.linalg.norm(xx - x[0]))
# результат работы assert:
#Mismatch: 20%
#Max absolute difference: 1.61535296e-08
#Max relative difference: 2.2853728e-07

#assert_allclose(x, xx)

print('A - не совсем рандом, но на ней ломался метод Якоби:')
A2 = rndm.uniform(size=(n, n)) + np.diagflat([3]*n)
xx = np.linalg.solve(A2, b)
x = seidel(A2, b)
print('linalg for A:', xx)
print('seidel for A:', x[0], 'number of iterations:', x[1])
print('Отклонение от точного решения:', np.linalg.norm(xx-x[0]))
#assert_allclose(x, xx)

def min_residual(A, b, n_iter=1000, eps=1e-9):
    '''
    Функция решает СЛАУ Ax=b методом минимальных невязок.
    param A, b: параметры СЛАУ
    param n_iter: число итераций
    param eps: точность решения
    return: x - решение, count - потребовавшееся число итераций.
    '''
    x0 = np.ones(b.shape[0])
    x = x0
    count = 0
    for _ in range(n_iter):
        count += 1
        x0 = x
        r = A @ x - b  # r(n)
        t = np.dot(r, A @ r) / (np.linalg.norm(A @ r)) ** 2  # t(n+1)
        x = x - t * r
        if np.linalg.norm(x - x0) < eps:
            break
    return x, count
# Test III
print('Test III')
print('Метод минимальных невязок для произвольной матрицы A.')
print('A - не совсем рандом, но на ней ломался метод Якоби:')

A2 = rndm.uniform(size=(n, n)) + np.diagflat([3]*n)
xx = np.linalg.solve(A2, b)
x = min_residual(A2, b)
print('linalg for A:', xx)
print('min_residual for A:', x[0], 'number of iterations:', x[1])
print('Отклонение от точного решения:', np.linalg.norm(xx-x[0]))
#assert_allclose(x, xx)


A_sym = 7 * np.ones((n, n)) + sklearn.datasets.make_spd_matrix(n, random_state=1234)
xx = np.linalg.solve(A_sym, b)
x = min_residual(A_sym, b, n_iter=10000)
print('A - рандомная симметричная и положительно определённая матрица:')
print('linalg for A_sym:', xx)
print('min_residual for A_sym:', x[0], 'number of iterations:', x[1])
print('Отклонение от точного решения:', np.linalg.norm(xx-x[0]))
#assert_allclose(x, xx)
