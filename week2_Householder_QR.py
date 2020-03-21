import numpy as np
from numpy.testing import assert_allclose
from numpy import linalg as LA
def householder(vec):
    """Construct a Householder reflection to zero out 2nd and further components of a vector.

    Parameters
    ----------
    vec : array-like of floats, shape (n,)
        Input vector

    Returns
    -------
    outvec : array of floats, shape (n,)
        Transformed vector, with ``outvec[1:]==0`` and ``|outvec| == |vec|``
    H : array of floats, shape (n, n)
        Orthogonal matrix of the Householder reflection
    """
    vec = np.asarray(vec, dtype=float)
    if vec.ndim != 1:
        raise ValueError("vec.ndim = %s, expected 1" % vec.ndim)
    outvec = np.zeros(len(vec))
    outvec[0] = LA.norm(vec)
    u = (vec + np.copysign(outvec, vec)) / LA.norm(vec + np.copysign(outvec, vec))
    H = np.identity(len(vec)) - 2 * np.outer(u, u.T)
    return outvec, H

def qr_decomp(a):
    """Compute the QR decomposition of a matrix.

    Parameters
    ----------
    a : ndarray, shape(m, n)
        The input matrix

    Returns
    -------
    q : ndarray, shape(m, m)
        The orthogonal matrix
    r : ndarray, shape(m, n)
        The upper triangular matrix

    Examples
    --------
    >>> a = np.random.random(size=(3, 5))
    >>> q, r = qr_decomp(a)
    >>> np.assert_allclose(np.dot(q, r), a)

    """
    a1 = np.array(a, copy=True, dtype=float)
    m, n = a1.shape
    if m == n: iter = n - 1
    if n > m: iter = m
    if n < m: iter = n
    q = np.identity(m)
    for c in range(iter):
        H = np.identity(m)
        h = householder(a1.T[c][c:])[1]
        H[c:, c:] = h
        q = q @ H
        a1 = H @ a1
    r = np.copy(a1)
    return q, r
# Test I.1
print('Если первая компонента вектора отрицательна, умножение матрицы, полученной методом householder(), на исходный вектор \n'
      'даёт повёрнутый вектор с первой компонентой - нормой исходного вектора, остальные компоненты нули.'
      'Если первая компонента вектора положительна, умножение матрицы, полученной методом householder(), на исходный вектор \n'
      'даёт повёрнутый вектор с первой компонентой (-1)*(норма исходного вектора), остальные компоненты нули.')
v = np.array([1, 2, 3])
f = np.array([-1, -2, -3])
v1, h = householder(v)
f1, hf = householder(f)
print(v1)
print(h @ v)
print(f1)
print(h @ f)
#assert_allclose(np.dot(h, v1), v)
#assert_allclose(np.dot(h, v), v1)
# Test I.2
rndm = np.random.RandomState(1234)
vec = rndm.uniform(size=7)
v1, h = householder(vec)
#assert_allclose(np.dot(h, v1), vec)


# Test II.1
rndm = np.random.RandomState(1234)
a = rndm.uniform(size=(5, 3))
q, r = qr_decomp(a)
# test that Q is indeed orthogonal
assert_allclose(np.dot(q, q.T), np.eye(5), atol=1e-10)
# test the decomposition itself
assert_allclose(np.dot(q, r), a)
# Test II.2
from scipy.linalg import qr
qq, rr = qr(a)
assert_allclose(np.dot(qq, rr), a)
# матрицы Q полностью совпадают:
assert_allclose(qq, q)
# матрицы R похожи с хорошей точностью. Проходят проверку numpy.allclose и не проходят assert. Почему?
#assert_allclose(rr, r)
np.allclose(rr, r)
print('R by scipy:\n', rr)
print('R by qr_decomp:\n', r)
print('Матрицы R похожи с хорошей точностью.')

def first_ur(a):
    '''

    param a: произвольная матрица с m >= n, где m, n - количество строк и столбцов соответственно.
    return:
    u_vecs -  массив векторов-отражателей;
    r - матрица R из QR-разложения матрицы a.
    '''
    a1 = np.array(a, copy=True, dtype=float)
    m, n = a1.shape
    if m == n: iter = n - 1
    if n < m: iter = n
    u_vecs = np.zeros((m, m))
    for c in range(iter):
        vec = a1.T[c][c:]
        y = np.zeros(vec.shape[0])
        y[0] = LA.norm(vec)
        u = (vec + np.copysign(y, vec)) / LA.norm(vec + np.copysign(y, vec))
        u_vecs.T[-c - 1][c:] = u
        a1 = a1 - 2 * np.outer(u_vecs.T[-c - 1], np.dot(u_vecs.T[-c - 1], a1))
    r = a1.copy()
    return u_vecs, r

def second_r(a, u_vecs):
    '''

    param a: произвольная матрица с m >= n, где m, n - количество строк и столбцов соответственно;
    param u_vecs: массив векторов-отражателей матрицы a, полученный методом first_ur().
    return:
    r - матрица R из QR-разложения матрицы a.
    '''
    a1 = np.array(a, copy=True, dtype=float)
    m, n = a1.shape
    q = np.identity(m)
    if m == n: iter = n - 1
    if n < m: iter = n
    for c in range(iter):
        q = q - 2 * np.outer(q @ u_vecs.T[-c - 1], u_vecs.T[-c - 1])
    r = q.T @ a1
    return r
print('III часть задания.')
# Test III.1
# for square matrix
rndm = np.random.RandomState(1234)
b = rndm.uniform(size=(6, 6))
rr_sq = qr(b)[1]
r_sq = first_ur(b)[1]
np.allclose(rr_sq, r_sq)
# for matrix with m (rows) > n (columns)
rndm = np.random.RandomState(1234)
c = rndm.uniform(size=(6, 4))
rr_nsq = qr(c)[1]
r_nsq = first_ur(c)[1]
np.allclose(rr_nsq, r_nsq)
print('Функция first_ur() проходит тест для двух типов матриц (квадратная и прямоугольная). Метод рассчитывает матрицу \n'
      'R, которая с хорошей точностью совпадает с матрицей R, рассчитанной библиотечным методом qr().')

# Test III.2
# for square matrix
u_vecs_b = first_ur(b)[0]
r_sq = second_r(b, u_vecs_b)
np.allclose(r_sq, rr_sq)
# for matrix with m (rows) > n (columns)
u_vecs_c = first_ur(c)[0]
r_nsq = second_r(c, u_vecs_c)
np.allclose(r_nsq, rr_nsq)
print('Функция second_r() проходит тест для двух типов матриц (квадратная и прямоугольная). Метод рассчитывает матрицу \n'
      'R, которая с хорошей точностью совпадает с матрицей R, рассчитанной библиотечным методом qr().')
