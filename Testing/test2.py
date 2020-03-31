import concurrent.futures
import time, random               # add some random sleep time
import numpy as np
import sys
from multiprocessing import Pool
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import as_completed
sys.setrecursionlimit(15000)

offset = 2                        # you don't supply these so
def calc_stuff(parameter=None):   # these are examples.
    sleep_time = random.choice([0, 1, 2, 3, 4, 5])
    time.sleep(sleep_time)
    return parameter / 2, sleep_time, parameter * parameter

def procedure(j):                 # just factoring out the
    parameter = j * offset        # procedure
    # call the calculation
    return calc_stuff(parameter=parameter)

def fib(n):
   if n <= 1:
       return n
   else:
       return(fib(n-1) + fib(n-2))

def main():
    output1 = list()
    output2 = list()
    output3 = list()
    start = time.time()           # let's see how long this takes

    arr = []
    # we can swap out ProcessPoolExecutor for ThreadPoolExecutor
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     for i, out1 in enumerate(executor.map(fib, range(0, 30))):
    #         # put results into correct output list
    #         arr[i] = out1
    def fib2(n):
        a, b = 0, 1
        for i in range(0, n):
            a, b = b, a + b
        return ((n, a))

    numeros = [10, 20, 25]

    with ThreadPoolExecutor(max_workers=15) as executor:
        fibSubmit = {executor.submit(fib2, n): n for n in range(0, 100)}

        for future in as_completed(fibSubmit):
            try:
                n, f = future.result()
            except Exception as exc:
                print("Erro! {0}".format(exc))
            else:
                arr.append(f)
    # try:
    #     p = Pool()
    #     arr = p.map(fib, range(0, 30))
    # finally:
    #     p.close()
    finish = time.time()
    # these kinds of format strings are only available on Python 3.6:
    # time to upgrade!
    print(arr)
    print(f'Parallel time: {(finish-start)}')

    start = time.time()           # let's see how long this takes

    arr = np.zeros(30, dtype=int)
    # we can swap out ProcessPoolExecutor for ThreadPoolExecutor
    for i in range(0, 30):
        # put results into correct output list
        arr[i] = fib(i)
    finish = time.time()
    # these kinds of format strings are only available on Python 3.6:
    # time to upgrade!
    print(arr)
    print(f'Standard time: {(finish-start)}')

    # print(f'original inputs: {repr(output1)}')
    # print(f'total time to execute {sum(output2)} = sum({repr(output2)})')
    # print(f'returned in order given: {repr(output3)}')



    from scipy.optimize import minimize


    def rosenbrock(x, N):
        out = 0.0
        for i in range(N-1):
            out += 100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return out

    from numba import jit, float64, int64

    @jit(float64(float64[:], int64), nopython=True, parallel=False)
    def fast_rosenbrock(x, N):
        out = 0.0
        for i in range(N-1):
            out += 100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return out


    @jit(float64[:](float64[:], int64), nopython=True, parallel=False)
    def fast_jac(x, N):
        h = 1e-9
        jac = np.zeros_like(x)
        f_0 = fast_rosenbrock(x, N)
        for i in range(N):
            x_d = np.copy(x)
            x_d[i] += h
            f_d = fast_rosenbrock(x_d, N)
            jac[i] = (f_d - f_0) / h
        return jac

    # slow optimize
    N = 20
    x_0 = - np.ones(N)
    start = time.time()
    res = minimize(fast_rosenbrock, x_0, args=(N,), method='SLSQP', options={'maxiter': 1e4})
    finish = time.time()
    print(res.x)
    print(f'Standard time: {(finish-start)}')
    print('with fast jacobian')
    start = time.time()
    res = minimize(fast_rosenbrock, x_0, args=(N,), method='SLSQP', options={'maxiter': 1e4}, jac=fast_jac)
    finish = time.time()
    print(res.x)
    print(f'Standard time: {(finish-start)}')

if __name__ == '__main__':
    main()