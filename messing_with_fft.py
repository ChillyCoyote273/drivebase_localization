import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt


def func(x):
    return (x - 5)**2 / 10


def main():
    # N = 1000
    # T = 10 / N
    xs = np.linspace(0, 10, 1000)
    ys = fft.ifft(np.array([1 if 300 < i < 600 else 0 for i in range(1000)]))
    
    plt.plot(xs, ys)
    plt.show()
    # ys = np.vectorize(func)(xs)
    # x_freqs = fft.fftfreq(N, T)[:N // 2]
    # y_freqs = fft.fft(ys)[:N // 2]
    
    # _, (time, freq) = plt.subplots(2)
    # time.plot(xs, ys)
    # freq.plot(x_freqs, y_freqs)
    # plt.show()



if __name__ == "__main__":
    main()
