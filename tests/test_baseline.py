import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from scipy.ndimage import minimum_filter, gaussian_filter1d, percentile_filter
from scipy.signal import savgol_filter, medfilt, find_peaks
from scipy.interpolate import splrep, splev
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from ChromaPy.data_handler import DataHandler
from ChromaPy.processor import Processor

# Initialize data handler
data_handler = DataHandler()

# %% Derivative Passing Accumulation Function


def dpa_operation(x, y, w):
    """
    Performs a derivative-based accumulation operation.
    Args:
        x (array): x-values of the signal
        y (array): y-values of the signal
        w (int): Maximum shift width for accumulation
    Returns:
        array: Accumulated quantity (alpha)
    """
    # Calculate the derivative trace
    d = [(y[k+1] - y[k]) / (x[k+1] - x[k]) for k in range(len(x) - 1)]
    # Split d into positive (P) and negative (N) components
    P = [dk if dk >= 0 else 0 for dk in d] + [0]  # padded for length match
    N = [dk if dk <= 0 else 0 for dk in d] + [0]

    # Initialize accumulation array
    alpha = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(w + 1):
            if i - j >= 0 and i + j < len(x):
                alpha[i] += P[i - j] - N[i + j]
    return alpha/w

# %%


def WhittakerSmooth(x, w, lambda_, differences=1):
    '''
    Penalized least squares algorithm for background fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties

    output
        the fitted background vector
    '''
    X = np.matrix(x)
    m = X.size
    E = eye(m, format='csc')
    for i in range(differences):
        E = E[1:]-E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W+(lambda_*E.T*E))
    B = csc_matrix(W*X.T)
    background = spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting

    output
        the fitted background vector
    '''
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax+1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x-z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001*(abs(x)).sum() or i == itermax):
            if (i == itermax):
                print('WARNING max iteration reached!')
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i*np.abs(d[d < 0])/dssn)
        w[0] = np.exp(i*(d[d < 0]).max()/dssn)
        w[-1] = w[0]

    z = np.where(z > y, y, z)
    return z

# %% Adaptive Derivative Passing Accumulation (Auto-DPA)


def auto_dpa(x, y):
    """
    Adaptive Derivative Passing Accumulation Method with adjustable window.
    Args:
        x (array): x-values of the signal
        y (array): y-values of the signal
    Returns:
        array: Adaptive accumulation vector (a)
    """
    m = len(x) - 1
    d = np.array([(y[k + 1] - y[k]) / (x[k + 1] - x[k]) for k in range(m)])
    P = np.array([dk if dk >= 0 else 0 for dk in d])
    N = np.array([dk if dk <= 0 else 0 for dk in d])
    a, B, c = np.zeros(m + 1), np.zeros(m + 1), np.zeros(m + 1)

    for w in range(1, m // 2):
        for i in range(w, m - w):
            if c[i] == 0 and not np.any(c[i-w:i+w+1] == 0):
                c[i] = 1
            if c[i] == 1:
                continue
            if a[i] == max(a[i - w:i + w + 1]):
                for j in range(1, w + 1):
                    B[i] += P[i - j] - N[i + j]
            else:
                c[i] = 1
        for i in range(w, m - w):
            a[i] = B[i]
    return a

# %% Bi-trapezoid Threshold Calculation


def bi_trapezoid_threshold(y):
    """
    Calculates a threshold using the bi-trapezoid method.
    Args:
        y (array): y-values of the signal
    Returns:
        float: Threshold (tau)
    """
    # Step 1: Sort the sequence and subtract the minimum value
    y_min = min(y)
    z = np.sort(y - y_min)  # Create sorted z array

    # Step 2: Initialize alpha and beta
    alpha = 0
    beta = np.sum(z)

    # Step 3: Compute bi-trapezoid values
    n = len(z)
    t = np.zeros(n)
    for i in range(n):
        alpha += z[i]
        beta -= z[i]
        t[i] = min(alpha / (i + 1), beta / (z[i] + z[-1]) * (n - i - 1))

    # Step 4: Find the maximum value in t and calculate threshold
    max_t_index = np.argmax(t)
    tau = z[max_t_index] + y_min

    return tau

# %% Derivative Passing Accumulation Method (Peak Detection)


def derivative_passing_accumulation_method(x, y, w):
    """
    Peak detection using Derivative Passing Accumulation Method.
    Args:
        x (array): x-values of the signal
        y (array): y-values of the signal
        w (int): Maximum shift width for accumulation
        t (float): Threshold for peak detection
    Returns:
        list: Recognized peaks and baseline
    """

    alpha = dpa_operation(x, y, w)
    t = 250  # bi_trapezoid_threshold(y)
    print(t)
    background_points = [(xi, yi) for xi, yi, a in zip(x, y, alpha) if a < t]

    x_background, y_background = zip(*background_points)
    tck = splrep(x_background, y_background, k=2)
    baseline_y = splev(x, tck)
    baseline_y = np.where(baseline_y > y, y, baseline_y)

    return baseline_y

# %%


def find_baseline(y, x, distance=5, prominence=0.1):
    baseline_distance = len(y) * distance / 100
    baseline_prominence = np.ptp(y) * prominence / 100

    baseline, _ = find_peaks(y, distance=baseline_distance, prominence=(None, baseline_prominence))
    baseline_x = x[baseline]
    baseline_y = y[baseline]

    tck = splrep(baseline_x, baseline_y, k=2)
    baseline_y = splev(x, tck)
    baseline_y = np.where(baseline_y > y, y, baseline_y)

    return baseline_y

# %%


def dpa_baseline(x, y, t):

    threshold = t

    alpha = dpa_operation(x, y, 25)
    threshold = np.percentile(alpha, 5)
    # alpha = processor.smooth(alpha)

    baseline_points = [(xi, yi) for xi, yi, a in zip(x, y, alpha) if a < threshold]

    baseline_x, baseline_y = zip(*baseline_points)
    tck = splrep(baseline_x, baseline_y, k=2)
    baseline_y = splev(x, tck)
    baseline_y = np.where(baseline_y > y, y, baseline_y)

    return baseline_y

# %%


processor = Processor(data_handler)

detector = 'FID1A'
sample_number = 20

method = 'HC' if detector == 'TCD' else 'HV'
sample = f'Signal: 240731 CO2 HYD run 11 {method}{sample_number}.D\\{detector}.ch'

raw_data = data_handler.raw_data[detector][sample]

x = raw_data.time_values
y = processor.smooth(raw_data.signal_values, kernel_size=35, window_length=35)

b = dpa_baseline(x, y, 25)


plt.plot(x, y, label='Chromatogram')
plt.plot(x, b, label='DPA')
plt.legend()
plt.show()

# %%


y_diff = np.abs(np.diff(y) / np.diff(x))
x_diff = (x[:-1] + x[1:]) / 2
alpha = dpa_operation(x, y, 50)
baseline = derivative_passing_accumulation_method(x, y, 50)

plt.plot(x_diff, y_diff, label='dy/dx', alpha=0.3)
plt.plot(x, y, label='Signal')
plt.plot(x, alpha, label='Alpha')
plt.plot(x, baseline, label='Baseline')
plt.legend()
plt.show()
