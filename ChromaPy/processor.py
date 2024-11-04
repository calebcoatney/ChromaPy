from .data_handler import DataHandler
from .raw_data import RawData
from .peak import Peak
import numpy as np
from tabulate import tabulate
import scipy
from scipy.signal import savgol_filter, medfilt
from scipy.interpolate import splrep, splev
from scipy.integrate import simpson


class Processor:

    def __init__(self, DataHandler: DataHandler = None):

        if DataHandler:
            self.DataHandler = DataHandler
        else:
            self.DataHandler = DataHandler()

    def smooth(self, values, kernel_size=15, window_length=25):
        """Apply smoothing to the provided values."""
        smoothed_values = medfilt(values, kernel_size=kernel_size)
        smoothed_values = savgol_filter(
            smoothed_values, window_length=window_length, polyorder=4)
        return smoothed_values

    def identify_compound(self, retention_time: float, RT_table: dict):

        compound = '#N/A'

        for key, value in RT_table.items():
            rt_min, rt_max = float(value[0]), float(value[1])
            if rt_min <= retention_time <= rt_max:
                compound = key

        return compound

    def dpa_baseline(self, x, y, preprocessing_parameters):
        """
        Performs a derivative-based accumulation operation and baseline estimation in one function.

        Args:
            x (array): x-values of the signal.
            y (array): y-values of the signal.
            preprocessing_parameters (dict): Dictionary containing parameters for 'window' and 'threshold'.

        Returns:
            array: Estimated baseline values for y.
        """
        # Retrieve parameters
        window = preprocessing_parameters['baseline']['window']

        # Calculate the derivative trace
        d = [(y[k + 1] - y[k]) / (x[k + 1] - x[k]) for k in range(len(x) - 1)]

        # Split d into positive (P) and negative (N) components
        P = [dk if dk >= 0 else 0 for dk in d] + [0]
        N = [dk if dk <= 0 else 0 for dk in d] + [0]

        # Initialize accumulation array
        alpha = np.zeros(len(x))
        for i in range(len(x)):
            for j in range(window + 1):
                if i - j >= 0 and i + j < len(x):
                    alpha[i] += P[i - j] - N[i + j]
        alpha /= window  # Normalize by width

        percentile = preprocessing_parameters['baseline']['threshold']
        threshold = np.percentile(alpha, percentile)

        # Identify baseline points using threshold
        baseline_points = [(xi, yi) for xi, yi, a in zip(x, y, alpha) if a < threshold]

        if not baseline_points:  # Handle empty baseline points case
            return np.zeros(len(y))

        # Fit spline to baseline points
        baseline_x, baseline_y = zip(*baseline_points)
        tck = splrep(baseline_x, baseline_y, k=2)
        baseline_y = splev(x, tck)

        # Ensure baseline does not exceed the signal
        baseline_y = np.where(baseline_y > y, y, baseline_y)

        return baseline_y

    def get_max_peak_prominence(self, raw_data: RawData):
        y = raw_data.signal_values

        min_prominence = (np.max(y) - np.median(y))/100
        max_prominence = (np.max(y) - np.median(y))/50
        increment = (np.max(y) - np.median(y))/100

        peaks, _ = scipy.signal.find_peaks(
            y, prominence=(min_prominence, max_prominence))
        while max_prominence < (np.max(y) - np.median(y))*4:
            max_prominence += increment
            peaks, _ = scipy.signal.find_peaks(
                y, prominence=(min_prominence, max_prominence))

        return max_prominence

    def preprocess(self, raw_data: RawData, preprocessing_parameters: dict):
        """Process the current sample based on the provided preprocessing parameters.

        Args:
            raw_data: The RawData sample object to be processed.
            preprocessing_parameters (dict): A dictionary of parameters for preprocessing.

        Returns:
            dict: A dictionary containing processed results, such as {'x': x, 'y': y, 'baseline_y': baseline_y, 'peaks_x': peaks_x, 'peaks_y': peaks_y}.
        """

        x = raw_data.time_values
        y = raw_data.signal_values

        # Initialize results dictionary
        results = {
            'x': None,
            'y': None,
            'baseline_y': None,
            'peaks_x': None,
            'peaks_y': None
        }

        x = raw_data.time_values
        y = raw_data.signal_values

        # Apply smoothing if specified
        if "smoothing" in preprocessing_parameters and preprocessing_parameters["smoothing"].get('is_smooth'):
            kernel_size = int(preprocessing_parameters["smoothing"].get(
                "kernel_size")) * 2 + 1  # must be odd
            window_length = int(preprocessing_parameters["smoothing"].get("window_length"))
            y = self.smooth(y, kernel_size, window_length)

        results['x'] = x
        results['y'] = y

        baseline_y = self.dpa_baseline(x, y, preprocessing_parameters)
        results['baseline_y'] = baseline_y

        if preprocessing_parameters["baseline"].get('is_baseline_corrected'):
            y -= baseline_y
            baseline_y -= baseline_y
            results['y'] = y
            results['baseline_y'] = baseline_y

        # Extract peak parameters
        if "peak" in preprocessing_parameters:
            min_peak_prominence = preprocessing_parameters["peak"].get(
                "min prominence")
            max_peak_prominence = preprocessing_parameters["peak"].get(
                "max prominence")
            peaks, _ = scipy.signal.find_peaks(y, prominence=(
                min_peak_prominence, max_peak_prominence))

            peaks_x = x[peaks]
            peaks_y = y[peaks]

            # If RT table is present, filter peaks according to retention times
            if "RT Table" in preprocessing_parameters and preprocessing_parameters["RT Table"]:
                rt_table = preprocessing_parameters["RT Table"]
                filtered_peaks_x = []
                filtered_peaks_y = []

                for peak_x, peak_y in zip(peaks_x, peaks_y):
                    for compound, (rt_min, rt_max) in rt_table.items():
                        # Ensure the RT values are floats for comparison
                        rt_min = float(rt_min)
                        rt_max = float(rt_max)
                        if rt_min <= peak_x <= rt_max:
                            filtered_peaks_x.append(peak_x)
                            filtered_peaks_y.append(peak_y)
                            break  # Stop checking once the peak is within one RT range

                # Replace peaks with filtered results
                peaks_x = np.array(filtered_peaks_x)
                peaks_y = np.array(filtered_peaks_y)

            # Store filtered or original peaks in the results dictionary
            results['peaks_x'] = peaks_x
            results['peaks_y'] = peaks_y

        return results

    def integrate(self, raw_data: RawData, preprocessing_parameters: dict, chemstation_area_factor: float = 580, verbose: bool = True):
        # Preprocess the raw data using the provided parameters
        preprocessed_data = self.preprocess(raw_data, preprocessing_parameters)
        criteria = ['threshold', 'minimum']

        if "RT Table" in preprocessing_parameters:
            rt_table = preprocessing_parameters.get("RT Table")

        # Extract relevant values from the preprocessed data
        x = preprocessed_data['x']
        y = preprocessed_data['y']
        baseline_y = preprocessed_data['baseline_y']
        peaks_x = preprocessed_data['peaks_x']
        peaks_y = preprocessed_data['peaks_y']

        # Initialize lists to store the integration data
        peaks_list = []
        ret_times = []
        integrated_areas = []
        integration_bounds = []
        x_peaks = []
        y_peaks = []
        baseline_peaks = []

        # Iterate through each peak for integration
        for i, (apex_x, apex_y) in enumerate(zip(peaks_x, peaks_y)):
            # Find the index of the apex in x and y arrays
            peak_idx = np.where(x == apex_x)[0][0]
            baseline_at_apex = baseline_y[peak_idx]

            # Initialize left and right bounds based on peak apex
            left_bound = peak_idx
            right_bound = peak_idx

            # Define the range for min_left and min_right calculations
            if i > 0:  # If not the first peak
                # Find the index of the previous peak's x value in the x array
                start_idx = np.where(x == peaks_x[i - 1]
                                     )[0][0] if np.any(x == peaks_x[i - 1]) else None
                if start_idx is not None:
                    min_left = start_idx + np.argmin(y[start_idx:peak_idx])
                else:
                    min_left = 0
            else:
                min_left = 0

            if i < len(peaks_x) - 1:  # If not the last peak
                # Find the index of the next peak's x value in the x array
                end_idx = np.where(x == peaks_x[i + 1]
                                   )[0][0] if np.any(x == peaks_x[i + 1]) else None
                if end_idx is not None:
                    min_right = peak_idx + np.argmin(y[peak_idx:end_idx])
                else:
                    min_right = -1
            else:
                min_right = -1

            # Calculate vertical distance between apex and baseline
            vertical_distance = apex_y - np.minimum(y[min_left], y[min_right])
            threshold = vertical_distance * 0.0025  # 0.25% threshold for signal proximity to baseline

            dx = 25

            # Calculate the left bound using the specified criteria
            previous_slope = None
            for j in range(peak_idx, 2, -1):
                diff = y[j] - baseline_y[j]

                # Central difference approximation
                if j >= dx and j < len(y) - dx:
                    slope = (y[j] - y[j - dx]) / (x[j] - x[j - dx])
                else:
                    # Edge case fallback to first-order difference
                    slope = (y[j] - y[j - 1]) / (x[j] - x[j - 1])

                # Check based on provided criteria
                if 'threshold' in criteria and diff <= threshold:
                    left_bound = j
                    break

                if 'minimum' in criteria and j == min_left:
                    left_bound = j
                    break

                if 'slope' in criteria and previous_slope is not None and np.sign(slope) != np.sign(previous_slope):
                    left_bound = j
                    break

                previous_slope = slope

            # Calculate the right bound similarly
            previous_slope = None
            for j in range(peak_idx, len(y) - 3):
                diff = y[j] - baseline_y[j]

                # Central difference approximation
                if j >= dx and j < len(y) - dx:
                    slope = (y[j + dx] - y[j]) / (x[j + dx] - x[j])
                else:
                    # Edge case fallback to first-order difference
                    slope = (y[j + 1] - y[j]) / (x[j + 1] - x[j])

                # Check based on provided criteria
                if 'threshold' in criteria and diff <= threshold:
                    right_bound = j
                    break

                if 'minimum' in criteria and j == min_right:
                    right_bound = j
                    break

                if 'slope' in criteria and previous_slope is not None and np.sign(slope) != np.sign(previous_slope):
                    right_bound = j
                    break

                previous_slope = slope

            # Append retention time at the peak (apex)
            ret_times.append(apex_x)

            # Extract data for integration
            x_peak = x[left_bound:right_bound + 1]
            y_peak = y[left_bound:right_bound + 1]
            baseline_peak = baseline_y[left_bound:right_bound + 1]

            x_peaks.append(x_peak)
            y_peaks.append(y_peak)
            baseline_peaks.append(baseline_peak)

            # Correct the signal by subtracting the baseline
            y_peak_corrected = y_peak - baseline_peak

            # Calculate the integrated area using Simpson's rule
            area = simpson(y_peak_corrected, x=x_peak)

            # Apply correction factor
            area *= chemstation_area_factor

            # Store integrated area and bounds
            integrated_areas.append(area)
            integration_bounds.append((x[left_bound], x[right_bound]))

            # Create a Peak object
            retention_time = apex_x
            compound_id = self.identify_compound(retention_time, rt_table)
            peak_number = i + 1
            integrator = 'py'
            start_time = x[left_bound]
            end_time = x[right_bound]
            width = end_time - start_time

            peaks_list.append(Peak(compound_id, peak_number, retention_time,
                                   integrator, width, area, start_time, end_time))

        if verbose:
            headers = ['Compound ID', 'Peak #', 'Ret Time',
                       'Integrator', 'Width', 'Area', 'Start Time', 'End Time']
            print(raw_data)
            print(tabulate([p.as_row for p in peaks_list], headers),
                  end='\n\n')

        return {
            'peaks': peaks_list,
            'x_peaks': x_peaks,
            'y_peaks': y_peaks,
            'baseline_peaks': baseline_peaks,
            'retention_times': ret_times,
            'integrated_areas': integrated_areas,
            'integration_bounds': integration_bounds,
            'peaks_list': peaks_list
        }
