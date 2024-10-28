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

    '''
    THIS NEEDS TO TAKE IN A PREPROCESSING PARAMETERS DICTIONARY
    '''

    def find_baseline(self, y, x, preprocessing_parameters: dict):
        baseline_distance = preprocessing_parameters['baseline'].get('distance')

        # Calculate baseline
        baseline_prominence = np.abs(np.max(-y) - np.median(y)) / preprocessing_parameters['baseline'].get(
            'prominence') or np.abs(np.max(-y) - np.median(y)) / 50
        baseline, _ = scipy.signal.find_peaks(
            -y, distance=baseline_distance, prominence=(None, baseline_prominence))
        baseline_x = x[baseline]
        baseline_y = y[baseline]

        tck = splrep(baseline_x, baseline_y)
        baseline_y = splev(x, tck)
        baseline_y = np.where(baseline_y > y, y, baseline_y)

        return baseline_y

    def get_baseline_limits(self, raw_data: RawData, preprocessing_parameters: dict, verbose: bool = False):
        x = raw_data.time_values
        y = raw_data.signal_values

        # Apply smoothing if specified
        if "smoothing" in preprocessing_parameters and preprocessing_parameters["smoothing"].get('is_smooth'):
            kernel_size = int(preprocessing_parameters["smoothing"].get(
                "kernel_size")) * 2 + 1  # must be odd
            window_length = int(preprocessing_parameters["smoothing"].get("window_length"))
            if verbose:
                print(
                    f"Smoothing enabled with kernel_size={kernel_size}, window_length={window_length}")
            y = self.smooth(y, kernel_size, window_length)

        # Calculate baseline prominence
        baseline_prominence = np.abs(np.max(-y) - np.median(y)) / preprocessing_parameters['baseline'].get(
            'prominence') or np.abs(np.max(-y) - np.median(y)) / 50
        if verbose:
            print(f"Calculated baseline prominence: {baseline_prominence}")

        min_distance = 1
        max_distance = preprocessing_parameters['baseline']['distance'] or len(
            x) // 5
        increment = len(x) // 100

        # Single attempt to find peaks at the initial max_distance
        try:
            if verbose:
                print(
                    f'Trying baseline detection at initial max_distance={max_distance}')
            baseline, _ = scipy.signal.find_peaks(
                -y, distance=max_distance, prominence=(None, baseline_prominence))
            baseline_x = x[baseline]
            baseline_y = y[baseline]

            if len(baseline) < 5:
                raise Exception

            # Perform spline interpolation once
            tck = splrep(baseline_x, baseline_y)
            if verbose:
                print(f'k={tck[-1]}, m={len(baseline)}')
            baseline_y = splev(x, tck)
            baseline_y = np.minimum(baseline_y, y)

        except Exception as e:
            if verbose:
                print(
                    f"Error finding baseline at initial max_distance={max_distance}: {e}")
            max_distance = len(x) // 100
            # return (min_distance, max_distance)  # Early return on failure

        # Increment max_distance to find the best peaks
        while max_distance < len(x):
            max_distance += increment
            if verbose:
                print(f'Increasing max distance: {max_distance}')
            try:
                baseline, _ = scipy.signal.find_peaks(
                    -y, distance=max_distance, prominence=(None, baseline_prominence))
                baseline_x = x[baseline]
                baseline_y = y[baseline]

                if len(baseline) < 5:
                    raise Exception

                # Perform spline interpolation once
                tck = splrep(baseline_x, baseline_y)
                if verbose:
                    print(f'k={tck[-1]}, m={len(baseline)}')
                baseline_y = splev(x, tck)
                baseline_y = np.minimum(baseline_y, y)
            except Exception as e:
                if verbose:
                    print(
                        f"Error interpolating baseline at max_distance={max_distance}: {e}")
                max_distance -= increment
                break

        if verbose:
            print(
                f"Returning baseline limits: min_distance={min_distance}, max_distance={max_distance}")
        return (min_distance, max_distance)

    def find_peaks(self, y, x, params):
        peak_prominence = params.get('peak_prominence')

        # Find peaks
        peaks, _ = scipy.signal.find_peaks(y, prominence=peak_prominence)
        peaks_x = x[peaks]
        peaks_y = y[peaks]

        return peaks_x, peaks_y

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

        # Extract baseline parameters
        if "baseline" in preprocessing_parameters:
            baseline_distance = preprocessing_parameters["baseline"].get(
                "distance")
            baseline_prominence = np.abs(np.max(-y) - np.median(y)) / preprocessing_parameters['baseline'].get(
                'prominence') or np.abs(np.max(-y) - np.median(y)) / 50
            baseline, _ = scipy.signal.find_peaks(
                -y, distance=baseline_distance, prominence=(None, baseline_prominence))
            baseline_x = x[baseline]
            baseline_y = y[baseline]

            try:
                tck = splrep(baseline_x, baseline_y)
                # print(f'k={tck[-1]}, m={len(baseline)}')
                baseline_y = splev(x, tck)
                baseline_y = np.minimum(baseline_y, y)
                results['baseline_y'] = baseline_y

                if preprocessing_parameters["baseline"].get('is_baseline_corrected'):
                    y -= baseline_y
                    baseline_y -= baseline_y
                    results['y'] = y
                    results['baseline_y'] = baseline_y

            except Exception as e:
                print(f'Error while preprocessing sample:\n{raw_data}\n{e}')
                print(
                    f'Smoothing enabled?: {preprocessing_parameters["smoothing"].get("is_smooth")}')
                print(
                    f'Tried to use baseline_distance={baseline_distance} with baseline_prominence={baseline_prominence}')
                print(f'Baseline found with this distance: {baseline}')

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

    def integrate(self, raw_data: RawData, preprocessing_parameters: dict, chemstation_area_factor: float = 588, verbose: bool = True):
        # Preprocess the raw data using the provided parameters
        preprocessed_data = self.preprocess(raw_data, preprocessing_parameters)

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

        # Iterate through the filtered peaks from preprocess and integrate
        for i, (apex_x, apex_y) in enumerate(zip(peaks_x, peaks_y)):
            # Find the index of the apex in x and y arrays
            peak_idx = np.where(x == apex_x)[0][0]
            baseline_at_apex = baseline_y[peak_idx]

            # Calculate vertical distance between apex and baseline
            vertical_distance = apex_y - baseline_at_apex
            threshold = vertical_distance * 0.0025  # 0.25% threshold

            # Initialize left and right bounds at the apex
            left_bound = peak_idx
            right_bound = peak_idx

            # Move left from the apex until the threshold condition is met
            while left_bound > 0 and (y[left_bound] - baseline_y[left_bound]) > threshold:
                left_bound -= 1

            # Move right from the apex until the threshold condition is met
            while right_bound < len(y) - 1 and (y[right_bound] - baseline_y[right_bound]) > threshold:
                right_bound += 1

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

            # To match the integrated areas and RFs from MSD Productivity ChemStation for GC3, a
            # correction factor is needed. Update default value as necessary in method parameters.
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
            'peaks_list': peaks_list,
            # 'processed_sample': processed_sample
        }