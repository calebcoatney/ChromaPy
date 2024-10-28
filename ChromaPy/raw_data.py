from datetime import datetime
import numpy as np

class RawData:

    def __init__(self, path: str, file: str, datetime: datetime, sample: str, signal: str, time_values: np.ndarray, signal_values: np.ndarray):
        self.path = path
        self.file = file
        self.datetime = datetime
        self.sample = sample
        self.signal = signal
        self.time_values = time_values
        self._signal_values = np.array(signal_values)

    def __repr__(self):
        return f'{self.signal}\n{self.sample}\n{self.datetime}'

    @property
    def signal_values(self):
        """Return the original values."""
        return self._signal_values

    def identify_method(self):
        method = None
        if 'HC' in self.signal:
            method = 'HC'
        elif 'HV' in self.signal:
            method = 'HV'

        if method:
            return method
        else:
            raise Exception('Method not identified (HC or HV).')

    def identify_detector(self):
        detector = None
        if 'FID1A' in self.signal:
            detector = 'FID1A'
        elif 'TCD2B' in self.signal:
            detector = 'TCD2B'
        elif 'TCD3C' in self.signal:
            detector = 'TCD3C'

        if detector:
            return detector
        else:
            raise Exception('Detector not identified (FID1A, TCD2B, TCD3C)')

    def get_sample_number(self) -> int:
        # Split the key into parts based on spaces
        parts = self.signal.split()

        # Find the index of the method 'HC'
        if 'HC' in parts:
            method_index = parts.index('HC')
        elif 'HV' in parts:
            method_index = parts.index('HV')

        # The sample number is the part immediately following the method
        sample_number_str = parts[method_index + 1]

        # Chop off the '.D'
        if '.D' in sample_number_str:
            sample_number_str = sample_number_str.split('.')[0]

        # Convert the sample number string to an integer and return it
        return int(sample_number_str)

    def name_sample(self):

        if 'HC' in self.sample:
            return self.sample.replace('HC', 'Sample ')
        if 'HV' in self.sample:
            return self.sample.replace('HV', 'Sample ')