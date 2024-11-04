import sys
import tkinter as tk
import tkinter.filedialog as tkFileDialog
from datetime import datetime
import pandas as pd
import numpy as np
from tests.test_raw_data import RawData


class DataHandler:

    def __init__(self, detectors: list = ['FID1A', 'TCD2B', 'TCD1B']):
        self.data = self._load_samples()
        self.detectors = detectors
        self.raw_data = self._parse_samples(self.data)

    def _select_file(self):
        root = tk.Tk()
        root.after(100, root.focus_force)
        root.after(200, root.withdraw)
        file_path = tkFileDialog.askopenfilename(parent=root, title='Pick a file')

        if not file_path:  # Check if no file was selected
            print("No file selected.")
            root.destroy()  # Close the Tkinter root window
            sys.exit()

        root.destroy()
        return file_path

    def _load_samples(self):
        data_file = self._select_file()
        data = pd.read_csv(data_file)
        return data

    def _parse_samples(self, df):
        """
        Parses data exported from GC3 to a single .csv file. Adapts to various GC data formats.
        """
        data = df.values.tolist()
        raw_data = {d: {} for d in self.detectors}
        path, file, date, sample, signal = None, None, None, None, None
        time_values, signal_values = [], []

        def store():
            # Determine the correct detector based on the current signal name
            detector = next((d for d in self.detectors if d in signal), None)
            if detector:
                raw_data[detector][signal] = RawData(
                    path, file, date, sample, signal, np.array(time_values), np.array(signal_values)
                )

        for row in data:
            # Skip rows where row[0] is not a string, thus preventing errors on 'Path' check
            if isinstance(row[0], str) and 'Path' in row[0]:
                continue

            # Detect start of a new data block with a file path
            if isinstance(row[0], str) and 'C:' in row[0]:
                path, file = row[0], row[1]
                date = datetime.strptime(row[2].strip(), '%d %b %Y %H:%M')
                sample = row[3]

            # New signal section starts here
            elif isinstance(row[0], str) and 'Signal' in row[0]:
                if signal is not None and signal != row[0]:
                    store()
                signal = row[0]
                time_values, signal_values = [], []  # Reset lists for new signal data

            else:
                try:
                    # Attempt to parse time and signal values as floats
                    time_values.append(float(row[0]))
                    signal_values.append(float(row[1]))
                except (ValueError, IndexError):
                    continue  # Skip row if unable to convert to floats or if row is malformed

        # Store any remaining data for the last sample read
        if sample is not None:
            store()

        return raw_data
