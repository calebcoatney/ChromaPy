from tests.test_raw_data import RawData
from tests.test_processor import Processor
from ChromaPy.rt_table import RTTableDialog
from ChromaPy.export_data import ExportDataDialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import customtkinter as ctk
import numpy as np
import threading


class App(ctk.CTk):
    def __init__(self, processor: Processor):
        super().__init__()

        self.is_smooth = False
        self.is_baseline_corrected = False
        self.automation_enabled = False
        self.current_fig = None
        self.current_ax = None

        self.preprocessing_parameters = {
            "smoothing": {
                "is_smooth": False,
                "kernel_size": 15,
                "window_length": 25
            },
            "baseline": {
                "is_baseline_corrected": False,
                "distance": None,
                "prominence": 50
            },
            "peak": {
                "min prominence": None,
                "max prominence": None
            },
            "RT Table": {
                'Carbon dioxide': ('1.85', '2.1'),
                'Helium': ('4.1', '4.3'),
                'Hydrogen': ('4.4', '4.9'),
                'Argon': ('7.4', '7.9'),
                'Nitrogen': ('9.4', '9.8'),
                'Carbon monoxide': ('14.9', '15.8')
            }
        }

        self.Processor = processor
        self._configure_window()
        self._create_widgets()

        self.load_samples(detector='TCD2B', sample=10)

    def _configure_window(self):
        """Configure the main window settings."""
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.protocol("WM_DELETE_WINDOW", self._quit)
        self.title("GC3 Chromatogram Analyzer")
        self.geometry(f"{1200}x{600}")
        self.minsize(1100, 500)

    def _create_widgets(self):
        """Create and organize all widgets."""
        self._create_tree_frame()
        self._create_plot_frame()
        self._create_sample_info_frame()
        self._create_parameter_controls()

    def _create_tree_frame(self):
        """Create and populate the tree view frame."""
        # Create the tree frame
        self.tree_frame = ctk.CTkFrame(
            self, width=250, corner_radius=0, bg_color="#2E2E2E"
        )
        self.tree_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.tree_frame.configure(fg_color="#2E2E2E")  # Set background color

        # Add a vertical scrollbar
        self.treeview_scrollbar = tk.Scrollbar(self.tree_frame)
        self.treeview_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add a horizontal scrollbar
        self.treeview_x_scrollbar = tk.Scrollbar(
            self.tree_frame, orient=tk.HORIZONTAL)
        self.treeview_x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Create the Treeview
        self.treeview = tk.ttk.Treeview(
            self.tree_frame,
            yscrollcommand=self.treeview_scrollbar.set,
            xscrollcommand=self.treeview_x_scrollbar.set,
            show='tree'  # Use 'tree' to indicate this is a tree structure
        )
        self.treeview.column("#0", width=250, stretch=True)
        self.treeview.pack(fill="both", expand=True)

        # Configure scrollbars
        self.treeview_scrollbar.config(command=self.treeview.yview)
        self.treeview_x_scrollbar.config(command=self.treeview.xview)

        # Populate the tree view with example data
        self.populate_tree("", self.Processor.DataHandler.raw_data)

    def _create_plot_frame(self):
        """Create the frame for the Matplotlib plot."""
        self.plot_frame = tk.Frame(self)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.plot_frame.pack_propagate(False)
        self.plot_frame.configure(width=600, height=400)

    def _create_sample_info_frame(self):
        """Create the frame for sample information and buttons."""
        self.sample_info_frame = ctk.CTkFrame(
            self.plot_frame, height=100, corner_radius=0, bg_color="#2E2E2E")
        self.sample_info_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False)
        self.sample_info_label = ctk.CTkLabel(
            self.sample_info_frame, text="Sample Info: None loaded", anchor="w")
        self.sample_info_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self._create_buttons()

    def _create_buttons(self):
        """Create Back, Next, Integrate buttons, Automation switch, and Export button."""
        # Create a frame for the buttons
        self.button_frame = ctk.CTkFrame(self.sample_info_frame, corner_radius=0)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.X, pady=10)

        # Arrange the buttons in two rows to make it more intuitive

        # Row 1: Back, Next, and Integrate buttons
        self.row1_frame = ctk.CTkFrame(self.button_frame, corner_radius=0, bg_color="#2E2E2E")
        self.row1_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.export_button = ctk.CTkButton(
            self.row1_frame, text="Export", command=self.open_export_dialog)
        self.export_button.pack(side=tk.LEFT, padx=5)

        self.back_button = ctk.CTkButton(self.row1_frame, text="Back", command=self.previous_sample)
        self.back_button.pack(side=tk.LEFT, padx=5)

        self.next_button = ctk.CTkButton(self.row1_frame, text="Next", command=self.next_sample)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.integrate_button = ctk.CTkButton(
            self.row1_frame, text="Integrate", command=self.on_integrate)
        self.integrate_button.pack(side=tk.LEFT, padx=5)

        # Row 2: Automation Switch and Export button
        self.row2_frame = ctk.CTkFrame(self.button_frame, corner_radius=0, bg_color="#2E2E2E")
        self.row2_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.automation_switch = ctk.CTkSwitch(
            self.row2_frame, text="Enable Automation", command=self.toggle_automation)
        self.automation_switch.pack(side=tk.RIGHT, padx=5)

    def _create_parameter_controls(self):
        """Create the frame and controls for parameter sliders."""
        self.parameters_frame = ctk.CTkScrollableFrame(
            self, width=300, corner_radius=0)
        self.parameters_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Smoothing Section
        ctk.CTkLabel(self.parameters_frame, text="Smoothing").pack(pady=10)

        smoothing_frame = ctk.CTkFrame(self.parameters_frame)
        smoothing_frame.pack(pady=10)

        # Add a toggle switch for enabling/disabling smoothing
        self.smoothing_switch = ctk.CTkSwitch(
            smoothing_frame, text="Enable Smoothing", command=self.toggle_smooth)
        self.smoothing_switch.pack(pady=10)

        # Smoothing Kernel Size Slider
        self.kernel_size_slider = ctk.CTkSlider(
            smoothing_frame, from_=1, to=35, command=self.update_preprocessing_parameters)
        self.kernel_size_slider.set(5)  # Default value
        self.kernel_size_slider.pack(pady=5)
        ctk.CTkLabel(smoothing_frame, text="Kernel Size").pack(pady=5)

        # Smoothing Window Length Slider
        self.window_length_slider = ctk.CTkSlider(
            smoothing_frame, from_=10, to=50, command=self.update_preprocessing_parameters)
        self.window_length_slider.set(5)  # Default value
        self.window_length_slider.pack(pady=5)
        ctk.CTkLabel(smoothing_frame, text="Window Length").pack(pady=5)

        # Baseline Section
        baseline_frame = ctk.CTkFrame(self.parameters_frame)
        baseline_frame.pack(pady=10)

        ctk.CTkLabel(baseline_frame, text="Baseline").pack(pady=5)

        # Toggle Subtract Baseline
        self.baseline_switch = ctk.CTkSwitch(
            baseline_frame, text="Subtract Baseline", command=self.toggle_subtract_baseline)
        self.baseline_switch.pack(pady=10)

        # Baseline Distance Slider
        self.baseline_distance_slider = ctk.CTkSlider(
            baseline_frame, from_=0, to=100, command=self.update_preprocessing_parameters)
        self.baseline_distance_slider.set(10)  # Default value
        self.baseline_distance_slider.pack(pady=5)
        ctk.CTkLabel(baseline_frame, text="Baseline Distance").pack(pady=5)

        # Baseline Prominence Slider
        self.baseline_prominence_slider = ctk.CTkSlider(
            baseline_frame, from_=1, to=100, command=self.update_preprocessing_parameters)
        self.baseline_prominence_slider.set(50)  # Default value
        self.baseline_prominence_slider.pack(pady=5)
        ctk.CTkLabel(baseline_frame, text="Baseline Prominence").pack(pady=5)

        # Peaks Section
        peaks_frame = ctk.CTkFrame(self.parameters_frame)
        peaks_frame.pack(pady=10)

        ctk.CTkLabel(peaks_frame, text="Peaks").pack(pady=5)

        # Peak Prominence Slider
        self.min_peak_prominence_slider = ctk.CTkSlider(
            peaks_frame, from_=0, to=100, command=self.update_preprocessing_parameters)
        self.min_peak_prominence_slider.set(50)  # Default value
        self.min_peak_prominence_slider.pack(pady=5)
        ctk.CTkLabel(peaks_frame, text="Minimum Peak Prominence").pack(pady=5)

        self.max_peak_prominence_slider = ctk.CTkSlider(
            peaks_frame, from_=0, to=100, command=self.update_preprocessing_parameters)
        self.max_peak_prominence_slider.set(50)  # Default value
        self.max_peak_prominence_slider.pack(pady=5)
        ctk.CTkLabel(peaks_frame, text="Maximum Peak Prominence").pack(pady=5)

        self.rt_table_button = ctk.CTkButton(
            peaks_frame, text="RT Table", command=self.open_rt_table)
        self.rt_table_button.pack(pady=10)

        # Reset Button
        self.reset_button = ctk.CTkButton(
            self.parameters_frame, text="Reset", command=self.reset
        )
        self.reset_button.pack(side=tk.BOTTOM, pady=10)

    def populate_tree(self, parent, dictionary):
        """Recursively populate the treeview with dictionary keys."""
        for key, value in dictionary.items():
            tree_id = self.treeview.insert(parent, "end", text=key)
            if isinstance(value, dict):
                self.populate_tree(tree_id, value)

        self.treeview.bind("<Double-1>", self.on_treeview_selection)

    def find_raw_data(self, handler_raw_data, sample_name):
        """Finds and returns the RawData object based on the sample name."""
        if isinstance(handler_raw_data, RawData):
            # Check if the current RawData object's signal matches the sample name
            if handler_raw_data.signal == sample_name:
                print(f"Found RawData: {handler_raw_data}")
                return handler_raw_data
        elif isinstance(handler_raw_data, dict):
            for key, value in handler_raw_data.items():
                found = self.find_raw_data(
                    value, sample_name)  # Recursively search
                if found:
                    return found
        return None

    def on_treeview_selection(self, event):
        selected_item = self.treeview.selection()
        if selected_item:
            item_id = selected_item[0]
            # Get the text of the selected item
            sample_name = self.treeview.item(item_id, 'text')
            print(f"Selected item ID: {item_id}, Sample name: {sample_name}")

            # Recursively search for the RawData object starting from the root
            self.current_sample = self.find_raw_data(
                self.Processor.DataHandler.raw_data, sample_name)

            if self.current_sample:
                try:
                    print(
                        f'Attempting to load chromatogram for {self.current_sample}...')
                    self.load_chromatogram(self.current_sample)
                except Exception as e:
                    print(f'Error loading: {self.current_sample}\n{e}')
            else:
                print("RawData object not found.")

    def load_chromatogram(self, raw_data: RawData):
        """Load and display the chromatogram in the plot frame."""
        self.update_sliders()

        # Extract time and signal values
        x, y = raw_data.time_values, raw_data.signal_values
        if self.is_smooth:
            y = self.Processor.smooth(y)

        baseline_y = self.Processor.find_baseline(y, x, self.preprocessing_parameters)
        peaks_x, peaks_y = self.Processor.find_peaks(
            y, x, {'peak_prominence': (self.preprocessing_parameters['peak']['min prominence'],
                                       self.preprocessing_parameters['peak']['max prominence'])})

        # Clear and plot new data
        self._clear_plot()
        self._plot_chromatogram(x, y, baseline_y, peaks_x, peaks_y)

        # Update sample info label
        self.sample_info_label.configure(
            text=f"Sample Info: {raw_data.signal}")

    def _clear_plot(self):
        """Clear the existing plot and prepare for a new one."""
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()
        if hasattr(self, 'toolbar'):
            self.toolbar.destroy()

        # Create new figure and axes
        self.current_fig = Figure()  # figsize=(5, 3), dpi=100)
        self.current_ax = self.current_fig.add_subplot(111)

        # Embed the new figure in the plot frame
        self._embed_plot_in_frame(self.current_fig)

    def _embed_plot_in_frame(self, fig):
        """Embed the Matplotlib figure into the Tkinter frame."""
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()

        # Create a new toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()

        # Pack the new toolbar and canvas
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _plot_chromatogram(self, x, y, baseline_y, peaks_x=None, peaks_y=None):
        """Plot chromatogram, baseline, and peaks."""
        self.current_ax.cla()  # Clear current axes
        self.current_ax.plot(x, y, label='Chromatogram', color='blue')
        self.current_ax.plot(x, baseline_y, label='Baseline', color='orange')
        if peaks_x is not None and peaks_y is not None:
            self.current_ax.plot(peaks_x, peaks_y, marker='o',
                                 color='green', label='Peaks', linestyle='None')

        self.current_ax.legend()

    def update_preprocessing_parameters(self, event=None):
        """Update the preprocessing parameters based on the current slider values."""
        self.preprocessing_parameters["smoothing"] = {
            "is_smooth": self.is_smooth,
            "kernel_size": self.kernel_size_slider.get(),
            "window_length": self.window_length_slider.get()
        }
        self.preprocessing_parameters["baseline"] = {
            "is_baseline_corrected": self.is_baseline_corrected,
            "distance": self.baseline_distance_slider.get(),
            "prominence": self.baseline_prominence_slider.get()
        }
        self.preprocessing_parameters["peak"] = {
            "min prominence": self.min_peak_prominence_slider.get(),
            "max prominence": self.max_peak_prominence_slider.get()
        }

        # Call the Processor's preprocess method
        results = self.Processor.preprocess(
            self.current_sample, self.preprocessing_parameters)
        self.update_plot(results)

    def update_sliders(self, verbose: bool = False):

        baseline_distance_limits = self.Processor.get_baseline_limits(
            self.current_sample, self.preprocessing_parameters)
        if self.baseline_distance_slider.get() > baseline_distance_limits[1]:
            self.baseline_distance_slider.set(baseline_distance_limits[1])
            self.preprocessing_parameters['baseline']['distance'] = baseline_distance_limits[1]
        self.baseline_distance_slider.configure(require_redraw=True,
                                                from_=baseline_distance_limits[0],
                                                to=baseline_distance_limits[1])

        # This may not be necessary, causes issues during automated integration
        max_peak_prominence = self.Processor.get_max_peak_prominence(
            self.current_sample)
        # if self.max_peak_prominence_slider.get() > max_peak_prominence:
        #   self.max_peak_prominence_slider.set(max_peak_prominence)
        self.max_peak_prominence_slider.configure(require_redraw=True,
                                                  to=max_peak_prominence)

        y = self.current_sample.signal_values
        min_peak_prominence = (np.max(y)-np.median(y))//100
        self.min_peak_prominence_slider.configure(require_redraw=True,
                                                  to=min_peak_prominence)

    def open_rt_table(self):
        # Pass the current RT table if it exists, otherwise None
        rt_table = self.preprocessing_parameters.get("RT Table", {})
        RTTableDialog(self, existing_rt_table=[
                      (name, *rt_table[name]) for name in rt_table])

    def open_export_dialog(self):
        ExportDataDialog(self, self.Processor.DataHandler)

    def update_plot(self, results):
        """Update the current plot with new data from preprocessing."""
        x = results['x']
        y = results['y']
        baseline_y = results['baseline_y']
        peaks_x = results['peaks_x']
        peaks_y = results['peaks_y']

        xlim = self.current_ax.get_xlim()
        ylim = self.current_ax.get_ylim()

        # Clear current axes and plot updated data
        self._plot_chromatogram(x, y, baseline_y, peaks_x, peaks_y)
        self.current_ax.set_xlim(xlim)
        self.current_ax.set_ylim(ylim)

        # Redraw the canvas
        self.canvas.draw()

    def shade_plot(self, integration_results: dict):
        """Shade the plot based on integration results."""

        results = self.Processor.preprocess(
            self.current_sample, self.preprocessing_parameters)
        x = results['x']
        y = results['y']
        baseline_y = results['baseline_y']

        self._plot_chromatogram(x, y, baseline_y)  # Re-plot without peaks

        for i in range(len(integration_results['x_peaks'])):
            self.current_ax.fill_between(
                integration_results['x_peaks'][i],
                integration_results['y_peaks'][i],
                integration_results['baseline_peaks'][i],
                alpha=0.9
            )

        self.canvas.draw()

    # Event Handlers

    def toggle_smooth(self):
        self.is_smooth = not self.is_smooth
        self.preprocessing_parameters['smoothing']['is_smooth'] = self.is_smooth
        self.update_sliders()
        self.update_preprocessing_parameters()
        # self.load_chromatogram(self.current_sample)

    def toggle_subtract_baseline(self):
        self.is_baseline_corrected = not self.is_baseline_corrected
        self.preprocessing_parameters['baseline']['is_baseline_corrected'] = self.is_baseline_corrected
        self.update_sliders()
        self.update_preprocessing_parameters()
        self.current_ax.set_xlim()
        self.current_ax.set_ylim()

    def toggle_automation(self):
        """Enable or disable automation based on the switch."""
        self.automation_enabled = not self.automation_enabled
        if not self.automation_enabled:
            self.integrate_button.configure(state='enabled')

    def load_samples(self, detector: str = 'FID1A', sample: int = 1):
        if detector in self.Processor.DataHandler.raw_data.keys():
            self.current_detector = detector
        else:
            raise ValueError(f"Detector {detector} not found in raw data.")

        # Extract sample keys from the specified detector
        self.sample_keys = list(
            self.Processor.DataHandler.raw_data[detector].keys())

        # Set current sample index and load the corresponding sample
        self.current_sample_index = sample - 1  # Make sample index 0-based
        if 0 <= self.current_sample_index < len(self.sample_keys):
            self.current_sample = self.Processor.DataHandler.raw_data[self.current_detector][
                self.sample_keys[self.current_sample_index]]
        else:
            raise IndexError(f"Sample index {sample} is out of range.")

        self.load_chromatogram(self.current_sample)

    def update_indexing(self, event=None):
        """
        Update self.sample_keys and self.current_sample_index
        """
        try:
            self.current_detector = self.current_sample.identify_detector()
            self.sample_keys = list(
                self.Processor.DataHandler.raw_data[self.current_detector].keys())
            self.current_sample_index = self.sample_keys.index(
                self.current_sample.signal)
        except Exception as e:
            print(f'sample indexing got screwed up: {e}')

    def next_sample(self, event=None):
        """
        Move to the next sample, if available.
        """
        self.update_indexing()
        self.update_idletasks()
        self.update()

        if self.current_sample_index < len(self.sample_keys) - 1:
            self.current_sample_index += 1
            sample_key = self.sample_keys[self.current_sample_index]
            self.current_sample = self.Processor.DataHandler.raw_data[
                self.current_detector][sample_key]
            self.load_chromatogram(self.current_sample)
            self.update_preprocessing_parameters()
            return True
        else:
            print("No more samples.")
            return False

    def previous_sample(self, event=None):
        """
        Move to the previous sample, if available.
        """
        if self.current_sample_index > 0:
            self.current_sample_index -= 1
            sample_key = self.sample_keys[self.current_sample_index]
            self.current_sample = self.Processor.DataHandler.raw_data[
                self.current_detector][sample_key]
            self.load_chromatogram(self.current_sample)
            self.update_preprocessing_parameters()
        else:
            print("Already at the first sample.")

    def on_integrate(self, print_output=True):
        """Handle manual integration and start automation if enabled."""
        self.update_preprocessing_parameters()
        result = self.Processor.integrate(
            self.current_sample, self.preprocessing_parameters)

        # Store the results and update the plot
        self.current_sample.peaks_list = result['peaks_list']
        self.shade_plot(result)

        # If automation is enabled, start the integration loop once
        if self.automation_enabled:
            self.integrate_button.configure(state='disabled')  # Disable during automation
            self.integrate_all()  # Start the automation loop

    def integrate_all(self):
        """Automate the integration process for all samples in a separate thread."""
        self.current_method = self.current_sample.identify_method()

        def integration_loop():
            while self.automation_enabled:
                self.update_preprocessing_parameters()
                if not self.next_sample():  # Move to the next sample and check if there are more samples
                    break  # Exit loop if there are no more samples

                current_method = self.current_sample.identify_method()
                if current_method != self.current_method:
                    break

                # Run integration for the current sample
                result = self.Processor.integrate(
                    self.current_sample, self.preprocessing_parameters)

                # Store the results and update the plot
                self.current_sample.peaks_list = result['peaks_list']
                self.shade_plot(result)

            # Re-enable the integrate button and deselect automation after completion
            self.integrate_button.configure(state='enabled')
            self.automation_switch.deselect()

        # Run the integration loop in a separate thread to avoid freezing the GUI
        threading.Thread(target=integration_loop).start()

    def reset(self):
        self.smoothing_switch.deselect()
        self.is_smooth = False
        self.automation_switch.deselect()
        self.automation_enabled = False
        if hasattr(self, 'toolbar'):
            self.toolbar.destroy()
        self.load_chromatogram(self.current_sample)

    def _quit(self):
        self.quit()
        self.destroy()
