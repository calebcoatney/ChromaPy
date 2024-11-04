from ChromaPy.raw_data import RawData
import tkinter as tk
import customtkinter as ctk
import csv
import json
import openpyxl

class ExportDataDialog(ctk.CTkToplevel):
    def __init__(self, parent, data_handler):
        super().__init__(parent)
        self.title("Export Data")
        self.geometry("400x300")

        # Set the window to appear above the main window
        self.transient(parent)
        self.lift()  # Raise the window to the top
        self.focus_set()  # Give it focus immediately

        self.data_handler = data_handler  # ChromatogramAnalyzer instance

        # Add options for file type
        self.file_type_label = ctk.CTkLabel(self, text="File Type:")
        self.file_type_label.pack(pady=5)

        self.file_type_var = tk.StringVar(value=".xlsx")
        self.file_type_menu = ctk.CTkOptionMenu(
            self, variable=self.file_type_var, values=[".csv", ".json", ".xlsx"])
        self.file_type_menu.pack(pady=5)

        # Button to select file location
        self.select_location_button = ctk.CTkButton(
            self, text="Select Export Location", command=self.select_export_location)
        self.select_location_button.pack(pady=10)

        # Export button
        self.export_button = ctk.CTkButton(self, text="Export", command=self.export_data)
        self.export_button.pack(pady=10)

        self.file_path = None
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def select_export_location(self):
        """Open a file dialog for the user to choose the export location."""
        if self.file_type_var.get() == ".csv":
            filetypes = [("CSV files", "*.csv")]
        elif self.file_type_var.get() == ".json":
            filetypes = [("JSON files", "*.json")]
        elif self.file_type_var.get() == ".xlsx":
            filetypes = [("Excel files", "*.xlsx")]
        else:
            filetypes = []

        self.file_path = tk.filedialog.asksaveasfilename(
            defaultextension=self.file_type_var.get(), filetypes=filetypes)

    def export_data(self):
        """Export the data to the selected file in the specified format."""
        if not self.file_path:
            tk.messagebox.showwarning("No File Selected", "Please select an export location.")
            return

        data = self.data_handler.raw_data

        # Export to CSV, JSON, or XLSX based on user selection
        if self.file_type_var.get() == ".csv":
            self.export_to_csv(data)
        elif self.file_type_var.get() == ".json":
            self.export_to_json(data)
        elif self.file_type_var.get() == ".xlsx":
            self.export_to_xlsx(data)
        else:
            tk.messagebox.showinfo(f'File type "{self.file_type_var.get()}" is not supported')

        tk.messagebox.showinfo("Success", "Data exported successfully!")
        self.destroy()  # Close the dialog after exporting

    def export_to_csv(self, data: dict):
        """Export the chromatogram data to CSV."""
        def recursive_export(writer, data):
            """Recursively process the dictionary until RawData objects are found."""
            for key, value in data.items():
                if isinstance(value, RawData):
                    # Check if the RawData object has a peaks list
                    if hasattr(value, 'peaks_list'):
                        # Write the RawData object using its __repr__ method
                        writer.writerow([value.signal])
                        writer.writerow([value.sample])
                        writer.writerow([value.datetime.strftime('%Y-%m-%d %H:%M:%S')])

                        # Write the header for peak data
                        writer.writerow(['Compound ID', 'Peak #', 'Ret Time',
                                        'Integrator', 'Width', 'Area', 'Start Time', 'End Time'])

                        # Write each peak as a row using the `as_row` property
                        for peak in value.peaks_list:
                            writer.writerow(peak.as_row_floats)

                        writer.writerow('')  # Add spacing
                elif isinstance(value, dict):
                    recursive_export(writer, value)

        # Open the file and write data to CSV
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            recursive_export(writer, data)

    def export_to_json(self, data: dict):
        """Export the chromatogram data to JSON."""
        def recursive_export(data):
            """Recursively process the dictionary to build a JSON-compatible structure."""
            json_data = {}
            for key, value in data.items():
                if isinstance(value, RawData):
                    raw_data_dict = {
                        'RawData': repr(value),
                        'Peaks': []
                    }
                    if hasattr(value, 'peaks_list'):
                        for peak in value.peaks_list:
                            raw_data_dict['Peaks'].append({
                                'Compound ID': peak.compound_id,
                                'Peak #': peak.peak_number,
                                'Ret Time': f'{peak.retention_time:.3f}',
                                'Integrator': peak.integrator,
                                'Width': f'{peak.width:.3f}',
                                'Area': f'{peak.area:.0f}',
                                'Start Time': f'{peak.start_time:.3f}',
                                'End Time': f'{peak.end_time:.3f}'
                            })
                    json_data[key] = raw_data_dict
                elif isinstance(value, dict):
                    json_data[key] = recursive_export(value)
            return json_data

        json_data = recursive_export(data)
        with open(self.file_path, mode='w') as file:
            json.dump(json_data, file, indent=4)

    def export_to_xlsx(self, data: dict):
        """Export the chromatogram data to an Excel (.xlsx) file."""
        def recursive_export(worksheet, data, row):
            for key, value in data.items():
                if isinstance(value, RawData):
                    if hasattr(value, 'peaks_list'):
                        worksheet.append([value.signal])
                        worksheet.append([value.sample])
                        worksheet.append([value.datetime.strftime("%m/%d/%Y %H:%M")])
                        worksheet.append(['Compound ID', 'Peak #', 'Ret Time',
                                         'Integrator', 'Width', 'Area', 'Start Time', 'End Time'])

                        for peak in value.peaks_list:
                            worksheet.append(peak.as_row_floats)
                        worksheet.append([''])
                        row += len(value.peaks_list) + 4
                elif isinstance(value, dict):
                    row = recursive_export(worksheet, value, row)
            return row

        workbook = openpyxl.Workbook()
        worksheet = workbook.active
        recursive_export(worksheet, data, row=1)
        workbook.save(self.file_path)

    def on_closing(self):
        self.destroy()
