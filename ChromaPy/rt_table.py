import tkinter as tk

class RTTableDialog(tk.Toplevel):
    def __init__(self, parent, existing_rt_table=None):
        super().__init__(parent)
        self.title("Enter RT Table")
        self.geometry("400x500")

        # Initialize the RT table
        self.rt_table = existing_rt_table if existing_rt_table else []

        # Frame for the table
        self.frame = tk.Frame(self)
        self.frame.pack(pady=10)

        # Create Treeview for the RT table
        self.tree = tk.ttk.Treeview(self.frame, columns=(
            "Compound Name", "Min RT", "Max RT"), show='headings')
        self.tree.heading("Compound Name", text="Compound Name")
        self.tree.heading("Min RT", text="Min RT")
        self.tree.heading("Max RT", text="Max RT")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Adjust the width for compound name
        self.tree.column("Compound Name", width=150)
        # Adjust the width for Min RT
        self.tree.column("Min RT", width=80, anchor='center')
        # Adjust the width for Max RT
        self.tree.column("Max RT", width=80, anchor='center')

        # Add a vertical scrollbar
        self.scrollbar = tk.ttk.Scrollbar(
            self.frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Load existing data into the Treeview
        self.load_existing_data()

        # Entry fields for adding/editing
        self.compound_entry = tk.Entry(self, width=20)
        self.compound_entry.pack(pady=5)
        self.compound_entry.insert(0, "Compound Name")

        self.min_rt_entry = tk.Entry(self, width=10)
        self.min_rt_entry.pack(pady=5)
        self.min_rt_entry.insert(0, "Min RT")

        self.max_rt_entry = tk.Entry(self, width=10)
        self.max_rt_entry.pack(pady=5)
        self.max_rt_entry.insert(0, "Max RT")

        # Buttons for adding, modifying, and deleting entries
        self.add_button = tk.Button(self, text="Add", command=self.add_row)
        self.add_button.pack(pady=5)

        self.modify_button = tk.Button(
            self, text="Modify", command=self.modify_row)
        self.modify_button.pack(pady=5)

        self.delete_button = tk.Button(
            self, text="Delete", command=self.delete_row)
        self.delete_button.pack(pady=5)

        self.save_button = tk.Button(
            self, text="Save", command=self.save_table)
        self.save_button.pack(pady=10)

        # Bind selection event
        self.tree.bind("<<TreeviewSelect>>", self.on_row_select)

    def load_existing_data(self):
        """Load existing RT table data into the Treeview."""
        for row in self.rt_table:
            self.tree.insert("", "end", values=row)

    def add_row(self):
        """Add a new row to the RT table."""
        compound_name = self.compound_entry.get()
        min_rt = self.min_rt_entry.get()
        max_rt = self.max_rt_entry.get()

        if min_rt and max_rt:
            self.tree.insert("", "end", values=(
                compound_name, float(min_rt), float(max_rt)))
            self.clear_entries()

    def modify_row(self):
        """Modify the selected row in the RT table."""
        selected_item = self.tree.selection()
        if selected_item:
            compound_name = self.compound_entry.get()
            min_rt = self.min_rt_entry.get()
            max_rt = self.max_rt_entry.get()

            self.tree.item(selected_item, values=(
                compound_name, float(min_rt), float(max_rt)))
            self.clear_entries()

    def delete_row(self):
        """Delete the selected row from the RT table."""
        selected_item = self.tree.selection()
        if selected_item:
            self.tree.delete(selected_item)

    def on_row_select(self, event):
        """Populate entry fields with the selected row's data."""
        selected_item = self.tree.selection()
        if selected_item:
            item_values = self.tree.item(selected_item, "values")
            self.compound_entry.delete(0, tk.END)
            self.compound_entry.insert(0, item_values[0])
            self.min_rt_entry.delete(0, tk.END)
            self.min_rt_entry.insert(0, item_values[1])
            self.max_rt_entry.delete(0, tk.END)
            self.max_rt_entry.insert(0, item_values[2])

    def clear_entries(self):
        """Clear the entry fields."""
        self.compound_entry.delete(0, tk.END)
        self.min_rt_entry.delete(0, tk.END)
        self.max_rt_entry.delete(0, tk.END)

    def save_table(self):
        """Save the RT table to the preprocessing parameters."""
        rt_dict = {}
        for item in self.tree.get_children():
            name, min_rt, max_rt = self.tree.item(item, "values")
            rt_dict[name] = (min_rt, max_rt)
        self.master.preprocessing_parameters["RT Table"] = rt_dict
        self.master.update_preprocessing_parameters()
        self.destroy()  # Close the dialog