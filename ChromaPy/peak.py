class Peak:
    def __init__(self, compound_id: str, peak_number: int, retention_time: float, integrator: str, width: float, area: int | float, start_time: float, end_time: float):
        self.compound_id = compound_id  # Name or ID of the compound
        self.peak_number = peak_number  # Peak number in sequence
        self.retention_time = retention_time
        self.integrator = integrator    # Integrator (py)
        self.width = width              # Peak width
        self.area = area                # Peak area
        self.start_time = start_time    # Start of peak
        self.end_time = end_time        # End of peak

    @property
    def as_row(self):
        """Returns the attributes as a list for tabulate."""
        return [self.compound_id, self.peak_number, f'{self.retention_time:.3f}', self.integrator,
                f'{self.width:.3f}', f'{self.area:.0f}', f'{self.start_time:.3f}', f'{self.end_time:.3f}']

    @property
    def as_row_floats(self):
        """Returns the attributes as a list with numbers as floats, rounded as specified."""
        return [self.compound_id,
                self.peak_number,
                round(self.retention_time, 3),
                self.integrator,
                round(self.width, 3),
                round(self.area, 0),
                round(self.start_time, 3),
                round(self.end_time, 3)]