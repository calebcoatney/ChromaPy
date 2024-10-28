# ChromaPy

ChromaPy is a Python application designed for processing gas chromatography (GC) data. It features an intuitive user interface built with CustomTkinter, allowing users to easily integrate, analyze, and export chromatographic data.

## Features

- Sample processing and integration
- Automatic baseline detection and peak identification
- Compound identification using retention time tables
- Visualization of chromatograms with Matplotlib
- Export capabilities to .xlsx, .csv, and .json formats
- User-friendly interface for seamless navigation and parameter adjustment

## Installation

To install ChromaPy, you can use pip. It's recommended to create a virtual environment first.

```bash
pip install -r requirements.txt
```

Alternatively, you can directly install the package if it's available on PyPI (replace `<version>` with the desired version):

```bash
pip install ChromaPy==<version>
```

## Usage

Here's a basic example of how to use ChromaPy:

```python
# Import necessary classes from ChromaPy modules
from ChromaPy.data_handler import DataHandler
from ChromaPy.processor import Processor
from ChromaPy.app import App

# Check if this script is the main program being executed
if __name__ == '__main__':
    # Initialize the DataHandler, which will handle data input and output
    data_handler = DataHandler()

    # Initialize the Processor with the data_handler to manage data processing
    processor = Processor(data_handler)

    # Initialize the App, passing in the processor to handle app logic
    app = App(processor)

    # Start the application main loop to display the GUI
    app.mainloop()
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) for the UI framework.
- [Matplotlib](https://matplotlib.org/) for data visualization.
- [Pandas](https://pandas.pydata.org/) for data manipulation.
- [NumPy](https://numpy.org/) for numerical computing.

## Contact

For questions or suggestions, feel free to reach out:

- Email: [Caleb.Coatney@nrel.gov](mailto:ccoatney@nrel.gov)