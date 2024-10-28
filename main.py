# Import necessary classes from ChromaPy modules
from ChromaPy.data_handler import DataHandler
from ChromaPy.processor import Processor
from ChromaPy.app import App

def main():
    # Initialize the DataHandler
    data_handler = DataHandler()

    # Initialize the Processor
    processor = Processor(data_handler)

    # Initialize the App
    app = App(processor)

    # Start the application main loop
    app.mainloop()

if __name__ == '__main__':
    main()