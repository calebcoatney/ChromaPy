# %%
from tests.test_data_handler import DataHandler
from tests.test_processor import Processor
from tests.test_app import App

# %%
# Initialize the DataHandler
data_handler = DataHandler()

# %%

if __name__ == '__main__':

    # Initialize the Processor
    processor = Processor(data_handler)

    # Initialize the App
    app = App(processor)

    # Start the application main loop
    app.mainloop()
