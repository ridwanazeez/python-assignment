import pandas as pd
import sqlalchemy as sa
import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import gridplot

# Load the datasets
training_data = pd.read_csv("datasets/train.csv")
test_data = pd.read_csv("datasets/test.csv")
ideal_functions = pd.read_csv("datasets/ideal.csv")

# Create SQLite database
engine = sa.create_engine('sqlite:///data.db')
conn = engine.connect()

# Load training data into database
training_data.to_sql('training_data', conn, if_exists='replace')

# Load ideal functions into database
ideal_functions.to_sql('ideal_functions', conn, if_exists='replace')

# Load test data into database
test_data.to_sql('test_data', conn, if_exists='replace')


def load_data_to_db():
    # Load the training data
    training_data.to_sql('training_data', engine, if_exists='replace', index=False)

    # Load the ideal functions
    ideal_functions.to_sql('ideal_functions', engine, if_exists='replace', index=False)

    # Load the test data
    test_data.to_sql('test_data', engine, if_exists='replace', index=False)

    print("Data loaded into the database successfully.")


# Call the function to load data
load_data_to_db()
