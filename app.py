import pandas as pd
import sqlalchemy as sa
import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_file

# Load the datasets
training_data = pd.read_csv("datasets/train.csv")
test_data = pd.read_csv("datasets/test.csv")
ideal_functions = pd.read_csv("datasets/ideal.csv")

# Create SQLite database
engine = sa.create_engine("sqlite:///data.db")


def load_data_to_db():
    """
    Load training data, ideal functions, and test data into the SQLite database.
    """
    # Load training data into database
    training_data.to_sql("training_data", engine, if_exists="replace", index=False)

    # Load ideal functions into database
    ideal_functions.to_sql("ideal_functions", engine, if_exists="replace", index=False)

    # Load test data into database
    test_data.to_sql("test_data", engine, if_exists="replace", index=False)

    print("Data loaded into the database successfully.")


# Call the function to load data
load_data_to_db()


def choose_ideal_functions(engine):
    """
    Choose the four ideal functions from the database based on the Least Squares criterion.

    Args:
        engine (sqlalchemy.engine.Engine): The SQLite database engine.

    Returns:
        list: A list of the four best functions from the ideal functions dataset.
    """
    # Read the training and ideal functions from the database
    df_train = pd.read_sql("SELECT * FROM training_data", engine)
    df_ideal = pd.read_sql("SELECT * FROM ideal_functions", engine)

    best_functions = []
    for y_col in df_train.columns[1:]:
        min_deviation = float("inf")
        best_function = None
        for ideal_col in df_ideal.columns[1:]:
            deviation = np.sum((df_train[y_col] - df_ideal[ideal_col]) ** 2)
            if deviation < min_deviation:
                min_deviation = deviation
                best_function = ideal_col
        best_functions.append(best_function)
    return best_functions


# Use the function
best_functions = choose_ideal_functions(engine)
print("Best functions:", best_functions)


def map_test_data(engine, best_functions):
    """
    Map test data to the chosen ideal functions based on the deviation criterion.

    Args:
        engine (sqlalchemy.engine.Engine): The SQLite database engine.
        best_functions (list): A list of the four best functions from the ideal functions dataset.

    Returns:
        pandas.DataFrame: A DataFrame containing the test data mapped to the chosen ideal functions.
    """
    df_test = pd.read_sql("SELECT * FROM test_data", engine)
    df_ideal = pd.read_sql("SELECT * FROM ideal_functions", engine)

    results = []
    for _, test_row in df_test.iterrows():
        x, y = test_row["x"], test_row["y"]
        min_deviation = float("inf")
        best_function = None
        for function in best_functions:
            ideal_y = df_ideal.loc[df_ideal["x"] == x, function].values[0]
            deviation = abs(y - ideal_y)
            if deviation < min_deviation:
                min_deviation = deviation
                best_function = function
        results.append(
            {"x": x, "y": y, "function": best_function, "deviation": min_deviation}
        )

    df_results = pd.DataFrame(results)
    df_results.to_sql("mapped_test_data", engine, if_exists="replace", index=False)
    print("Test data mapped and saved successfully.")
    return df_results


# Use the function
mapped_data = map_test_data(engine, best_functions)
print(mapped_data)


def visualize_data(engine, best_functions):
    """
    Visualize the training data, ideal functions, test data, and mapped test data using Bokeh.

    Args:
        engine (sqlalchemy.engine.Engine): The SQLite database engine.
        best_functions (list): A list of the four best functions from the ideal functions dataset.
    """
    df_train = pd.read_sql('SELECT * FROM training_data', engine)
    df_ideal = pd.read_sql('SELECT * FROM ideal_functions', engine)
    df_test = pd.read_sql('SELECT * FROM test_data', engine)
    df_mapped = pd.read_sql('SELECT * FROM mapped_test_data', engine)

    output_file("visualization.html")
    p = figure(title="Data Visualization", x_axis_label='x', y_axis_label='y')

    # Plot training data
    for col in df_train.columns[1:]:
        p.line(df_train['x'], df_train[col], legend_label=f"Train {col}", line_width=2)

    # Plot ideal functions
    for func in best_functions:
        p.line(df_ideal['x'], df_ideal[func], legend_label=f"Ideal {func}", line_width=2, line_dash='dashed')

    # Plot test data
    p.scatter(df_test['x'], df_test['y'], legend_label="Test Data", marker="circle", size=8, fill_color="white")

    # Plot mapped data
    p.scatter(df_mapped['x'], df_mapped['y'], legend_label="Mapped Data", marker="square", size=6, fill_color="red")

    show(p)


# Use the function
visualize_data(engine, best_functions)
