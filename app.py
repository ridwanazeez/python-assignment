import pandas as pd
import sqlalchemy as sa
import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.palettes import Category20


class DatabaseError(Exception):
    """Custom exception for database errors."""
    pass


class DataProcessor:
    """Base class for data processing."""

    def __init__(self, db_name):
        """
        Initialize the data processor with a SQLite database.

        Args:
            db_name (str): Name of the SQLite database file.
        """
        self.engine = sa.create_engine(f"sqlite:///{db_name}")

    def load_data(self, data, table_name):
        """
        Load data into the SQLite database.

        Args:
            data (pd.DataFrame): Data to load into the database.
            table_name (str): Name of the table to load the data into.
        """
        try:
            data.to_sql(table_name, self.engine, if_exists="replace", index=False)
        except Exception as e:
            raise DatabaseError(f"Error loading data into {table_name}: {e}")


class IdealFunctionProcessor(DataProcessor):
    """Class for processing ideal functions."""

    def choose_ideal_functions(self):
        """
        Choose the four ideal functions from the database based on the Least Squares criterion.

        Returns:
            list: A list of the four best functions from the ideal functions dataset.
        """
        df_train = pd.read_sql("SELECT * FROM training_data", self.engine)
        df_ideal = pd.read_sql("SELECT * FROM ideal_functions", self.engine)

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


class TestDataMapper(IdealFunctionProcessor):
    """Class for mapping test data to ideal functions."""

    def map_test_data(self, best_functions):
        """
        Map test data to the chosen ideal functions based on the deviation criterion.

        Args:
            best_functions (list): A list of the four best functions from the ideal functions dataset.

        Returns:
            pd.DataFrame: A DataFrame containing the test data mapped to the chosen ideal functions.
        """
        df_test = pd.read_sql("SELECT * FROM test_data", self.engine)
        df_ideal = pd.read_sql("SELECT * FROM ideal_functions", self.engine)

        results = []
        for _, test_row in df_test.iterrows():
            x, y = test_row["x"], test_row["y"]
            min_deviation = float("inf")
            best_function = None
            for function in best_functions:
                try:
                    ideal_y = df_ideal.loc[df_ideal["x"] == x, function].values[0]
                except IndexError:
                    continue
                deviation = abs(y - ideal_y)
                if deviation < min_deviation:
                    min_deviation = deviation
                    best_function = function
            results.append(
                {"x": x, "y": y, "function": best_function, "deviation": min_deviation}
            )

        df_results = pd.DataFrame(results)
        df_results.to_sql("mapped_test_data", self.engine, if_exists="replace", index=False)
        return df_results


class DataVisualizer(DataProcessor):
    """Class for visualizing data."""

    def visualize_data(self, best_functions):
        """
        Visualize the training data, ideal functions, test data, and mapped test data using Bokeh.

        Args:
            best_functions (list): A list of the four best functions from the ideal functions dataset.
        """
        df_train = pd.read_sql('SELECT * FROM training_data', self.engine)
        df_ideal = pd.read_sql('SELECT * FROM ideal_functions', self.engine)
        df_test = pd.read_sql('SELECT * FROM test_data', self.engine)
        df_mapped = pd.read_sql('SELECT * FROM mapped_test_data', self.engine)

        output_file("visualization.html")
        p = figure(
            title="Data Visualization",
            x_axis_label='x', y_axis_label='y',
            width=2000, height=2000
        )

        # Define a palette of colors
        colors = Category20[20]  # Use the Category20 palette for variety

        # Plot training data
        for i, col in enumerate(df_train.columns[1:], start=0):
            p.line(df_train['x'], df_train[col], legend_label=f"Train {col}", line_width=2,
                   line_color=colors[i % len(colors)])

        # Plot ideal functions
        for i, func in enumerate(best_functions, start=len(df_train.columns) - 1):
            p.line(df_ideal['x'], df_ideal[func], legend_label=f"Ideal {func}", line_width=2, line_dash='dashed',
                   line_color=colors[i % len(colors)])

        # Plot test data
        p.scatter(df_test['x'], df_test['y'], legend_label="Test Data", marker="circle", size=8, fill_color="white",
                  line_color="black")

        # Plot mapped data
        p.scatter(df_mapped['x'], df_mapped['y'], legend_label="Mapped Data", marker="square", size=6, fill_color="red",
                  line_color="black")

        show(p)


def main():
    # Load the datasets
    training_data = pd.read_csv("datasets/train.csv")
    test_data = pd.read_csv("datasets/test.csv")
    ideal_functions = pd.read_csv("datasets/ideal.csv")

    # Create data processor objects
    data_processor = DataProcessor("data.db")
    ideal_processor = IdealFunctionProcessor("data.db")
    test_mapper = TestDataMapper("data.db")
    visualizer = DataVisualizer("data.db")

    # Load data into the database
    data_processor.load_data(training_data, "training_data")
    data_processor.load_data(ideal_functions, "ideal_functions")
    data_processor.load_data(test_data, "test_data")

    # Choose ideal functions
    best_functions = ideal_processor.choose_ideal_functions()
    print("Best functions:", best_functions)

    # Map test data
    mapped_data = test_mapper.map_test_data(best_functions)
    print(mapped_data)

    # Visualize data
    visualizer.visualize_data(best_functions)


if __name__ == "__main__":
    main()
