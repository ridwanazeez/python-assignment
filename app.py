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


def least_squares(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)


def find_best_fit(training_data, ideal_functions):
    best_fit_functions = {}
    for i in range(1, 5):
        y_train = training_data[f'Y{i}']
        min_deviation = float('inf')
        best_fit = None
        for col in ideal_functions.columns[1:]:
            deviation = least_squares(y_train, ideal_functions[col])
            if deviation < min_deviation:
                min_deviation = deviation
                best_fit = col
        best_fit_functions[f'Y{i}'] = best_fit
    return best_fit_functions


def map_test_data(test_data, best_fit_functions, ideal_functions, training_data):
    mapped_data = []
    sqrt_2 = np.sqrt(2)

    for i, row in test_data.iterrows():
        x_test = row['X']
        y_test = row['Y']
        for key, ideal_func in best_fit_functions.items():
            y_ideal = ideal_functions[ideal_func].values
            y_train = training_data[key].values
            max_train_deviation = np.max(np.abs(y_train - y_ideal))
            test_deviation = np.abs(y_test - y_ideal)
            if test_deviation <= max_train_deviation * sqrt_2:
                mapped_data.append({
                    'X': x_test,
                    'Y': y_test,
                    'Delta Y': test_deviation,
                    'Ideal Function': ideal_func
                })
                break
    return pd.DataFrame(mapped_data)


def visualize_data(training_data, test_data, ideal_functions, best_fit_functions):
    p = figure(title="Training Data vs Ideal Functions", x_axis_label='X', y_axis_label='Y')

    # Plot training data
    for i in range(1, 5):
        p.line(training_data['X'], training_data[f'Y{i}'], legend_label=f'Training Y{i}', line_width=2)

    # Plot ideal functions
    for key, ideal_func in best_fit_functions.items():
        p.line(training_data['X'], ideal_functions[ideal_func], legend_label=f'Ideal {ideal_func}', line_width=2,
               line_dash="dashed")

    # Plot test data
    p.circle(test_data['X'], test_data['Y'], size=10, color="navy", alpha=0.5, legend_label="Test Data")

    output_file("data_visualization.html")
    show(p)


# Example usage
best_fit_functions = find_best_fit(training_data, ideal_functions)
mapped_test_data = map_test_data(test_data, best_fit_functions, ideal_functions, training_data)
visualize_data(training_data, mapped_test_data, ideal_functions, best_fit_functions)
