# Machine Learning: Logistic Regression

## Description
- Gradient descent algorithm with vectorized implementation is applied to logistic regression.
- In gradient descent algorithm, parameters corresponding to each input features are simultaneously and repeatedly updated until they are convergent.
- Parameters converge when partial derivative of squared error cost function of each parameters reaches the minimum value.
- In other words, gradient descent algorithm is used to get parameters that minimize the cost function.
- Plotting *cost function - number of iterations* curve is a sufficient way to monitor functionality of gradient descent algorithm. Gradient descent algorithm works correctly when cost function decreases on every iterations. Also a reasonable learning rate in gradient descent algorithm is set based on performance of *cost function - number of iterations* curve.

## Difference Between Linear Regression and Logistic Regression
If gradient descent algorithm is run on logistic regression where hypothesis shows strong non-linearity in cost function, it is not guaranteed that parameters can converge to the global minimum. On the other hand, large amount of local minimums make the cost function "Non-convex". 

### In Cost Function
- Linear Regression
```python
for mm in range(0,m):
    hypothesis = theta.transpose() * self.x[:,mm]
    squared_error = (hypothesis[0,0] - self.y[0,mm]) ** 2
    squared_error_sum += squared_error
cost_function = squared_error_sum / (2 * m)
```

- Logistic Regression
```python
for mm in range(0,m):
    exponent_of_base_e = -(theta.transpose() * self.x[:,mm])
    exponential_function = np.exp(exponent_of_base_e[0,0])
    hypothesis = 1 / (1 + exponential_function)
    squared_error = self.y[0,mm] * np.log(hypothesis) + (1 - self.y[0,mm]) * np.log(1 - hypothesis)
    squared_error_sum += squared_error
cost_function = -(squared_error_sum / m)
```

### In Partial Derivative of Cost Function
- Linear Regression
```python
for mm in range(0,m):
    hypothesis = theta.transpose() * self.x[:,mm]
    cost = (hypothesis - self.y[0,mm])[0,0]
    partial_derivative_of_squared_error = self.x[:,mm] * cost * 2
    partial_derivative_of_squared_error_sum += partial_derivative_of_squared_error
partial_derivative_of_cost_function = partial_derivative_of_squared_error_sum / (2 * m)
```

- Logistic Regression
```python
for mm in range(0,m):
    exponent_of_base_e = -(theta.transpose() * self.x[:,mm])
    exponential_function = np.exp(exponent_of_base_e[0,0])
    hypothesis = 1 / (1 + exponential_function)
    cost = hypothesis - self.y[0,mm]
    partial_derivative_of_squared_error = self.x[:,mm] * cost * 2
    partial_derivative_of_squared_error_sum += partial_derivative_of_squared_error
partial_derivative_of_cost_function = partial_derivative_of_squared_error_sum / (2 * m)
```

## Code Version
Python 2.7.10

## Execution
Learning rate alpha and number of total iterations can be set in `Main function` in the `logistic_regression.py` file.
```python
# Main function
plt.figure()
input_csv_file = sys.argv[1]
R = read_csv_file(input_csv_file)
number_of_lines, line_length, data_classification_list = R.read_csv_file_method()
L = logistic_random_number_generator(number_of_training_examples=number_of_lines, number_of_features=line_length-1, data_classification_list=data_classification_list)
input_x, output_y = L.logistic_random_number_generator_method()
G = gradient_descent_algorithm(x=input_x, y=output_y, alpha=0.005, total_iteration=50000)
final_theta = G.gradient_descent(G)
P = plot_hypothesis_and_logistic_random_number(x=input_x, y=output_y, theta=final_theta)
P.plot_hypothesis_and_logistic_random_number_method()
plt.show()
```
