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

### Random Training Sets
Run the `logistic_regression.py` file using Python
```
python logistic_regression.py data_classification_random.csv
```
The output will look like this
```
========================================
iteration_count =  1
cost_function =  0.69314718056
theta =  [[0.00025]
 [0.005900488155428321]
 [0.004079350796993822]]
========================================
iteration_count =  2
cost_function =  0.683240419519
theta =  [[0.00043422666344872474]
 [0.011377430047002715]
 [0.007802079627621637]]
========================================
iteration_count =  3
cost_function =  0.674799257705
theta =  [[0.0005579588799096167]
 [0.016464402349599578]
 [0.011197426574921444]]
========================================
iteration_count =  4
cost_function =  0.667600231323
theta =  [[0.0006261240622279723]
 [0.021192826656498426]
 [0.014292735368931124]]
========================================
......
......
========================================
iteration_count =  49997
cost_function =  0.255243843291
theta =  [[-7.628616336071029]
 [0.8630933357469219]
 [0.6863890728329858]]
========================================
iteration_count =  49998
cost_function =  0.255243075811
theta =  [[-7.628677705074991]
 [0.8630996000575065]
 [0.686394721562345]]
========================================
iteration_count =  49999
cost_function =  0.255242308356
theta =  [[-7.628739073051786]
 [0.8631058642650018]
 [0.686400370200373]]
========================================
iteration_count =  50000
cost_function =  0.255241540928
theta =  [[-7.628800440001449]
 [0.8631121283694113]
 [0.6864060187470727]]
========================================
Learning rate =  0.005
Final theta =  [[-7.628800440001449]
 [0.8631121283694113]
 [0.6864060187470727]]
========================================
```
![image](https://github.com/cjchengusc/logistic_regression/blob/master/logistic_regression_convergent_random.png)

### Vertical Training Sets 
Run the `logistic_regression.py` file using Python
```

```
The output will look like this
```

```
![image]()

### Horizontal Training Sets
Run the `logistic_regression.py` file using Python
```

```
The output will look like this
```

```
![image]()

### Diagonal Training Sets
Run the `logistic_regression.py` file using Python
```

```
The output will look like this
```

```
![image]()
