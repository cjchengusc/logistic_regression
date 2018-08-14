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
python logistic_regression.py data_classification_vertical.csv
```
The output will look like this
```
========================================
iteration_count =  1
cost_function =  0.69314718056
theta =  [[-0.00019230769230769233]
 [-0.009230769230769232]
 [-0.0009615384615384616]]
========================================
iteration_count =  2
cost_function =  0.676519288936
theta =  [[-0.00030918019147674325]
 [-0.017847564941860506]
 [-0.0015338962629170149]]
========================================
iteration_count =  3
cost_function =  0.662103355042
theta =  [[-0.0003579765741462289]
 [-0.025906171607212605]
 [-0.0017587889296733499]]
========================================
iteration_count =  4
cost_function =  0.64952037293
theta =  [[-0.00034549043057493844]
 [-0.033458495405739176]
 [-0.0016745717876290522]]
========================================
......
......
========================================
iteration_count =  49997
cost_function =  0.0894921615779
theta =  [[7.01038935458141]
 [-1.4187258604850272]
 [0.11329408203838266]]
========================================
iteration_count =  49998
cost_function =  0.0894913907779
theta =  [[7.010450601861184]
 [-1.4187359612783175]
 [0.11329326808025945]]
========================================
iteration_count =  49999
cost_function =  0.0894906199998
theta =  [[7.0105118482683775]
 [-1.41874606194367]
 [0.11329245414626196]]
========================================
iteration_count =  50000
cost_function =  0.0894898492436
theta =  [[7.010573093803024]
 [-1.4187561624810892]
 [0.11329164023638898]]
========================================
Learning rate =  0.005
Final theta =  [[7.010573093803024]
 [-1.4187561624810892]
 [0.11329164023638898]]
========================================
```
![image](https://github.com/cjchengusc/logistic_regression/blob/master/logistic_regression_convergent_vertical.png)

### Horizontal Training Sets
Run the `logistic_regression.py` file using Python
```
python logistic_regression.py data_classification_horizontal.csv
```
The output will look like this
```
========================================
iteration_count =  1
cost_function =  0.69314718056
theta =  [[-0.00022727272727272727]
 [-0.0013636363636363635]
 [0.005681818181818183]]
========================================
iteration_count =  2
cost_function =  0.686401939111
theta =  [[-0.0004795424413853487]
 [-0.002853395415555973]
 [0.011167639665363521]]
========================================
iteration_count =  3
cost_function =  0.680009506816
theta =  [[-0.0007545909499011492]
 [-0.004453786680807707]
 [0.016471064209875545]]
========================================
iteration_count =  4
cost_function =  0.67393095556
theta =  [[-0.00105039279318794]
 [-0.006150770990091652]
 [0.0216046002294948]]
========================================
......
......
========================================
iteration_count =  49997
cost_function =  0.0989548941811
theta =  [[-7.484104900809093]
 [-0.09174230176556121]
 [1.4996110667409677]]
========================================
iteration_count =  49998
cost_function =  0.098954032538
theta =  [[-7.484169639521039]
 [-0.09174165662410082]
 [1.4996218652264213]]
========================================
iteration_count =  49999
cost_function =  0.0989531709192
theta =  [[-7.484234377315073]
 [-0.09174101150164435]
 [1.4996326635734638]]
========================================
iteration_count =  50000
cost_function =  0.0989523093248
theta =  [[-7.484299114191226]
 [-0.09174036639819089]
 [1.4996434617820995]]
========================================
Learning rate =  0.005
Final theta =  [[-7.484299114191226]
 [-0.09174036639819089]
 [1.4996434617820995]]
========================================
```
![image](https://github.com/cjchengusc/logistic_regression/blob/master/logistic_regression_convergent_horizontal.png)

### Diagonal Training Sets
Run the `logistic_regression.py` file using Python
```
python logistic_regression.py data_classification_diagonal.csv
```
The output will look like this
```
========================================
iteration_count =  1
cost_function =  0.69314718056
theta =  [[-0.00019230769230769233]
 [-0.007307692307692308]
 [-0.004807692307692308]]
========================================
iteration_count =  2
cost_function =  0.678537920112
theta =  [[-0.00029957286293549586]
 [-0.01397741443639735]
 [-0.009130158655111069]]
========================================
iteration_count =  3
cost_function =  0.666477215717
theta =  [[-0.00032998325876452874]
 [-0.020070293703232753]
 [-0.013014807470125591]]
========================================
iteration_count =  4
cost_function =  0.656505145071
theta =  [[-0.000291094697714786]
 [-0.0256430205883506]
 [-0.01650549023958364]]
========================================
......
......
========================================
iteration_count =  49997
cost_function =  0.141062280168
theta =  [[8.528115687044226]
 [-0.856394059756624]
 [-0.8041648717043091]]
========================================
iteration_count =  49998
cost_function =  0.1410611872
theta =  [[8.528188973740559]
 [-0.8564008279491652]
 [-0.8041717984496991]]
========================================
iteration_count =  49999
cost_function =  0.141060094262
theta =  [[8.528262259412395]
 [-0.856407596049623]
 [-0.8041787250993238]]
========================================
iteration_count =  50000
cost_function =  0.141059001354
theta =  [[8.528335544059772]
 [-0.8564143640580005]
 [-0.8041856516531862]]
========================================
Learning rate =  0.005
Final theta =  [[8.528335544059772]
 [-0.8564143640580005]
 [-0.8041856516531862]]
========================================
```
![image](https://github.com/cjchengusc/logistic_regression/blob/master/logistic_regression_convergent_diagonal.png)

## Reference
Andrew Ng, [Machine Learning](https://www.coursera.org/learn/machine-learning), Stanford University Coursera
