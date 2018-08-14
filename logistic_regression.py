#!/usr/bin/env python
#encoding: utf-8

import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

class gradient_descent_algorithm:
    def __init__(self,x,y,alpha,total_iteration):
        self.x = x    # x represents input features of each training examples. x is a nxm matrix. 
        self.y = y    # y represents outputs of each training examples. y is a real number. 
        self.alpha = alpha
        self.total_iteration = total_iteration

    def get_cost_function(self,theta):
        m = self.x.shape[1]    # m represents number of training examples. Amount of columns in matrix x is passed to m.
        squared_error = 0
        squared_error_sum = 0
        for mm in range(0,m):
            exponent_of_base_e = -(theta.transpose() * self.x[:,mm])
            exponential_function = np.exp(exponent_of_base_e[0,0])
            hypothesis = 1 / (1 + exponential_function)
            squared_error = self.y[0,mm] * np.log(hypothesis) + (1 - self.y[0,mm]) * np.log(1 - hypothesis)
            squared_error_sum += squared_error
        cost_function = -(squared_error_sum / m)
        return cost_function

    def gradient_descent(self,object):
        n = self.x.shape[0]    # n represents number of features. Amount of rows in matrix x is passed to n. 
        m = self.x.shape[1]    # m represents number of training examples. Amount of columns in matrix x is passed to m. 
        iteration_count = 0
        cost_function   = 0
        theta = np.matrix([[None]]*n)    # theta represents parameters of x of each training examples. theta is a nx1 matrix. 
        for nn in range(0,n):
            theta[nn,0] = 0
        cost_function_vs_iteration_count_plot = plt.subplot(212)
        cost_function_vs_iteration_count_plot.set_xlabel('Iterations of gradient descent')
        cost_function_vs_iteration_count_plot.set_ylabel(r'Cost function J($\Theta$)')
        for iteration in range(self.total_iteration):
            cost_function = object.get_cost_function(theta)
            partial_derivative_of_squared_error_sum = np.matrix([[None]]*n)
            for nn in range(0,n):
                partial_derivative_of_squared_error_sum[nn,0] = 0
            for mm in range(0,m):
                exponent_of_base_e = -(theta.transpose() * self.x[:,mm])
                exponential_function = np.exp(exponent_of_base_e[0,0])
                hypothesis = 1 / (1 + exponential_function)
                cost = hypothesis - self.y[0,mm]
                partial_derivative_of_squared_error = self.x[:,mm] * cost * 2
                partial_derivative_of_squared_error_sum += partial_derivative_of_squared_error
            partial_derivative_of_cost_function = partial_derivative_of_squared_error_sum / (2 * m)
            theta = theta - partial_derivative_of_cost_function * self.alpha
            iteration_count += 1
            print '========================================'
            print 'iteration_count = ' , iteration_count    
            print 'cost_function = '   , cost_function      
            print 'theta = '           , theta              
            cost_function_vs_iteration_count_plot.plot(iteration_count, cost_function, color='black', marker='o', markersize=2)
        print '========================================'
        print 'Learning rate = ', self.alpha            
        print 'Final theta = ', theta                   
        print '========================================'
        return theta

class read_csv_file:
    def __init__(self,csv_file_name):
        self.csv_file_name = csv_file_name

    def read_csv_file_method(self):
        with open(self.csv_file_name) as csvfile:
            data_classification_list = []
            lines = csv.reader(csvfile, delimiter=',')
            number_of_lines = 0
            for line in lines:
                line_list = [1]
                for line_length in range(0,len(line)):
                    line_list.append(float(line[line_length]))
                data_classification_list.append(line_list)
                line_length = len(line_list)
                number_of_lines = number_of_lines + 1
        return number_of_lines, line_length, data_classification_list

class logistic_random_number_generator:
    def __init__(self,number_of_training_examples,number_of_features,data_classification_list):
        self.number_of_training_examples = number_of_training_examples
        self.number_of_features = number_of_features
        self.data_classification_list = data_classification_list

    def logistic_random_number_generator_method(self):
        x = np.matrix([[None]*self.number_of_training_examples]*self.number_of_features)
        y = np.matrix([[None]*self.number_of_training_examples])
        for training_example in range(0,self.number_of_training_examples):
            y[0,training_example] = self.data_classification_list[training_example][line_length-1]
            for feature in range(0,self.number_of_features):
                x[feature,training_example] = self.data_classification_list[training_example][feature]
        return x, y

class plot_hypothesis_and_logistic_random_number:
    def __init__(self,x,y,theta):
        self.x = x
        self.y = y
        self.theta = theta

    def plot_hypothesis_and_logistic_random_number_method(self):
        number_of_training_examples = self.x.shape[1]
        plot_generator = plt.subplot(211)
        plot_generator.set_xlabel('Input x1 of training examples')
        plot_generator.set_ylabel('Input x2 of training examples')
        for training_example in range(0,number_of_training_examples):
            if self.y[0,training_example] == 1:
                plot_generator.plot(self.x[1,training_example], self.x[2,training_example], color='black', marker='x')
            else:
                plot_generator.plot(self.x[1,training_example], self.x[2,training_example], color='black', marker='o')
        theta_2 = self.theta[2,0]
        theta_1 = self.theta[1,0]
        theta_0 = self.theta[0,0]
        x_axis = np.arange(0.,10.,0.02)
        hypothesis = -((theta_0 + theta_1 * x_axis) / theta_2)  # theta_2 * hypothesis + theta_1 * x_axis + theta_0 * 1 = 0 is the Decision Boundary. 
        y_axis = hypothesis
        plot_generator.plot(x_axis, y_axis, color='black', linestyle='solid', linewidth=2)

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
