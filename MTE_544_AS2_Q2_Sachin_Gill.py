"""
Student Name & Last Name: Sachin Gill
Origianl Author : Pi Thanacha Choopojcharoen
You must change the name of your file to MTE_544_AS2_Q2_(your full name).py
Do not use jupyter notebook.

*You may want to install the following libraries if you haven't done so.*

pip install numpy matplotlib pandas scipy

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import random
import pandas as pd

def plot_ellipse(S, b, ax):
    lambda_var = 9.80665
    eigvals, eigvecs = np.linalg.eigh(lambda_var*S)
    aa, bb = eigvals

    theta = np.linspace(0, 2 * np.pi, 100)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    ellipse_param = np.array([aa * unit_circle[0], bb * unit_circle[1]])
    ellipse_points = eigvecs @ ellipse_param + b[:, np.newaxis]

    ax.plot(ellipse_points[0, :], ellipse_points[1, :], 'b-', label='Fitted Ellipse')
    ax.plot(b[0], b[1], 'ro', label='Center')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='best')
    ax.grid(True)

def visualize_data(p, ax, inliers, threshold):
    ax.scatter(p[:, 0], p[:, 1], color='red', alpha=0.5, label='Raw Measurements (Ellipse)')
    ax.scatter(inliers[:, 0], inliers[:, 1], color='purple', alpha=0.7, label='Inliers')

    for point in inliers:
        circle = plt.Circle(point, threshold, color='orange', fill=False, linestyle='--', alpha=0.7)
        ax.add_patch(circle)

def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def fit_ellipse_subset(points):
    # Given some number of points (you have to determined this), 
    # construct an ellipse that fits through those points.
    

    ##### ADD your code here : #####
    ## Determined from part 1a to add 5 points to fit an ellipse
    ## first need to get the x and y points from the input points
    x = points[:, 0]
    y = points[:, 1]

    ## Then build the matrix, same way I did in Question1  

    M = np.column_stack((x**2, 2*x*y, y**2, -2*x, -2*y))
    X = np.array([-1, -1, -1, -1, -1]) 
    A, B, C, D, E = np.linalg.solve(M,X)

    # Convert the polynomial coefficients to Q and b.
    F = np.array([[A, B], [B, C]])
    G = np.array([D, E])

    alpha = 1 / (G.T @ np.linalg.inv(F) @ G - 1)
    Q = alpha * F
    b = np.linalg.solve(Q, alpha * G)

    ##### END #####
    return Q, b

def ransac_ellipse(data, num_iterations=1000, threshold=0.2):
    inliers = []

    # Given the data sets, perform RANSAC to find the best Q and b as well as the inliers
    # Hint: You should use fit_ellipse_subset 
    # Hint: in some case, the Q matrix might not be positive defintie, use is_positive_definite to check.

    ##### ADD your code here : #####
    Q = None
    b = None 
    for i in range(num_iterations):
        # Randomly select 5 points from the dataset
        random_indices = np.random.choice(data.shape[0], 5, replace=False)
        subset = data[random_indices]

        # Fit an ellipse to the subset of points
        Test_Q, Test_b = fit_ellipse_subset(subset)

        # Check if Q is positive definite as mentioned in the hint
        if not is_positive_definite(Test_Q):
            continue

        # Have to apply the inlier check from the assignment instructions to find the inliers
        residuals = []

        # Will calculate the residual for each point in the dataset and check if it is less than the threshold
        for point in data:
            offset = point - Test_b
            quadratic_value = offset.T @ Test_Q @ offset
            residual = np.abs(np.sqrt(quadratic_value) - 1)
            residuals.append(residual)

        residuals = np.array(residuals)
        Test_inliers = data[residuals <= threshold]

        ## update all parameters if the number of inliers is greater than the previous best
        if Q is None or len(Test_inliers) > len(inliers):
            Q = Test_Q
            b = Test_b
            inliers = Test_inliers

    ##### END #####
    return Q,b,inliers

if __name__ == "__main__":
    # Load the data from CSV file and select N random points
    N = 500
    all_data = pd.read_csv('data_x_y.csv').to_numpy()
    dataset = all_data[np.random.choice(all_data.shape[0], N, replace=False), :]
    # dataset is p
    
    Q, b_est, inliers = ransac_ellipse(dataset)
    
    # Plot the raw measurements and fitted ellipse
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    visualize_data(dataset, ax1, inliers, threshold=0.5)
    plot_ellipse(Q, b_est, ax1)
    ax1.set_title("RANSAC Ellipse Fitting with Threshold Visualization")

    plt.show()