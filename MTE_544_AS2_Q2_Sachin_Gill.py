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

def plot_ellipse(Q, b, ax):
    S = scipy.linalg.sqrtm(np.linalg.inv(Q))
    
    eigvals, eigvecs = np.linalg.eigh(S)
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

    ## Then build the matrix, same way I did in Question 1  

    M = np.column_stack((x**2, 2*x*y, y**2, -2*x, -2*y))
    X = np.array([-1, -1, -1, -1, -1]) 

    ## solve the linear system of equations to get the coefficients A, B, C, D, E
    A, B, C, D, E = np.linalg.solve(M,X)

    # Convert the polynomial coefficients to Q and b.
    F = np.array([[A, B], [B, C]])
    G = np.array([D, E])

    alpha = 1 / (G.T @ np.linalg.inv(F) @ G - 1)
    Q = alpha * F
    b = np.linalg.solve(Q, alpha * G)

    ##### END #####
    return Q, b

def ransac_ellipse(data, num_iterations=1000, threshold=0.5):
    inliers = []

    ##### ADD your code here : #####
    Q = None
    b = None

    data_min = data.min(axis=0)
    data_max = data.max(axis=0)

    for i in range(num_iterations):
        # Randomly select 5 points from the dataset
        random_indices = np.random.choice(data.shape[0], 5, replace=False)
        subset = data[random_indices]

        # Fit an ellipse to the subset of points
        Test_Q, Test_b = fit_ellipse_subset(subset)

        ## Check if Q is positive definite as mentioned in the hint 
        if not is_positive_definite(Test_Q):
            continue

        ## I added this to avoid huge ellipses that are not relevant to the data
        if np.any(Test_b < data_min) or np.any(Test_b > data_max):
            continue

        # Apply the assignment's inlier check
        residuals = []

        ## Looped through the data, you can also use whole dataset to speed this up but I kept it simple for clarity
        for point in data:
            offset = point - Test_b
            quadratic_value = offset.T @ Test_Q @ offset
            residual = np.abs(np.sqrt(quadratic_value) - 1)
            residuals.append(residual)

        residuals = np.array(residuals)
        Test_inliers = data[residuals <= threshold]

        # Update if these test values have more inliers
        if len(Test_inliers) > len(inliers):
            Q = Test_Q
            b = Test_b
            inliers = Test_inliers

    ##### END #####
    return Q, b, inliers

if __name__ == "__main__":
    N = 500
    all_data = pd.read_csv('data_x_y.csv').to_numpy()
    dataset = all_data[np.random.choice(all_data.shape[0], N, replace=False), :]
    # dataset is p
    
    Q, b_est, inliers = ransac_ellipse(dataset)

    ## Couple of parameters to help with the comparison (part C)
    print (len(inliers)/len(dataset)) ## % inliers
    print (b_est[0], b_est[1]) ## estimated center of the ellipse

    
    # Plot the raw measurements and fitted ellipse
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    visualize_data(dataset, ax1, inliers, threshold=0.2)
    plot_ellipse(Q, b_est, ax1)
    ax1.set_title("RANSAC Ellipse Fitting with Threshold Visualization")
    plt.show()