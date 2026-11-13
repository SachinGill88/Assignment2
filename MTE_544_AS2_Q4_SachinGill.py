"""
Student Name & Last Name: Sachin Gill
Origianl Author : Pi Thanacha Choopojcharoen
You must change the name of your file to MTE_544_AS2_Q4_(your full name).py
Do not use jupyter notebook.

*You may want to install the following libraries if you haven't done so.*

pip install numpy matplotlib pandas scipy

"""

import numpy as np

def decode_measurement(measurements):
    readings = []
    for measurement in measurements:
        readings.append(np.argmax(measurement) + 1)  # +1 for 1-based indexing
    return readings

def moodeng_behavior_update(state, A):
    # Given the current state, and the transition matrix A, 
    # randomly return the next state of Moo-Deng based on A
    ##### ADD your code here : #####
    ## need to randomly choose the next location based on the state

    probabilities = A[:, state - 1]  # Get the transition probabilities for the current state, -1 converts to python indices

    next_state = np.random.choice([1, 2, 3], p=probabilities)  # Randomly choose the next state based on probabilities

    ##### END #####
    return next_state


def sensor_measurement(state, C):
    # Given a state, and the matrix C, 
    # randomly return the encoded measurement based on C
    # Note that : 
    # F -> np.array([1,0,0])
    # R -> np.array([0,1,0])
    # P -> np.array([0,0,1])
    ##### ADD your code here : #####
    probabilities = C[:, state - 1]  # Get the measurement probabilities for the current state, -1 converts to python indices

    report_state = np.random.choice([1, 2, 3], p=probabilities)  # Randomly choose the reported state based on probabilities

    ## creating the vector to represent the sensor measurement 
    measurement = np.zeros(3)
    measurement[report_state - 1] = 1  # Set the corresponding index to 1 for the current state

    ##### END #####
    return measurement

def sim_moodeng(initial_state=1,iteration = 20):
    # Given an initial state of Moo-Deng's whereabout and number of iteration,
    # simulate Moo-Deng's behavior and the designed state estimator
    
    belief = np.array([1/3, 1/3, 1/3])  # Initial belief for the Bayesian filter
    
    ##### ADD your code here : #####
    ## Defining the transition matrix A and the measurement matrix C, all work shown in the report
    A = np.array([[0.6, 0.2, 0.2],
                  [0.4, 0.4, 0.1],
                  [0, 0.4, 0.7]]) 
    
    C = np.array([[0.8, 0.2, 0.05],
                  [0.1, 0.7, 0.1],
                  [0.1, 0.1, 0.85]]) 

    
    ##### END #####

    states = []
    measurements = []
    estimated_states = []
    beliefs = []
    state = initial_state
    for i in range(iteration):
               
        ##### ADD your code here : #####
        next_state = moodeng_behavior_update(state, A)
        measurement = sensor_measurement(next_state, C)

        ## Predict
        ## Decided to keep overwriting the updated belief to keep the variables the same as in the template
        updated_belief = A @ belief 

        ## Correct 
        updated_belief = updated_belief * (C.T @ measurement)  ## Correct prediction with the measurement, C.T @ measurement gives the likelihood of the measurement given each state

        ## Normalize 
        updated_belief = updated_belief / np.sum(updated_belief)  # Normalize the belief to sum to 1

        estimated_state = np.argmax(updated_belief) + 1  # Estimate state based on updated belief, get the index of the max value and add 1 for python based indexing

        ## Carry over for the next iteration
        state = next_state
        belief = updated_belief

        ##### END #####
        states.append(next_state)
        measurements.append(measurement)
        beliefs.append(updated_belief)
        estimated_states.append(estimated_state)
        
    return states, measurements, estimated_states, beliefs

    
# Run the simulation

states, measurements, estimated, beliefs = sim_moodeng(initial_state=1, iteration=20)

readings = decode_measurement(measurements)
# Print results
print("True states:          ", states)
print("Sensor measurements:  ", readings)
print("Estimated states:     ", estimated)
print("Belief sequence:      ", np.array(beliefs))
