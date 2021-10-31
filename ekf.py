import numpy as np
import matplotlib.pyplot as plt

def find_covariance(x_actual, x_pred, u):
    d_e = ((x_actual[0]-x_pred[0]) + (x_actual[1]-x_pred[1]))/2
    a_e = np.arctan2(x_actual[1]-x_pred[1], x_actual[0]-x_pred[0])

    u[0] = u[0] + a_e
    u[1] = u[1] + d_e

    E = np.sqrt((x_actual[0]-x_pred[0])**2 + (x_actual[1]-x_pred[1])**2)
    return u, E

def predict_state(x, u):
    x_x = x[0] + u[1] * np.cos(u[0])
    x_y = x[1] + u[1] * np.sin(u[0])
    x_pred = np.array([[x_x],[x_y]])
    return x_pred

def run_ekf(States, Features):
    x0 = States[0]
    u = [0, 1]  # direction, step
    covariances = []
    x_pred = predict_state(x0, u)
    u, covariance = find_covariance(States[1], x_pred, u)
    covariances.append(covariance)
    for index in range(States.shape[0]):
        if index == 0:
            continue
        else:
            x_pred = predict_state(States[index], u)
            u, covariance = find_covariance(States[1], x_pred, u)

    plot(np.transpose(States), np.transpose(Features), np.array(covariances))



def plot(Cameras, Final_Point_Cloud, Covariances):
    fig = plt.figure()
    fig.suptitle('3D reconstructed', fontsize=16)
    ax = fig.gca(projection='3d')
    # Plot point cloud
    ax.plot(Final_Point_Cloud[0], Final_Point_Cloud[1], Final_Point_Cloud[2], 'b.')

    # Plot Camera
    ax.plot(Cameras[0], Cameras[1], Cameras[2], "r.")
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(elev=135, azim=90)
    plt.show()