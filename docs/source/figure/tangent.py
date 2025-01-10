import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_tangent_space_plot():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    scale = 2.5
    resolution = 150
    z_offset = -1.5
    x_offset = -1.5
    
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = scale * np.outer(np.cos(u), np.sin(v)) + x_offset
    y = scale * np.outer(np.sin(u), np.sin(v))
    z = scale * np.outer(np.ones(np.size(u)), np.cos(v)) + z_offset
    
    ax.plot_surface(x, y, z, alpha=1.0, color='royalblue', antialiased=True)

    point = scale * np.array([1/np.sqrt(2.5), 1/np.sqrt(2.5), 1/np.sqrt(5)]) + np.array([x_offset, 0, z_offset])
    v1 = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])
    v2 = np.array([-1/np.sqrt(10), -1/np.sqrt(10), 2*np.sqrt(3/10)])
    
    grid_size = 1.0
    grid_points = 15
    xx = np.linspace(-grid_size, grid_size, grid_points)
    yy = np.linspace(-grid_size, grid_size, grid_points)
    XX, YY = np.meshgrid(xx, yy)
    plane_points = np.zeros((XX.shape[0], XX.shape[1], 3))
    
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            plane_points[i,j] = point + scale * (XX[i,j]*v1 + YY[i,j]*v2)
    
    ax.plot_surface(plane_points[:,:,0], plane_points[:,:,1], 
                   plane_points[:,:,2], alpha=0.6, color='crimson',
                   antialiased=True)
    
    ax.scatter([point[0]], [point[1]], [point[2]], 
              color='black', s=150, alpha=0.4, zorder=100)

    ax.set_xlim(-3.5, 1.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-2.5, 1.5)
    
    ax.set_axis_off()
    ax.view_init(elev=20, azim=-30)
    ax.set_box_aspect([1,1,0.8])
    
    return fig

fig = create_tangent_space_plot()
plt.show()