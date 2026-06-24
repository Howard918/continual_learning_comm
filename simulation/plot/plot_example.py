import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from math import sqrt
from utility.utility import *



random_z_matrix = [[ 9,6,8,10,7,0,7,5,9,3,3,0,1,5,8,6,9,7,8,5,0],
 [ 8,3,3,10,6,5,2,0,1,6,5,8,7,1,1,6,0,4,4,1,10],
 [ 9,6,3,3,9,0,7,1,10,3,1,7,0,8,5,10,9,0,3,4,3],
 [ 1,10,3,3,1,8,7,2,3,6,3,8,5,2,7,5,4,7,7,0,8],
 [ 6,9,4,6,3,10,1,6,7,0,0,9,2,10,10,6,10,0,10,8,10],
 [ 6,8,6,4,10,6,4,8,3,0,10,7,3,10,8,10,8,6,8,8,4],
 [ 3,2,8,7,10,0,10,10,3,2,10,0,1,1,4,4,5,7,6,6,6],
 [ 7,2,9,5,2,7,8,10,10,2,1,1,2,9,8,5,6,7,10,1,1],
 [ 0,4,8,3,9,4,7,10,7,1,6,9,9,6,0,10,8,3,9,10,8],
 [10,1,3,2,10,10,1,7,8,10,8,7,8,2,5,1,7,1,4,3,8],
 [ 9,10,1,3,2,0,10,8,6,1,1,4,1,2,7,8,0,7,2,7,4],
 [ 5,2,2,4,7,3,8,0,6,2,10,5,3,6,1,5,8,3,8,2,3],
 [ 9,2,7,1,6,6,9,4,8,5,2,9,7,7,9,6,2,3,10,4,3],
 [ 7,10,2,6,5,6,9,1,7,8,3,4,1,8,2,4,10,3,5,7,4],
 [ 4,5,1,3,10,9,0,10,1,9,4,8,7,7,2,5,0,7,2,6,4],
 [ 1,6,10,8,4,4,2,7,4,10,3,4,3,5,5,5,3,9,10,8,0],
 [ 8,7,9,3,0,0,9,3,0,7,2,7,8,2,9,7,9,0,7,0,4],
 [ 2,10,3,1,0,3,9,3,0,3,7,1,8,3,7,9,10,0,0,7,7],
 [ 2,2,1,1,0,3,3,3,2,1,2,7,0,10,5,3,10,9,9,6,9],
 [ 5,2,1,1,9,9,9,10,10,2,5,1,4,2,5,0,6,5,0,6,4],
 [ 6,10,5,1,1,2,9,8,10,7,5,9,6,10,9,0,3,5,5,3,5],]


def plot_example_of_dots_in_line(t_x,t_y,r_x,r_y):
    dots = [(t_x,t_y)]
    dots += get_dots_in_line(t_x,t_y,r_x,r_y)
    dots += [(r_x,r_y)]
    x_list = [t_x,r_x]
    y_list = [t_y,r_y]

    _, ax =plt.subplots()
    ax.set_xlim([-2,22])
    ax.set_ylim([-2,22])

    x_ticks = np.arange(-1.5, 22.5, 1)
    x_major_ticks = np.arange(-1.5, 22.5, 10)
    y_ticks = np.arange(-1.5, 22.5, 1)
    y_major_ticks =np.arange(-1.5, 22.5, 10)

    ax.set_xticks(x_ticks, minor=True)
    ax.set_xticks(x_major_ticks,)
    ax.set_yticks(y_ticks, minor=True)
    ax.set_yticks(y_major_ticks)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    plt.plot(x_list, y_list)
    for dot in dots:
        ax.add_patch(
        patches.Rectangle(
            (dot[0]-0.5,dot[1]-0.5),
            1,
            1,
            edgecolor = 'blue',
            fill=True, 
            alpha=0.5
        ))
    plt.show()

def plot_example_of_dots(t_x,t_y,r_x,r_y):
    _, ax =plt.subplots()

    x_list = list(range(t_x,r_x))
    y_list = list(range(t_y,r_y))

    print(random_z_matrix)
    x_grid, y_grid = np.meshgrid(x_list, y_list)   # lo

    ax.pcolormesh(x_grid,y_grid,random_z_matrix)

    ax.set_xlim([t_x-2,r_x+1])
    ax.set_ylim([t_y-2,r_y+1])

    x_ticks = np.arange(t_x-1.5, r_x+1.5, 1)
    x_major_ticks = np.arange(t_x-1.5, r_x+1.5, 10)
    y_ticks = np.arange(t_y-1.5, r_y+1.5, 1)
    y_major_ticks =np.arange(t_y-1.5, r_y+1.5, 10)

    ax.set_xticks(x_ticks, minor=True)
    ax.set_xticks(x_major_ticks,)
    ax.set_yticks(y_ticks, minor=True)
    ax.set_yticks(y_major_ticks)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    for dot in [(2,1),(17,18)]:
        ax.add_patch(
        patches.Rectangle(
            (dot[0]-0.5,dot[1]-0.5),
            1,
            1,
            edgecolor = 'red',
            fill=False
        ))
    
    plt.show()


def plot_height_of_line():
    global random_z_matrix
    dots = [(2,1)]
    dots += get_dots_in_line(2,1,17,18)
    dots += [(17,18)]

    x_list = list(range(0,21))
    y_list = list(range(0,21))

    x_grid, y_grid = np.meshgrid(x_list, y_list)   # lo

    _, ax =plt.subplots()
    ax.set_xlim([-2,22])
    ax.set_ylim([-2,22])

    x_ticks = np.arange(-1.5, 22.5, 1)
    x_major_ticks = np.arange(-1.5, 22.5, 10)
    y_ticks = np.arange(-1.5, 22.5, 1)
    y_major_ticks =np.arange(-1.5, 22.5, 10)

    ax.set_xticks(x_ticks, minor=True)
    ax.set_xticks(x_major_ticks,)
    ax.set_yticks(y_ticks, minor=True)
    ax.set_yticks(y_major_ticks)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    t_z = random_z_matrix[1][2]
    r_z = random_z_matrix[18][17]
    print(t_z, r_z)

    random_z_matrix = np.empty((21,21))
    random_z_matrix[:] = np.NaN

    D = sqrt(15*15+17*17)
    for dot in dots:
        d = sqrt((2-dot[0])*(2-dot[0])+(1-dot[1])*(1-dot[1]))
        lz = (r_z-t_z)/D*d +t_z
        random_z_matrix[dot[1]][dot[0]] = lz

    ax.pcolormesh(x_grid,y_grid,random_z_matrix,vmin=0,vmax=10)

    for dot in dots:
        ax.add_patch(
        patches.Rectangle(
            (dot[0]-0.5,dot[1]-0.5),
            1,
            1,
            edgecolor = 'red',
            fill=False
        ))

    plt.show()