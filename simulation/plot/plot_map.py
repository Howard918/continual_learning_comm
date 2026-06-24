import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from map_factory.map_factory import get_korea_dted, get_local_dted


def plot_korea(t_lon=127.3845, t_lat=36.3504, span_lon=1.0, span_lat=1.0):
    dted_data = get_korea_dted()
    """ Plot imported DTED Data """
    X_LON, Y_LAT = np.meshgrid(
        dted_data["grid_lon"], dted_data["grid_lat"])  # lon & lat tiles
    fig = plt.pcolormesh(
        X_LON, Y_LAT, dted_data["grid_height"], shading="auto")
    fig.axes.set_aspect("equal")
    fig.axes.add_patch(
        patches.Rectangle(
            (t_lon - span_lon/2, t_lat - span_lat/2),
            span_lon, span_lat,
            edgecolor='red', facecolor='blue', fill=False))

    plt.xlabel("Longitude, degrees E")
    plt.ylabel("Latitude, degrees N")
    plt.title("Elevation Map - Korea")
    cbar = plt.colorbar(fig)
    cbar.set_label('Height, m')
    plt.savefig(f"simulation/IMGS/MAP/KOREA.png")
    print(f"KOREA.png CREATED")
    plt.clf()


def plot_surround_of_transmitter(transmitter):
    """ Plot Near Daejeon """
    name, t_lon, t_lat, span_lon, span_lat, _ = transmitter()
    dted_data = get_local_dted(t_lon, t_lat, span_lon, span_lat)
    plt.figure()
    LON, LAT = np.meshgrid(
        dted_data["grid_lon"], dted_data["grid_lat"])   # lon & lat tiles
    fig = plt.pcolormesh(LON, LAT, dted_data["grid_height"], shading="auto")
    fig.axes.set_aspect("equal")
    plt.plot(t_lon, t_lat, "ro", markersize=2)
    plt.xlabel("Longitude, degrees E")
    plt.ylabel("Latitude, degrees N")
    plt.title(f"Elevation Map - Near {name}")
    cbar = plt.colorbar(fig)
    cbar.set_label('Height, m')
    plt.savefig(f"simulation/IMGS/MAP/{name}.png")
    print(f"{name}.png CREATED")
    plt.clf()