import matplotlib.pyplot as plt
from map_factory.dted import get_utm_from_dted, get_korea_dted, get_local_dted
from utility.utility import to_utm


def plot_korea_utm():
    utm_data = get_utm_from_dted(get_korea_dted())
    fig = plt.pcolormesh(
        utm_data["X"], utm_data["Y"], utm_data["Z"], shading="auto")
    fig.axes.set_aspect("equal")
    plt.xlabel("UTM-X, (m)")
    plt.ylabel("UTM-N, (m)")
    plt.title("UTM Map - Korea")
    cbar = plt.colorbar(fig)
    cbar.set_label('Height, m')
    plt.savefig(
        f"simulation/IMGS/UTM/KOREA.png")
    print(f"KOREA.png CREATED")
    plt.clf()


def plot_utm_near_transmitter(transmitter):
    name, t_lon, t_lat, span_lon, span_lat, _ = transmitter()
    utm_data = get_utm_from_dted(
        get_local_dted(t_lon, t_lat, span_lon, span_lat))
    fig = plt.pcolormesh(
        utm_data["X"], utm_data["Y"], utm_data["Z"], shading="auto")
    fig.axes.set_aspect("equal")
    x, y = to_utm(t_lon, t_lat)
    plt.plot(x, y, 'ro', markersize=2)
    plt.xlabel("UTM-X, (m)")
    plt.ylabel("UTM-N, (m)")
    plt.title(f"UTM Map near {name}")
    cbar = plt.colorbar(fig)
    cbar.set_label('Height, m')
    plt.savefig(
        f"simulation/IMGS/UTM/NEAR_{name}.png")
    print(f"NEAR_{name}.png CREATED")
    plt.clf()
