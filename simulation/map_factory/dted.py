import numpy as np
import pandas as pd
from utility.utility import to_utm


def get_korea_dted():
    grid_lon_all = pd.read_csv("./DTED_Lv0/longrid.dat",
                               header=None)   # degrees, E
    grid_lat_all = pd.read_csv("./DTED_Lv0/latgrid.dat",
                               header=None)   # degrees, N
    grid_height_all = pd.read_csv("./DTED_Lv0/dted_korea.dat",          # meters
                                  sep="  ", header=None, engine='python')

    # Store as numpy arrays
    grid_lon_all = np.asarray(grid_lon_all).reshape(-1, 1)
    grid_lat_all = np.asarray(grid_lat_all).reshape(-1, 1)
    grid_height_all = np.asarray(grid_height_all).reshape(
        len(grid_lat_all), len(grid_lon_all))

    return {"grid_lon": grid_lon_all, "grid_lat": grid_lat_all, "grid_height": grid_height_all}


def get_local_dted(t_lon=127.3845, t_lat=36.3504, span_lon=1.0, span_lat=1.0):
    dted_data = get_korea_dted()

    ndx_lon_Tx = np.absolute(
        dted_data["grid_lon"] - t_lon).argmin()  # nearest lon index
    ndx_lat_Tx = np.absolute(
        dted_data["grid_lat"] - t_lat).argmin()  # nearest lat index
    # nearest longitude
    Tx_lon = dted_data["grid_lon"][ndx_lon_Tx]
    # nearest latitude
    Tx_lat = dted_data["grid_lat"][ndx_lat_Tx]
    ndx_lon = np.nonzero(
        np.abs(dted_data["grid_lon"]-Tx_lon).ravel() < span_lon/2)
    ndx_lat = np.nonzero(
        np.abs(dted_data["grid_lat"]-Tx_lat).ravel() < span_lat/2)

    return {"grid_lon": dted_data["grid_lon"][ndx_lon],
            "grid_lat": dted_data["grid_lat"][ndx_lat],
            "grid_height": dted_data["grid_height"][np.min(ndx_lat):np.max(ndx_lat)+1,
                                                    np.min(ndx_lon):np.max(ndx_lon)+1]}


def get_utm_from_dted(dted_data):
    utm_x_all = []
    utm_y_all = []
    for i in range(len(dted_data["grid_lat"])):
        utm_x_row = []
        utm_y_row = []
        for j in range(len(dted_data["grid_lon"])):
            x, y = to_utm(dted_data["grid_lon"][j],
                          dted_data["grid_lat"][i])
            x = int(x)
            y = int(y)
            utm_x_row.append(x)
            utm_y_row.append(y)
        utm_x_all.append(utm_x_row)
        utm_y_all.append(utm_y_row)

    return {"X": utm_x_all, "Y": utm_y_all, "Z": dted_data["grid_height"]}


def get_height(t_lon, t_lat):
    dted_data = get_korea_dted()
    index_lon = np.absolute(
        dted_data["grid_lon"] - t_lon).argmin()
    index_lat = np.absolute(
        dted_data["grid_lat"] - t_lat).argmin()
    return dted_data["grid_height"][index_lat][index_lon]
