import numpy as np
import pandas as pd
from math import sqrt, atan, log10
from utility.utility import *
from utility.calculator import *
from environments.model import model, scaler
from map_factory.dted import get_korea_dted, get_local_dted, get_height


def get_height_map(t_lon, t_lat):
    dted_data = get_korea_dted()
    index_lon = np.absolute(
        dted_data["grid_lon"] - t_lon).argmin()  # nearest lon index
    index_lat = np.absolute(
        dted_data["grid_lat"] - t_lat).argmin()  # nearest lat index
    return dted_data["grid_height"][index_lat][index_lon]

def get_distance_map(t_lon=127.3845, t_lat=36.3504, span_lon=1.0, span_lat=1.0):
    t_x, t_y = to_utm(t_lon, t_lat)
    t_z = get_height(t_lon, t_lat)
    dted_data = get_local_dted(t_lon, t_lat, span_lon, span_lat)
    utm_x_all = []
    utm_y_all = []
    utm_r_all = []
    for r_iy in range(len(dted_data["grid_lat"])):
        utm_x_row = []
        utm_y_row = []
        utm_r_row = []
        for r_ix in range(len(dted_data["grid_lon"])):
            x, y = to_utm(dted_data["grid_lon"][r_ix],
                          dted_data["grid_lat"][r_iy])
            z = dted_data["grid_height"][r_iy][r_ix]
            r = sqrt((t_x - x)*(t_x - x) + (t_y - y)
                     * (t_y - y) + (t_z - z)*(t_z - z))

            x = int(x)
            y = int(y)
            r = int(r)
            utm_x_row.append(x)
            utm_y_row.append(y)
            utm_r_row.append(r)
        utm_x_all.append(utm_x_row)
        utm_y_all.append(utm_y_row)
        utm_r_all.append(utm_r_row)
        
        print_process(r_iy, len(dted_data["grid_lat"]))
    return {"X": utm_x_all, "Y": utm_y_all, "R": utm_r_all}

def get_slope_map(t_lon=127.3845, t_lat=36.3504, span_lon=1.0, span_lat=1.0):
    dted_data = get_local_dted(t_lon, t_lat, span_lon, span_lat)
    t_x, t_y = to_utm(t_lon, t_lat)
    t_z = get_height(t_lon, t_lat)
    utm_x_all = []
    utm_y_all = []
    utm_slope_all = []
    for r_iy in range(len(dted_data["grid_lat"])):
        utm_x_row = []
        utm_y_row = []
        utm_slope_row = []
        for r_ix in range(len(dted_data["grid_lon"])):
            x, y = to_utm(dted_data["grid_lon"][r_ix],
                          dted_data["grid_lat"][r_iy])
            z = dted_data["grid_height"][r_iy][r_ix]
            tan = (t_z-z)/sqrt((t_x - x)*(t_x - x) + (t_y - y) * (t_y - y))
            slope = atan(tan)
            x = int(x)
            y = int(y)
            utm_x_row.append(x)
            utm_y_row.append(y)
            utm_slope_row.append(slope)
        utm_x_all.append(utm_x_row)
        utm_y_all.append(utm_y_row)
        utm_slope_all.append(utm_slope_row)

        print_process(r_iy, len(dted_data["grid_lat"]))
    return {"X": utm_x_all, "Y": utm_y_all, "S": utm_slope_all}

def get_v_map(f, t_h=10, r_h=10, t_lon=127.3845, t_lat=36.3504, span_lon=1.0, span_lat=1.0):
    dted_data = get_local_dted(t_lon, t_lat, span_lon, span_lat)
    t_ix, t_iy = get_index(dted_data, t_lon, t_lat)

    # MATRIX OF (d, (h,d1,d2)) TUPLE. MidHeight would be none if there is no midheight
    V_ALL = []
    H_ALL = []
    for r_iy in range(len(dted_data["grid_lat"])):
        V_ROW = []
        H_ROW = []
        for r_ix in range(len(dted_data["grid_lon"])):
            V = get_max_v(dted_data, f, t_ix, t_iy, t_h, r_ix, r_iy, r_h)
            V_ROW.append(V["v"])
            H_ROW.append(V["h"])
        V_ALL.append(V_ROW)
        H_ALL.append(H_ROW)
        print_process(r_iy, len(dted_data["grid_lat"]))
    LON, LAT = np.meshgrid(
        dted_data["grid_lon"], dted_data["grid_lat"])   # lon & lat tiles
    return {"X": LON, "Y": LAT, "V": V_ALL, "H": H_ALL}

def get_received_power_map(f, t_h=10, r_h=10, t_lon=127.3845, t_lat=36.3504, span_lon=1.0, span_lat=1.0):
    dted_data = get_local_dted(t_lon, t_lat, span_lon, span_lat)
    t_ix, t_iy = get_index(dted_data, t_lon, t_lat)

    # MATRIX OF (d, (h,d1,d2)) TUPLE. MidHeight would be none if there is no midheight
    RP_all = []
    for r_iy in range(len(dted_data["grid_lat"])):
        RP_row = []
        for r_ix in range(len(dted_data["grid_lon"])):
            RP = get_received_power(dted_data, f, t_ix, t_iy, t_h, r_ix, r_iy, r_h)
            RP_row.append(RP)
        RP_all.append(RP_row)
        print_process(r_iy, len(dted_data["grid_lat"]))
    LON, LAT = np.meshgrid(
        dted_data["grid_lon"], dted_data["grid_lat"])   # lon & lat tiles
    return {"X": LON, "Y": LAT, "RP": RP_all}

def get_friis_power_map(f, t_h=10, r_h=10, t_lon=127.3845, t_lat=36.3504, span_lon=1.0, span_lat=1.0):
    dted_data = get_local_dted(t_lon, t_lat, span_lon, span_lat)
    t_ix, t_iy = get_index(dted_data, t_lon, t_lat)
    # MATRIX OF (d, (h,d1,d2)) TUPLE. MidHeight would be none if there is no midheight
    RP_all = []
    for r_iy in range(len(dted_data["grid_lat"])):
        RP_row = []
        for r_ix in range(len(dted_data["grid_lon"])):
            RP = get_friis_power(dted_data, f, t_ix, t_iy, t_h, r_ix, r_iy, r_h)
            RP_row.append(RP)
        RP_all.append(RP_row)
        print_process(r_iy, len(dted_data["grid_lat"]))
    LON, LAT = np.meshgrid(
        dted_data["grid_lon"], dted_data["grid_lat"])   # lon & lat tiles
    return {"X": LON, "Y": LAT, "RP": RP_all}

def get_observer_predicted_power_map(f, t_h=10, r_h=10, t_lon=127.3845, t_lat=36.3504, span_lon=1.0, span_lat=1.0):
    dted_data = get_local_dted(t_lon, t_lat, span_lon, span_lat)
    t_ix, t_iy = get_index(dted_data, t_lon, t_lat)

    # MATRIX OF (d, (h,d1,d2)) TUPLE. MidHeight would be none if there is no midheight
    RP_all = []
    for r_iy in range(len(dted_data["grid_lat"])):
        RP_row = []
        for r_ix in range(len(dted_data["grid_lon"])):
            RP = get_observer_predicted_power(
                dted_data, f, t_ix, t_iy, t_h, r_ix, r_iy, r_h)
            RP_row.append(RP)
        RP_all.append(RP_row)
        print_process(r_iy, len(dted_data["grid_lat"]))
    LON, LAT = np.meshgrid(
        dted_data["grid_lon"], dted_data["grid_lat"])   # lon & lat tiles
    return {"X": LON, "Y": LAT, "RP": RP_all}

def get_observer_predicted_v_map(f, t_h=10, r_h=10, t_lon=127.3845, t_lat=36.3504, span_lon=1.0, span_lat=1.0):
    dted_data = get_local_dted(t_lon, t_lat, span_lon, span_lat)
    t_ix, t_iy = get_index(dted_data, t_lon, t_lat)

    # MATRIX OF (d, (h,d1,d2)) TUPLE. MidHeight would be none if there is no midheight
    V_all = []
    for r_iy in range(len(dted_data["grid_lat"])):
        V_row = []
        for r_ix in range(len(dted_data["grid_lon"])):
            V = get_observer_predicted_v(
                dted_data, f, t_ix, t_iy, t_h, r_ix, r_iy, r_h)
            V_row.append(V)
        V_all.append(V_row)
        print_process(r_iy, len(dted_data["grid_lat"]))
    LON, LAT = np.meshgrid(
        dted_data["grid_lon"], dted_data["grid_lat"])   # lon & lat tiles
    return {"X": LON, "Y": LAT, "V": V_all}

def get_csv_map(f, t_h=10, r_h=10, t_lon=127.3845, t_lat=36.3504, span_lon=1.0, span_lat=1.0):
    dted_data = get_local_dted(t_lon, t_lat, span_lon, span_lat)
    t_ix, t_iy = get_index(dted_data, t_lon, t_lat)

    RP_all = []
    R_all = []
    D_all = []
    H_all = []
    for r_iy in range(len(dted_data["grid_lat"])):
        RP_row = []
        R_row = []
        D_row = []
        H_row = []
        for r_ix in range(len(dted_data["grid_lon"])):
            RP = get_received_power(dted_data, f, t_ix, t_iy, t_h, r_ix, r_iy, r_h)
            INFO = get_info_about_observer_predicted_power(
                dted_data, t_ix, t_iy, t_h, r_ix, r_iy, r_h)
            RP_row.append(RP)
            R_row.append(INFO['R'])
            D_row.append(INFO['D'])
            H_row.append(INFO['H'])
        RP_all.append(RP_row)
        R_all.append(R_row)
        D_all.append(D_row)
        H_all.append(H_row)
        print_process(r_iy, len(dted_data["grid_lat"]))
    return {"RP": RP_all, "R": R_all, "D": D_all, "H": H_all}

def get_predicted_power_map(f, t_h=10, r_h=10, t_lon=127.3845, t_lat=36.3504, span_lon=1.0, span_lat=1.0):
    dted_data = get_local_dted(t_lon, t_lat, span_lon, span_lat)
    t_ix, t_iy = get_index(dted_data, t_lon, t_lat)
    RP_all = []
    for r_iy in range(len(dted_data["grid_lat"])):
        RP_row = []
        for r_ix in range(len(dted_data["grid_lon"])):
            try:
                INFO = get_info_about_observer_predicted_power(
                    dted_data, t_ix, t_iy, t_h, r_ix, r_iy, r_h)
                # ,R,D,H,F
                INPUT = pd.DataFrame([[INFO["R"], INFO["D"], INFO["H"], log10(f)]], columns=[
                                     "R", 'D', "H", "F"])
                INPUT = scaler.transform(INPUT)
                RP = model.predict(INPUT,verbose=0)[0][0]
                RP_row.append(RP)
            except Exception as e:
                print(e)
                RP_row.append(np.nan)
        RP_all.append(RP_row)
        print_process(r_iy, len(dted_data["grid_lat"]))
    LON, LAT = np.meshgrid(
        dted_data["grid_lon"], dted_data["grid_lat"])   # lon & lat tiles
    return {"X": LON, "Y": LAT, "RP": RP_all}
