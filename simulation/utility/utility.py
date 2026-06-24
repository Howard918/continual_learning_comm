import numpy as np
from math import floor, ceil, sqrt, pi, log10
from pyproj import Proj
# Utm zone of Daejeon is 52

to_utm = Proj(proj='utm', zone=52, ellps='WGS84', preserve_units=False)
c = 299792458

def print_process(current_index, whole, name=""):
    percent = int((current_index+1)/whole*100)
    if current_index+1!=whole:
        print(f'\r{name} PROCESSING {percent}%', end='', flush=True)
    else:
        print(f'\r{name} PROCESSING 100%')

def convert_to_si(num):
    K = 1000
    M = 1000000
    G = 1000000000
    T = 1000000000000
    if num < K:
        return f"{num}"
    elif num >= K and num < M:
        return f"{num/K}K"
    elif num >= M and num < G:
        return f"{num/M}M"
    elif num >= G and num < T:
        return f"{num/G}G"
    elif num >= T:
        return f"{num/T}T"

def get_index(dted_data, t_lon, t_lat):
    t_ix = np.absolute(
        dted_data["grid_lon"] - t_lon).argmin()  # nearest lon index
    t_iy = np.absolute(
        dted_data["grid_lat"] - t_lat).argmin()  # nearest lat index
    return (t_ix, t_iy)

def close_bound(bounding_num, center_num):
    bounding_num = float(bounding_num)
    if bounding_num - floor(bounding_num) == 0.5:
        if bounding_num > center_num:
            return floor(bounding_num)
        else:
            return ceil(bounding_num)
    else:
        return round(bounding_num)

def get_range(start, end):
    if start <= end:
        return range(start, end+1)
    else:
        return range(start, end-1, -1)

def get_dots_in_line(t_ix, t_iy, r_ix, r_iy):
    dots_in_line = []

    if t_ix == r_ix:
        dots_in_line += [(t_ix, iy) for iy in get_range(t_iy, r_iy)]
    else:
        a = float(t_iy - r_iy)/float(t_ix-r_ix)
        if r_ix > t_ix:
            dots_in_line += [(t_ix, t_iy+iy)
                             for iy in list(get_range(0, close_bound(0.5*a, 0)))]
            if r_ix-t_ix != 1:
                for ix in get_range(1, r_ix-t_ix-1):
                    by = t_iy + a*ix
                    min_bound = close_bound(by-0.5*a, by)
                    max_bound = close_bound(by+0.5*a, by)
                    dots_in_line += [(t_ix+ix, iy)
                                     for iy in list(get_range(min_bound, max_bound))]
            dots_in_line += [(r_ix, r_iy+iy)
                             for iy in list(get_range(0, close_bound(-0.5*a, 0)))]
        elif r_ix < t_ix:
            dots_in_line += [(t_ix, t_iy+iy)
                             for iy in list(get_range(0, close_bound(-0.5*a, 0)))]
            if t_ix-r_ix != 1:
                for ix in get_range(-1, r_ix-t_ix+1):
                    by = t_iy + a*ix
                    min_bound = close_bound(by-0.5*a, by)
                    max_bound = close_bound(by+0.5*a, by)
                    dots_in_line += [(t_ix+ix, iy)
                                     for iy in list(get_range(min_bound, max_bound))]
            dots_in_line += [(r_ix, r_iy+iy)
                             for iy in list(get_range(0, close_bound(0.5*a, 0)))]

    
    dots_in_line.sort(key=lambda dot:(t_ix-dot[0])*(t_ix-dot[0])+(t_iy-dot[1])*(t_iy-dot[1]))
    return dots_in_line

def get_friis_gain(f, r):
    try:
        return 10 * log10(c*c/((4*pi*r*f) * (4*pi*r*f)))
    except Exception:
        return np.NaN

def get_v_factor(h, f, d1, d2):
    if h == 0:
        return -1
    try:
        return h * sqrt(2*f*(d1+d2)/(c*d1*d2))
    except Exception:
        return np.NaN

def get_loss_by_knife_edge(v):
    try:
        if v > -0.78:
            loss = 6.9 + 20 * log10(sqrt((v-0.1)*(v-0.1)+1) + v - 0.1)
            return loss
        else:
            return 0

    except Exception:
        return np.NaN

def get_received_power_using_r(f,r,v):
    try:
        loss = get_loss_by_knife_edge(v)
        return get_friis_gain(f, r) - loss
    except Exception:
        return np.NaN

def get_received_power_using_raw_data(f, r, h, d1, d2):
    try:
        v = get_v_factor(h, f, d1, d2)
        loss = get_loss_by_knife_edge(v)
        return get_friis_gain(f, r) - loss
    except Exception:
        return np.NaN
