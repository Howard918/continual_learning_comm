
from utility.utility import *

def get_max_v(dted_data, f, t_ix, t_iy, t_h, r_ix, r_iy, r_h):
    if abs(t_ix - r_ix)<= 3 and abs(t_iy - r_iy)<=3:
        return {"v": np.nan, "h": np.nan, "d1": np.nan, "d2":np.nan}

    dots_in_line = get_dots_in_line(t_ix, t_iy, r_ix, r_iy)
    t_x, t_y = to_utm(dted_data["grid_lon"][t_ix], dted_data["grid_lat"][t_iy])
    t_z = dted_data["grid_height"][t_iy][t_ix]
    r_x, r_y = to_utm(dted_data["grid_lon"][r_ix], dted_data["grid_lat"][r_iy])
    r_z = dted_data["grid_height"][r_iy][r_ix]
    distance = sqrt((t_x-r_x)*(t_x-r_x) + (t_y-r_y)
                    * (t_y-r_y) + (t_z + t_h - r_z - r_h) * (t_z + t_h - r_z - r_h))
    D = sqrt((t_x-r_x)*(t_x-r_x) + (t_y-r_y)*(t_y-r_y))
    DZ = r_z + r_h - t_z - t_h

    # v_info : {"v": float, "h': float, "d1": float, "d2": float}
    v_info_list = []
    for dot in dots_in_line:
        x, y = to_utm(dted_data["grid_lon"][dot[0]],
                      dted_data["grid_lat"][dot[1]])
        z = dted_data["grid_height"][dot[1]][dot[0]]
        d = sqrt((x-t_x) * (x-t_x) + (y-t_y)*(y-t_y))
        lz = t_z + t_h + DZ / D * d
        v = get_v_factor(z-lz, f, d, D-d)
        if np.isnan(v):
            v_info = {"v": -100, "h": np.nan, "d1": np.nan, "d2": np.nan}
        else:
            v_info = {"v": v, "h": z-lz, "d1": d, "d2": D-d}
        v_info_list.append(v_info)
    if len(v_info_list) == 0:
        return {"v": np.nan, "h": np.nan, "d1": np.nan, "d2": np.nan}
    else:
        max_V_INDEX = np.nanargmax([v["v"] for v in v_info_list])
        max_V =v_info_list[max_V_INDEX]
        return max_V

def get_observer_predicted_v(dted_data, f, t_ix, t_iy, t_h, r_ix, r_iy, r_h):
    if abs(t_ix - r_ix)<= 3 and abs(t_iy - r_iy)<=3:
        return np.nan

    dots_in_line = get_dots_in_line(t_ix, t_iy, r_ix, r_iy)
    t_x, t_y = to_utm(dted_data["grid_lon"][t_ix], dted_data["grid_lat"][t_iy])
    t_z = dted_data["grid_height"][t_iy][t_ix] + t_h
    r_x, r_y = to_utm(dted_data["grid_lon"][r_ix], dted_data["grid_lat"][r_iy])
    r_z = dted_data["grid_height"][r_iy][r_ix] + r_h
    D = sqrt((t_x-r_x)*(t_x-r_x) + (t_y-r_y)*(t_y-r_y))
    DZ = r_z-t_z

    # v_info : {"v": float, "h': float, "d1": float, "d2": float}
    tan_value_list = []
    for i, dot in enumerate(dots_in_line):
        if i==0 or i==len(dots_in_line)-1:
            continue
        x, y = to_utm(dted_data["grid_lon"][dot[0]],
                      dted_data["grid_lat"][dot[1]])
        z = dted_data["grid_height"][dot[1]][dot[0]]
        d2 = sqrt((x-r_x) * (x-r_x) + (y-r_y)*(y-r_y))
        if d2==0:
            print(x,r_x,y,r_y)
        tan_value =  (z - r_z)/d2
        tan_value_list.append(tan_value)
    
    index_of_max_tan_value = np.argmax(tan_value_list)+1 # +1 because of ignored first dot
    max_angle_dot = dots_in_line[index_of_max_tan_value]
    x, y = to_utm(dted_data["grid_lon"][max_angle_dot[0]],
                      dted_data["grid_lat"][max_angle_dot[1]])
    z = dted_data["grid_height"][max_angle_dot[1]][max_angle_dot[0]]
    d2 = sqrt((x-r_x)*(x-r_x)+(y-r_y)*(y-r_y))
    lz = DZ/D*(D-d2) + t_z
    h = z - lz
    v = get_v_factor(h,f,D-d2,d2)
    return v

def get_observer_predicted_power(dted_data, f, t_ix, t_iy, t_h, r_ix, r_iy, r_h):
    if abs(t_ix - r_ix)<= 3 and abs(t_iy - r_iy)<=3:
        return np.nan

    dots_in_line = get_dots_in_line(t_ix, t_iy, r_ix, r_iy)
    t_x, t_y = to_utm(dted_data["grid_lon"][t_ix], dted_data["grid_lat"][t_iy])
    t_z = dted_data["grid_height"][t_iy][t_ix] + t_h
    r_x, r_y = to_utm(dted_data["grid_lon"][r_ix], dted_data["grid_lat"][r_iy])
    r_z = dted_data["grid_height"][r_iy][r_ix] + r_h
    D = sqrt((t_x-r_x)*(t_x-r_x) + (t_y-r_y)*(t_y-r_y))
    DZ = r_z-t_z

    # v_info : {"v": float, "h': float, "d1": float, "d2": float}
    tan_value_list = []
    for i, dot in enumerate(dots_in_line):
        if i==0 or i==len(dots_in_line)-1:
            continue
        x, y = to_utm(dted_data["grid_lon"][dot[0]],
                      dted_data["grid_lat"][dot[1]])
        z = dted_data["grid_height"][dot[1]][dot[0]]
        d2 = sqrt((x-r_x) * (x-r_x) + (y-r_y)*(y-r_y))
        if d2==0:
            print(x,r_x,y,r_y)
        tan_value =  (z - r_z)/d2
        tan_value_list.append(tan_value)
    
    index_of_max_tan_value = np.argmax(tan_value_list)+1 # +1 because of ignored first dot
    max_angle_dot = dots_in_line[index_of_max_tan_value]
    x, y = to_utm(dted_data["grid_lon"][max_angle_dot[0]],
                      dted_data["grid_lat"][max_angle_dot[1]])
    z = dted_data["grid_height"][max_angle_dot[1]][max_angle_dot[0]]
    d2 = sqrt((x-r_x)*(x-r_x)+(y-r_y)*(y-r_y))
    lz = DZ/D*(D-d2) + t_z
    h = z - lz
    v = get_v_factor(h,f,D-d2,d2)
    return get_received_power_using_r(f,D,v)

def get_friis_power(dted_data, f, t_ix, t_iy, t_h, r_ix, r_iy, r_h):
    if abs(t_ix - r_ix)<= 3 and abs(t_iy - r_iy)<=3:
        return np.nan
    t_x, t_y = to_utm(dted_data["grid_lon"][t_ix], dted_data["grid_lat"][t_iy])
    r_x, r_y = to_utm(dted_data["grid_lon"][r_ix], dted_data["grid_lat"][r_iy])
    t_z = dted_data["grid_height"][t_iy][t_ix] + t_h
    r_z = dted_data["grid_height"][r_iy][r_ix] + r_h
    D = sqrt((t_x-r_x)*(t_x-r_x) + (t_y-r_y)*(t_y-r_y) + (t_z-r_z)*(t_z-r_z))
    return get_friis_gain(f,D)

def get_info_about_observer_predicted_power(dted_data, t_ix, t_iy, t_h, r_ix, r_iy, r_h):
    if abs(t_ix - r_ix)<= 3 and abs(t_iy - r_iy)<=3:
        return {"R":np.nan, "D":np.nan, "H":np.nan}

    dots_in_line = get_dots_in_line(t_ix, t_iy, r_ix, r_iy)
    t_x, t_y = to_utm(dted_data["grid_lon"][t_ix], dted_data["grid_lat"][t_iy])
    t_z = dted_data["grid_height"][t_iy][t_ix] + t_h
    r_x, r_y = to_utm(dted_data["grid_lon"][r_ix], dted_data["grid_lat"][r_iy])
    r_z = dted_data["grid_height"][r_iy][r_ix] + r_h
    R = sqrt((t_x-r_x)*(t_x-r_x) + (t_y-r_y)*(t_y-r_y))

    # v_info : {"v": float, "h': float, "d1": float, "d2": float}
    tan_value_list = []
    for i, dot in enumerate(dots_in_line):
        if i==0 or i==len(dots_in_line)-1:
            continue
        x, y = to_utm(dted_data["grid_lon"][dot[0]],
                      dted_data["grid_lat"][dot[1]])
        z = dted_data["grid_height"][dot[1]][dot[0]]
        d2 = sqrt((x-r_x) * (x-r_x) + (y-r_y)*(y-r_y))
        if d2==0:
            print(x,r_x,y,r_y)
        tan_value =  (z - r_z)/d2
        tan_value_list.append(tan_value)
    
    index_of_max_tan_value = np.argmax(tan_value_list)+1 # +1 because of ignored first dot
    max_angle_dot = dots_in_line[index_of_max_tan_value]
    x, y = to_utm(dted_data["grid_lon"][max_angle_dot[0]],
                      dted_data["grid_lat"][max_angle_dot[1]])
    z = dted_data["grid_height"][max_angle_dot[1]][max_angle_dot[0]]
    D = sqrt((x-r_x)*(x-r_x) + (y-r_y)*(y-r_y))
    H = z - r_z
    return {"R":R, "D":D, "H":H}

def get_received_power(dted_data, f, t_ix, t_iy, t_h, r_ix, r_iy, r_h):
    if abs(t_ix - r_ix)<= 3 and abs(t_iy - r_iy)<=3:
        return np.nan
    t_x, t_y = to_utm(dted_data["grid_lon"][t_ix], dted_data["grid_lat"][t_iy])
    r_x, r_y = to_utm(dted_data["grid_lon"][r_ix], dted_data["grid_lat"][r_iy])
    D = sqrt((t_x-r_x)*(t_x-r_x) + (t_y-r_y)*(t_y-r_y))

    dots_in_line = get_dots_in_line(t_ix, t_iy, r_ix, r_iy)
    dot_list = []
    for i,dot in enumerate(dots_in_line):
        x, y = to_utm(dted_data["grid_lon"][dot[0]],
                      dted_data["grid_lat"][dot[1]])
        z = dted_data["grid_height"][dot[1]][dot[0]]
        if i==0:
            dot_list.append({"x":x,"y":y, "z":z+t_h})
        elif i==len(dots_in_line)-1:
            dot_list.append({"x":x,"y":y, "z":z+r_h})
        else:
            dot_list.append({"x":x, "y":y, "z":z})
    loss = get_deygout_loss(f,dot_list)
    rp = get_friis_gain(f,D) - loss
    return rp
#첫 점, 끝점에 th, rh 더해줘야 함, dot은 x,y,z정보를 가짐
def get_deygout_loss(f,dot_list): 
    if len(dot_list)<3:
        return None
    TX, TY, TZ = dot_list[0]["x"], dot_list[0]["y"], dot_list[0]["z"]
    DX = dot_list[len(dot_list)-1]["x"] - dot_list[0]["x"]
    DY = dot_list[len(dot_list)-1]["y"] - dot_list[0]["y"]
    DZ = dot_list[len(dot_list)-1]["z"] - dot_list[0]["z"]
    D = sqrt(DX*DX+DY*DY)

    v_list = []
    for i, dot in enumerate(dot_list):
        if i==0 or i==len(dot_list)-1:
            continue
        d = sqrt((dot["x"]-TX) * (dot["x"]-TX) + (dot["y"]-TY)*(dot["y"]-TY))
        lz = DZ/D*d + TZ
        h = dot["z"] - lz
        v = get_v_factor(h,f,d,D-d)
        v_list.append(v)

    max_v=max(v_list)
    
    if max_v>0: #NLOS
        index_of_max_v=np.argmax(v_list)+1 # +1 because of ignored first dot
        before_loss = get_deygout_loss(f,dot_list[:index_of_max_v+1])
        after_loss = get_deygout_loss(f,dot_list[index_of_max_v:])
        if before_loss and after_loss: #deygout
            return before_loss+after_loss
        else: #normal knife edge
            return get_loss_by_knife_edge(max_v)
    else: #LOS
        return get_loss_by_knife_edge(max_v)


