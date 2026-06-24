import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from map_factory.map_factory import *
from utility.utility import convert_to_si

r_h = 10


def plot_received_power_near_transmitter(frequency, transmitter):
    name, t_lon, t_lat, span_lon, span_lat, t_h = transmitter()
    DATA = get_received_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    fig = plt.pcolormesh(
        DATA["X"], DATA["Y"], DATA["RP"], shading="auto")
    fig.axes.set_aspect("equal")
    plt.plot(t_lon, t_lat, "ro", markersize=2)
    plt.xlabel("LAT, (degree)")
    plt.ylabel("LON, (degree)")
    plt.title(f"Received Power near {name} ({convert_to_si(frequency)}Hz)")
    cbar = plt.colorbar(fig)
    cbar.set_label('Received Power (dBm)')
    plt.savefig(f"simulation/IMGS/RP/RP_{name}_{convert_to_si(frequency)}Hz.png")
    print(f"{name}_{convert_to_si(frequency)}Hz.png CREATED")
    plt.clf()


def plot_observer_prediected_power_near_transmitter(frequency, transmitter):
    name, t_lon, t_lat, span_lon, span_lat, t_h = transmitter()
    DATA = get_observer_predicted_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    fig = plt.pcolormesh(
        DATA["X"], DATA["Y"], DATA["RP"], shading="auto")
    fig.axes.set_aspect("equal")
    plt.plot(t_lon, t_lat, "ro", markersize=2)
    plt.xlabel("LAT, (degree)")
    plt.ylabel("LON, (degree)")
    plt.title(
        f"Predicted Power By Observer near {name} ({convert_to_si(frequency)}Hz)")
    cbar = plt.colorbar(fig)
    cbar.set_label('Received Power (dBm)')
    plt.savefig(f"simulation/IMGS/RP/PPB_{name}_{convert_to_si(frequency)}Hz.png")
    print(f"CAL_{name}_{convert_to_si(frequency)}Hz.png CREATED")
    plt.clf()


def plot_predicted_power_near_transmitter(frequency, transmitter):
    name, t_lon, t_lat, span_lon, span_lat, t_h = transmitter()
    DATA = get_predicted_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    fig = plt.pcolormesh(
        DATA["X"], DATA["Y"], DATA["RP"], shading="auto")
    fig.axes.set_aspect("equal")
    plt.plot(t_lon, t_lat, "ro", markersize=2)
    plt.xlabel("LAT, (degree)")
    plt.ylabel("LON, (degree)")
    plt.title(
        f"Predicted Received Power near {name} ({convert_to_si(frequency)}Hz)")
    cbar = plt.colorbar(fig)
    cbar.set_label('Received Power (dBm)')
    plt.savefig(f"simulation/IMGS/RP/PRED_{name}_{convert_to_si(frequency)}Hz.png")
    print(f"PRED_{name}_{convert_to_si(frequency)}Hz.png CREATED")
    plt.clf()


def plot_differences_between_pred_and_calc(frequency, transmitter):
    name, t_lon, t_lat, span_lon, span_lat, t_h = transmitter()
    REAL = get_received_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    CALC = get_observer_predicted_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    CALC_DIFFERENCE = pd.DataFrame(REAL["RP"]) - pd.DataFrame(CALC["RP"])

    PRED = get_predicted_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    PRED_DIFFERENCE = pd.DataFrame(REAL["RP"]) - pd.DataFrame(PRED["RP"])

    v_min = min([np.nanmin(PRED_DIFFERENCE), np.nanmin(CALC_DIFFERENCE)])
    v_max = max([np.nanmax(PRED_DIFFERENCE), np.nanmax(CALC_DIFFERENCE)])

    fig = plt.pcolormesh(
        REAL["X"], REAL["Y"], CALC_DIFFERENCE, shading="auto")
    ERRORS = CALC_DIFFERENCE.abs()
    ERROR = ERRORS.stack().mean()
    fig.axes.set_aspect("equal")
    plt.plot(t_lon, t_lat, "ro", markersize=2)
    plt.xlabel("LAT, (degree)")
    plt.ylabel("LON, (degree)")
    plt.title(f"CALC Difference")
    cbar = plt.colorbar(fig)
    plt.clim(v_min, v_max)
    cbar.set_label('Received Power (dBm)')
    plt.savefig(
        f"simulation/IMGS/RP_DIFF/CALC_{name}_{convert_to_si(frequency)}Hz_ERROR:{ERROR:.3f}.png")
    print(f"DIFF_CALC_{name}_{convert_to_si(frequency)}Hz.png CREATED")
    print(f"ERROR: {ERROR:.3f}")
    plt.clf()

    fig = plt.pcolormesh(
        REAL["X"], REAL["Y"], PRED_DIFFERENCE, shading="auto")
    ERRORS = PRED_DIFFERENCE.abs()
    ERROR = ERRORS.stack().mean()
    fig.axes.set_aspect("equal")
    plt.plot(t_lon, t_lat, "ro", markersize=2)
    plt.xlabel("LAT, (degree)")
    plt.ylabel("LON, (degree)")
    plt.title(
        f"PRED Difference")
    cbar = plt.colorbar(fig)
    plt.clim(v_min, v_max)
    cbar.set_label('Received Power (dBm)')
    plt.savefig(
        f"simulation/IMGS/RP_DIFF/PRED_{name}_{convert_to_si(frequency)}Hz_ERROR:{ERROR:.3f}.png")
    print(f"DIFF_PRED_{name}_{convert_to_si(frequency)}Hz.png CREATED")
    print(f"ERROR: {ERROR:.3f}")
    plt.clf()

def plot_abs_differences_between_pred_and_calc(frequency, transmitter):
    name, t_lon, t_lat, span_lon, span_lat, t_h = transmitter()
    REAL = get_received_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    CALC = get_observer_predicted_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    CALC_DIFFERENCE = pd.DataFrame(REAL["RP"]) - pd.DataFrame(CALC["RP"])
    CALC_ERRORS = CALC_DIFFERENCE.abs()

    PRED = get_predicted_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    PRED_DIFFERENCE = pd.DataFrame(REAL["RP"]) - pd.DataFrame(PRED["RP"])
    PRED_ERRORS = PRED_DIFFERENCE.abs()

    v_max = max([np.nanmax(PRED_ERRORS), np.nanmax(CALC_ERRORS)])

    fig = plt.pcolormesh(
        REAL["X"], REAL["Y"], CALC_ERRORS, shading="auto")
    ERROR = CALC_ERRORS.stack().mean()
    fig.axes.set_aspect("equal")
    plt.plot(t_lon, t_lat, "ro", markersize=2)
    plt.xlabel("LAT, (degree)")
    plt.ylabel("LON, (degree)")
    plt.title(f"CALC Difference")
    cbar = plt.colorbar(fig)
    plt.clim(0, v_max)
    cbar.set_label('Received Power (dBm)')
    plt.savefig(
        f"simulation/IMGS/RP_ABS_DIFF/CALC_{name}_{convert_to_si(frequency)}Hz_ERROR:{ERROR:.3f}.png")
    print(f"DIFF_CALC_{name}_{convert_to_si(frequency)}Hz.png CREATED")
    print(f"ERROR: {ERROR:.3f}")
    plt.clf()

    fig = plt.pcolormesh(
        REAL["X"], REAL["Y"], PRED_ERRORS, shading="auto")
    ERROR = PRED_ERRORS.stack().mean()
    fig.axes.set_aspect("equal")
    plt.plot(t_lon, t_lat, "ro", markersize=2)
    plt.xlabel("LAT, (degree)")
    plt.ylabel("LON, (degree)")
    plt.title(
        f"PRED Difference")
    cbar = plt.colorbar(fig)
    plt.clim(0, v_max)
    cbar.set_label('Received Power (dBm)')
    plt.savefig(
        f"simulation/IMGS/RP_ABS_DIFF/PRED_{name}_{convert_to_si(frequency)}Hz_ERROR:{ERROR:.3f}.png")
    print(f"DIFF_PRED_{name}_{convert_to_si(frequency)}Hz.png CREATED")
    print(f"ERROR: {ERROR:.3f}")
    plt.clf()


def plot_received_and_observer_predicted(frequency, transmitter):
    name, t_lon, t_lat, span_lon, span_lat, t_h = transmitter()
    RP = get_received_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    OP = get_observer_predicted_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    RP_DATA = pd.DataFrame(RP["RP"])
    OP_DATA = pd.DataFrame(OP["RP"])
    v_min = min([np.nanmin(RP_DATA), np.nanmin(OP_DATA)])
    v_max = max([np.nanmax(RP_DATA), np.nanmax(OP_DATA)])

    fig = plt.pcolormesh(
        RP["X"], RP["Y"], RP["RP"], shading="auto")
    fig.axes.set_aspect("equal")
    plt.plot(t_lon, t_lat, "ro", markersize=2)
    plt.xlabel("LAT, (degree)")
    plt.ylabel("LON, (degree)")
    plt.title(f"Received Power near {name} ({convert_to_si(frequency)}Hz)")
    cbar = plt.colorbar(fig)
    plt.clim(v_min, v_max)
    cbar.set_label('Received Power (dBm)')
    plt.savefig(f"simulation/IMGS/RP/RP_{name}_{convert_to_si(frequency)}Hz.png")
    print(f"{name}_{convert_to_si(frequency)}Hz.png CREATED")
    plt.clf()

    fig = plt.pcolormesh(
        OP["X"], OP["Y"], OP["RP"], shading="auto")
    fig.axes.set_aspect("equal")
    plt.plot(t_lon, t_lat, "ro", markersize=2)
    plt.xlabel("LAT, (degree)")
    plt.ylabel("LON, (degree)")
    plt.title(
        f"Predicted Power By Observer near {name} ({convert_to_si(frequency)}Hz)")
    cbar = plt.colorbar(fig)
    plt.clim(v_min, v_max)
    cbar.set_label('Received Power (dBm)')
    plt.savefig(f"simulation/IMGS/RP/PPB_{name}_{convert_to_si(frequency)}Hz.png")
    print(f"CAL_{name}_{convert_to_si(frequency)}Hz.png CREATED")
    plt.clf()

def plot_diffence_of_error(frequency, transmitter):
    name, t_lon, t_lat, span_lon, span_lat, t_h = transmitter()
    REAL = get_received_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    CALC = get_observer_predicted_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    CALC_DIFFERENCE = pd.DataFrame(REAL["RP"]) - pd.DataFrame(CALC["RP"])
    CALC_ERRORS = CALC_DIFFERENCE.abs()

    PRED = get_predicted_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    PRED_DIFFERENCE = pd.DataFrame(REAL["RP"]) - pd.DataFrame(PRED["RP"])
    PRED_ERRORS = PRED_DIFFERENCE.abs()

    ERROR_DIFFRENCE = PRED_ERRORS - CALC_ERRORS

    fig = plt.pcolormesh(
        REAL["X"], REAL["Y"], ERROR_DIFFRENCE, shading="auto", cmap='RdBu')
    fig.axes.set_aspect("equal")
    plt.plot(t_lon, t_lat, "ro", markersize=2)
    plt.xlabel("LAT, (degree)")
    plt.ylabel("LON, (degree)")
    plt.title(f"PRED Error - CALC Error ({name} {convert_to_si(frequency)}Hz)")
    cbar = plt.colorbar(fig)
    plt.clim(-20, 20)
    cbar.set_label('Received Power (dBm)')
    plt.savefig(
        f"simulation/IMGS/RP_ERROR_DIFF/{name}_{convert_to_si(frequency)}Hz.png")
    print(f"RP_ERROR_DIFF_{name}_{convert_to_si(frequency)}Hz.png CREATED")
    plt.clf()

    return {"PRED_ERROR": PRED_ERRORS.stack().mean(), "CALC_ERROR": CALC_ERRORS.stack().mean()}


def plot_differenc_of_error_with_friis(frequency,transmitter):
    name, t_lon, t_lat, span_lon, span_lat, t_h = transmitter()
    REAL = get_received_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    FRIIS = get_friis_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    FRIIS_DIFFERENCE = pd.DataFrame(REAL["RP"]) - pd.DataFrame(FRIIS["RP"])
    FRIIS_ERRORS = FRIIS_DIFFERENCE.abs()

    PRED = get_predicted_power_map(
        frequency, t_h, r_h, t_lon, t_lat, span_lon, span_lat)
    PRED_DIFFERENCE = pd.DataFrame(REAL["RP"]) - pd.DataFrame(PRED["RP"])
    PRED_ERRORS = PRED_DIFFERENCE.abs()
 
    ERROR_DIFFRENCE = PRED_ERRORS - FRIIS_ERRORS

    fig = plt.pcolormesh(
        REAL["X"], REAL["Y"], ERROR_DIFFRENCE, shading="auto", cmap='RdBu')
    fig.axes.set_aspect("equal")
    plt.plot(t_lon, t_lat, "ro", markersize=2)
    plt.xlabel("LAT, (degree)")
    plt.ylabel("LON, (degree)")
    plt.title(f"PRED Error - FRIIS Error ({name} {convert_to_si(frequency)}Hz)")
    cbar = plt.colorbar(fig)
    plt.clim(-20, 20)
    cbar.set_label('Received Power (dBm)')
    plt.savefig(
        f"simulation/IMGS/RP_DIFF_FRIIS_BETW_PRED/{name}_{convert_to_si(frequency)}Hz.png")
    print(f"RP_DIFF_FRIIS_BETW_PRED_{name}_{convert_to_si(frequency)}Hz.png CREATED")
    plt.clf()

    return {"PRED_ERROR": PRED_ERRORS.stack().mean(), "FRIIS_ERROR": FRIIS_ERRORS.stack().mean()}
    