from random import uniform
from environments.transmitter import Transmitter


rand_lon = uniform(127, 128.5)
rand_lat = uniform(35, 38)
random_transmitter = Transmitter("RAND", rand_lon,rand_lat)
frequency = 100000000



from plot.plot_map import plot_surround_of_transmitter
plot_surround_of_transmitter(random_transmitter)

if "n" == input("Continue?(y/n) "):
    exit()

from map_factory.outlier_check import check_outlier_error
check_outlier_error(frequency, random_transmitter)

if "n" == input("Continue?(y/n) "):
    exit()

from plot.plot_received_power import plot_received_power_near_transmitter, plot_diffence_of_error
plot_received_power_near_transmitter(frequency, random_transmitter)

if "n" == input("Continue?(y/n) "):
    exit()

plot_diffence_of_error(frequency, random_transmitter)

if "n" == input("Continue?(y/n) "):
    exit()

from csv_factory import create_CSV
create_CSV([random_transmitter], [frequency])