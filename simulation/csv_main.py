#import and implement main code

import pandas as pd
from environments.transmitter import all_transmitters
from plot.plot_received_power import plot_differenc_of_error_with_friis
from utility.utility import convert_to_si
frequency_list = [100000000]

data = pd.DataFrame()
for frequency in frequency_list:
    for transmitter in all_transmitters:
        error = plot_differenc_of_error_with_friis(frequency, transmitter)
        name=f"{transmitter.name} {convert_to_si(frequency)}Hz"
        temp = pd.DataFrame({"PRED_ERROR":[error["PRED_ERROR"]], "FRIIS_ERROR":[error["FRIIS_ERROR"]]}, index=[name])
        data = data.append(temp)

data.to_csv("friis_and_pred_error.csv")

