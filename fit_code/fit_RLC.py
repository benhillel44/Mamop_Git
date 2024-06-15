import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import constants



def transfer_function_22kOhm(f, R, L, C):
    Rm = 22000 # Ohms
    f = f * 1e3 # Convert kHz to Hz
    # H_im = Rm / ((Rm + R) + 1j * 2 * np.pi * f * L - 1j / (2 * np.pi * f * C))
    # H = np.abs(H_im)
    H = Rm / np.sqrt( (Rm + R)**2 + (2*np.pi*f*L - 1/(2*np.pi*f*C))**2 ) #
    Amplitude = H# * 1e3 # Convert V to mV
    return Amplitude
def main(filename):
    csv_dir = constants.PATH_TO_MEASUREMENT_CSV
    path_to_csv = csv_dir + filename
    x_name = 'freq_A'
    y_name = 'amp_A'

    df_data = pd.read_csv(path_to_csv)
    df_data["ratio"] = df_data['amp_B'] / df_data['amp_A']
    plt.scatter(x=df_data['freq_A'], y=df_data['amp_A'], label='A', s=10)
    plt.scatter(x=df_data['freq_B'], y=df_data['amp_B'], label='B', s=10)
    plt.legend()
    plt.xlabel("Frequency [KHz]")
    plt.ylabel("Voltage [V]")
    plt.title(f"Voltage vs Frequency {filename}")
    plt.savefig(f'{path_to_csv[:-4]}.png')
    plt.figure()


    plt.scatter(x=df_data['freq_A'], y=df_data['amp_ratio'], label='A/B')
    plt.xlabel("Frequency [KHz]")
    plt.ylabel("Voltage [V]")
    plt.title(f"Voltage vs Frequency {filename} Ratio")
    plt.savefig(f'{path_to_csv[:-4]}_ratio.png')
    plt.show()



    x = df_data['omega_A']*1000
    y = df_data['amp_ratio']

# # Fit the custom function to the data
# params, _ = curve_fit(transfer_function_22kOhm, x, y, bounds=([0, 0, 0], [100, 0.000001, 0.00001]))
# # Extract the fitted parameters
# R_fit, L_fit, C_fit = params
# print("Fitted Parameters: R_fit, L_fit, C_fit", R_fit, L_fit, C_fit)
# plt.scatter(x, y)
# # x = np.linspace(0, 1000, 10000)
# R_fit = 2
# L_fit = 0.001
# C_fit = 0.000000000015
# # x = np.linspace(1, 2500, 10000)
# # print(len(x))
# # plt.plot(x, transfer_function_22kOhm(x, R_fit, L_fit, C_fit), color='red')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Amplitude Ratio')
# plt.title('Amplitude Ratio vs Frequency')
# plt.grid(True)
# plt.show()

def run_fit_RLC(measurement_):
    filename_ = f"measurement_{measurement_}.csv"
    main(filename_)

if __name__ == '__main__':
    measurement = 3
    filename = f"measurement_{measurement}.csv"
    main(filename)

