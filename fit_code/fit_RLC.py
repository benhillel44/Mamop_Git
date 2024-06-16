import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import fit_code.constants as constants

# CONSTANTS
R_1 = 22000 # ohm

def transfer_function_22kOhm(f, R, L, C):
    Rm = 22000 # Ohms
    f = f * 1e3 # Convert kHz to Hz
    # H_im = Rm / ((Rm + R) + 1j * 2 * np.pi * f * L - 1j / (2 * np.pi * f * C))
    # H = np.abs(H_im)
    H = Rm / np.sqrt( (Rm + R)**2 + (2*np.pi*f*L - 1/(2*np.pi*f*C))**2 ) #
    Amplitude = H# * 1e3 # Convert V to mV
    return Amplitude

def plot_shit(df_data, path_to_csv):
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

    # TODO - DELELTE THIS
    y_moved = ((df_data['phase_diff'] + np.pi) % (2 * np.pi)) - np.pi

    plt.scatter(x=df_data['freq_A'], y=y_moved, label='A/B')
    plt.xlabel("Frequency [KHz]")
    plt.ylabel("Phase diff")
    plt.title(f"phase diff  vs Frequency {filename} Ratio")
    plt.savefig(f'{path_to_csv[:-4]}_ratio.png')
    plt.show()

def run_fit_RLC(measurement_):
    filename_ = f"measurement_{measurement_}.csv"
    main(filename_)

def RCL_model(x, R_L, L, C_L):
    """
    model (impedance) of -
                        ==== coil (L) ============
                        ==                      ==
    ==== resistor (R_L) ==                      =======
                        ==                      ==
                        ==== capacitor (C_L) =====
    """
    return np.sqrt(
        (R_L**2 + ((L*x) / (1 - L*C_L*(x**2)))**2)
        / ((R_1 +R_L)**2 + ((L*x) / (1 - L*C_L*(x**2)))**2)
    )

def RLC_model_with_knowen_LC(x, R_L, L):
    """
    model (impedance) of -
                        ==== coil (L) ============
                        ==                      ==
    ==== resistor (R_L) ==                      =======
                        ==                      ==
                        ==== capacitor (C_L) =====
    """
    # TODO - dont forget to update this
    LC = 5.047458080518233e-12

    return np.sqrt(
        (R_L**2 + ((L*x) / (1 - LC*(x**2)))**2)
        / ((R_1 +R_L)**2 + ((L*x) / (1 - LC*(x**2)))**2)
    )
def find_LC(x, y):
    # calculate the impedance of the coil
    Z_A = (R_1 * y) / (1 - y)

    # detect maximum in the impedance => x_max = 1+sqrt(1 + 1/(LC))
    x_max = x[np.argmax(Z_A)]

    # derive the value of LC
    LC = 1 / ((x_max - 1)**2 - 1)
    print(f"LC = {LC}")
    return LC

def main(filename):
    csv_dir = constants.PATH_TO_MEASUREMENT_CSV
    path_to_csv = csv_dir + filename
    # (1) plot the basic curves
    df_data = pd.read_csv(path_to_csv)
    plot_shit(df_data, path_to_csv)

    # (2) fit a curve to the set of points
    x_data = df_data['freq_A'] * 1000 # Hertz
    y_data = df_data['amp_ratio']

    # (2.1) find LC
    # TODO - each time you change expirement, remember to manually change the LC value in the model
    LC = find_LC(x_data, y_data)

    # (2,2) define bounds and fit the parameters
    lower_bounds = [0,0] # bounds for R_L, L
    upper_bounds = [10, 5e-3] # bounds for R_L, L
    params, covariance = curve_fit(RLC_model_with_knowen_LC, x_data, y_data,
                                   bounds=(lower_bounds, upper_bounds))

    R_L, L = params
    C_L = LC / L
    print(f"R_L = {R_L}, L = {L}, C_L = {C_L}")
    print(f">> Covariances \nCov(R_L,R_L)={covariance[0,0]}\n"
          f"Cov(R_L,L)={covariance[0,1]}\nCov(L,L)={covariance[1,1]}")

    # Plotting the data and the fitted curve
    plt.scatter(x_data, y_data, color='red', label='Data Points')
    x_model = np.linspace(min(x_data), max(x_data), 100)
    y_model = RCL_model(x_model, R_L, L, C_L)
    plt.plot(x_model, y_model, label='Fitted Curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Curve Fitting')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    measurement = 9
    filename = f"measurement_{measurement}.csv"
    main(filename)

