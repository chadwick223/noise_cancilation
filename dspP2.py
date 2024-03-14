import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os



def ANC(desired_audio,corrupted_audio):
    # Parameters for the first adaptive filter
    order_1 = 20
    mu_1 = 0.005

    # Processing for the first noise cancellation stage
    w_1 = np.zeros(order_1)
    y_out_1 = np.zeros(len(x))

    for i in range(len(x) - order_1):
        buffer_1 = x[i:i + order_1]
        y_out_1[i] = np.dot(buffer_1, w_1)
        error_1 = y[i] - y_out_1[i]

        # Handle NaN values in the error array
        error_1 = np.nan_to_num(error_1)

        # Check for zero division in buffer_scale and error_scale
        max_abs_buffer_1 = np.max(np.abs(buffer_1))
        max_abs_error_1 = np.max(np.abs(error_1))

        buffer_scale_1 = buffer_1 / max_abs_buffer_1 if max_abs_buffer_1 != 0 else buffer_1
        error_scale_1 = error_1 / max_abs_error_1 if max_abs_error_1 != 0 else error_1

        w_1 = w_1 + buffer_scale_1 * mu_1 * error_scale_1

    # Normalize
    y_out_norm_1 = y_out_1 / np.max(np.abs(y_out_1))

    # Convert to integer format
    y_out_int_1 = np.int16(y_out_norm_1 * 32767)

    # Parameters for the second adaptive filter (NLMS)
    order_2 = 20
    mu_2 = 0.001

    # Processing for the second noise cancellation stage (NLMS)
    w_2 = np.zeros(order_2)
    y_out_2 = np.zeros(len(y_out_1))

    for i in range(len(y_out_1) - order_2):
        buffer_2 = y_out_int_1[i:i + order_2]
        y_out_2[i] = np.dot(buffer_2, w_2)
        error_2 = y_out_1[i] - y_out_2[i]

        # Handle NaN values in the error array
        error_2 = np.nan_to_num(error_2)

        # Check for zero division in buffer_scale and error_scale
        max_abs_buffer_2 = np.max(np.abs(buffer_2))
        max_abs_error_2 = np.max(np.abs(error_2))

        buffer_scale_2 = buffer_2 / max_abs_buffer_2 if max_abs_buffer_2 != 0 else buffer_2
        error_scale_2 = error_2 / max_abs_error_2 if max_abs_error_2 != 0 else error_2

        w_2 = w_2 + buffer_scale_2 * mu_2 * error_scale_2

    # Normalize
    y_out_norm_2 = y_out_2 / np.max(np.abs(y_out_2))

    # Convert to integer format
    y_out_int_2 = np.int16(y_out_norm_2 * 32767)

    # Save noise-cancelled output after the second stage
    output_file_path_2 = input("Enter the path to save the noise-cancelled output after the second stage: ")
    
    # Save noise-cancelled output after the second stage
    wavfile.write(output_file_path_2, 44100, y_out_int_2)

    # Calculate performance metrics after the second stage
    mse_2 = mean_squared_error(y_out_1, y_out_2[:len(y_out_1)])
    snr_2 = 10 * np.log10(np.var(y_out_1) / mse_2)

    print(f"Mean Squared Error after the second stage (MSE): {mse_2}")
    print(f"Signal-to-Noise Ratio after the second stage (SNR): {snr_2} dB")

    # Plotting
    time_1 = np.arange(len(x)) / 44100.0  # Assuming a sample rate of 44100 Hz
    time_2 = np.arange(len(y_out_1)) / 44100.0  # Assuming a sample rate of 44100 Hz

    plt.figure(figsize=(14, 12))

    # Find the minimum length of the signals for plotting
    min_length = min(len(y), len(y_out_int_1), len(y_out_int_2), len(x))

    # Plotting for the first noise cancellation stage
    # ...

    # Plotting for the first noise cancellation stage
    plt.subplot(4, 2, 1)
    plt.plot(time_1[:min_length], y[:min_length], label='Corrupted Signal', color='red')
    plt.title('Corrupted Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(4, 2, 2)
    plt.plot(time_1[:min_length], x[:min_length], label='Desired Signal', color='green')
    plt.title('Desired Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(4, 2, 4)
    plt.plot(time_1[:min_length], y_out_int_1[:min_length], label='Noise-Cancelled Signal (1st stage)', color='blue')
    plt.title('Noise-Cancelled Signal (1st stage)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plotting for the second noise cancellation stage
    plt.subplot(4, 2, 6)
    plt.plot(time_1[:min_length], y_out_int_1[:min_length] - y_out_int_2[:min_length], label='Difference (1st stage - 2nd stage)', color='purple')
    plt.title('Difference (1st stage - 2nd stage)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(4, 2, 5)
    plt.plot(time_2[:min_length], y_out_int_2[:min_length], label='Noise-Cancelled Signal (2nd stage)', color='orange')
    plt.title('Noise-Cancelled Signal (2nd stage)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plotting for the noise signal (corrupted - desired)
    plt.subplot(4, 2, 3)
    noise_signal = y[:min_length] - x[:min_length]
    plt.plot(time_1[:min_length], noise_signal, label='Noise Signal (Corrupted - Desired)', color='brown')
    plt.title('Noise Signal (Corrupted - Desired)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plotting for performance metrics after the second stage
    # ...

    plt.tight_layout()
    plt.show()
    
def stereo_conv(x_stereo):
    x = x_stereo[:, 0]
    return x

#Test case 1
# File paths
x_file = "P2A1D.wav"  # Replace with the actual path
y_file = "P2A1C.wav"  # Replace with the actual path

# Read audio files
_, x_stereo = wavfile.read(x_file)
_, y_stereo = wavfile.read(y_file)

# Assuming x_stereo and y_stereo are stereo signals, take only one channel (e.g., left channel)
x = stereo_conv(x_stereo)
y = stereo_conv(y_stereo)

ANC(x, y)

#Test case 2
# File paths
x_file = "P2A2D.wav"  # Replace with the actual path
y_file = "P2A2C.wav"  # Replace with the actual path

# Read audio files
_, x_stereo = wavfile.read(x_file)
_, y_stereo = wavfile.read(y_file)

# Assuming x_stereo and y_stereo are stereo signals, take only one channel (e.g., left channel)
x = stereo_conv(x_stereo)
y = stereo_conv(y_stereo)

ANC(x, y)

x_file = "P1A1D.wav"  # Replace with the actual path
y_file = "P1A1C.wav"  # Replace with the actual path

# Read audio files
_, x = wavfile.read(x_file)
_, y = wavfile.read(y_file)

ANC(x, y)
 
##there is issue with mean square error which not been calculated correctly .
