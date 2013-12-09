""" RFI Code Challenge:
Sample code to demonstrate a possible mitigation technique for radio
interference. Produced for SKA position application.

Task:
    The task is to write code to demonstrate a possible mitigation technique
    for radio interference.

    In particular, we would like you to generate radio telescope appropriate
    data (basically noise like), insert an artificial source of interference,
    and the demonstrate recovery of the insertion by indicating a set of flags
    on the original dataset.

    This is not intended to be a detailed simulation of radio astronomy data,
    a simple model is perfectly adequate. You may assume a single telescope
    producing channelised noise like data.

Author: David Sharpe
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt


def plot_helper(x_axis, data, title, x_label, y_lable):
    """ Helper function for plotting subplots
    """
    plot_helper.i += 1
    plt.subplot(3,2,plot_helper.i)
    if x_axis is not None:
        plt.plot(x_axis, data)
    else:
        plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_lable)


def remove_known_interference_from_signal():
    # GIVEN: a sample signal of two sinusoidal waves added together
    f_s = 500.0  # Hz
    f = 1.0  # Hz
    time = np.arange(0.0, 3.0, 1 / f_s)
    data = 5 * np.sin(2 * np.pi * f * time) + \
        3 * np.sin(8 * 2 * np.pi * f * time)

    # IF: we add some "noise" to the signal, where the noise is in a known
    # frequency
    f_noise = 100  # Hz
    data += 1 * np.sin(2 * np.pi * f_noise * time)

    plot_helper.i = 0
    plt.figure(num=None, figsize=(16, 12), dpi=80, edgecolor='k')
    plot_helper(time, data, 'Known signal + 100Hz sinusoidal '
                            'noise', 'Time (sec)', 'Amplitude')

    # IF: we take the fast fourier transform of the signal
    fft_x = np.fft.fft(data)
    n = len(fft_x)
    freq = np.fft.fftfreq(n, 1 / f_s)

    plot_helper(None, np.abs(fft_x), 'FFT of signal', 'Frequency (Hz)',
                'Amplitude')

    fft_x_shifted = np.fft.fftshift(fft_x)
    freq_shifted = np.fft.fftshift(freq)

    plot_helper(freq_shifted, np.abs(fft_x_shifted),
                'FFT of signal, zero-frequency component to the center of the '
                'spectrum', 'Frequency (Hz)', 'Amplitude')

    # filter the Fourier transform
    def filter_rule(x, freq):
        band = 0.05
        if abs(freq) > f_noise + band or abs(freq) < f_noise - band:
            return x
        else:
            return 0

    # THEN: we can remove the offending noisy frequency
    fft_x_filtered = np.array([filter_rule(x, freq_s) for
                               x, freq_s in zip(fft_x_shifted, freq_shifted)])

    plot_helper(freq_shifted, np.abs(fft_x_filtered),
                'FFT of signal, with 100Hz component removed',
                'Frequency (Hz)', 'Amplitude')

    # THEN: when we perform the inverse FFT, we can see the original signal
    fft_x_filtered_inverted = np.fft.ifft(np.fft.ifftshift(fft_x_filtered))

    plot_helper(time, fft_x_filtered_inverted,
                'Original signal with 100Hz component removed',
                'Time (sec)', 'Amplitude')

   # AND: we can show the difference between the two signals is the noise we
   # have removed
    plot_helper(time[:100], fft_x_filtered_inverted[:100] - data[:100],
                'Difference between original signal and nose component',
                'Time (sec)', 'Amplitude')

    plt.show()


def remove_known_interference_from_noisy_signal():
    # GIVEN: a sample signal
    f_s = 500.0  # Hz
    f = 1.0  # Hz
    time = np.arange(0.0, 3.0, 1 / f_s)

    data = 2 * np.sin(2 * np.pi * f * time) + \
        10 * scipy.random.random(len(time))

    # IF: we add some "noise" to the signal, where the noise is in a known
    # frequency
    f_noise = 100  # Hz
    data += 1 * np.sin(2 * np.pi * f_noise * time)

    plot_helper.i = 0
    plt.figure(num=None, figsize=(16, 12), dpi=80, edgecolor='k')
    plot_helper(time, data, 'Known signal + 100Hz sinusoidal '
                            'noise', 'Time (sec)', 'Amplitude')

    # IF: we take the fast fourier transform of the signal
    fft_x = np.fft.fft(data)
    n = len(fft_x)
    freq = np.fft.fftfreq(n, 1 / f_s)

    plot_helper(None, np.abs(fft_x), 'FFT of signal', 'Frequency (Hz)',
                'Amplitude')

    fft_x_shifted = np.fft.fftshift(fft_x)
    freq_shifted = np.fft.fftshift(freq)

    plot_helper(freq_shifted, np.abs(fft_x_shifted),
                'FFT of signal, zero-frequency component to the center of the '
                'spectrum', 'Frequency (Hz)', 'Amplitude')

    # filter the Fourier transform
    def filter_rule(x, freq):
        band = 0.05
        if abs(freq) > f_noise + band or abs(freq) < f_noise - band:
            return x
        else:
            return 0

    # THEN: we can remove the offending noisy frequency
    fft_x_filtered = np.array([filter_rule(x, freq_s) for
                               x, freq_s in zip(fft_x_shifted, freq_shifted)])

    plot_helper(freq_shifted, np.abs(fft_x_filtered),
                'FFT of signal, with 100Hz component removed',
                'Frequency (Hz)', 'Amplitude')

    # THEN: when we perform the inverse FFT, we can see the original signal
    fft_x_filtered_inverted = np.fft.ifft(np.fft.ifftshift(fft_x_filtered))

    plot_helper(time, fft_x_filtered_inverted,
                'Original signal with 100Hz component removed',
                'Time (sec)', 'Amplitude')

    # AND: we can show the difference between the two signals is the noise we
    # have removed
    plot_helper(time[:100], fft_x_filtered_inverted[:100] - data[:100],
                'Difference between original signal and nose component',
                'Time (sec)', 'Amplitude')

    plt.show()


def extract_sinusoid_from_noise():
    # GIVEN: a sample signal
    f_signal = 2.0  # Hz
    time = scipy.linspace(0, 120, 10000)

    data = 1 * scipy.sin(2 * scipy.pi * f_signal * time)

    # IF: we add some random noise to the signal
    data += 2 * scipy.random.random(len(time))

    plot_helper.i = 0
    plt.figure(num=None, figsize=(16, 12), dpi=80, edgecolor='k')
    plot_helper(time, data, 'Known signal + 100Hz sinusoidal '
                            'noise', 'Time (sec)', 'Amplitude')

    # IF: we take the fast fourier transform of the signal
    fft_x = np.fft.fft(data)
    n = len(fft_x)
    freq = np.fft.fftfreq(n, time[1] - time[0])

    plot_helper(None, np.abs(fft_x), 'FFT of signal', 'Frequency (Hz)',
                'Amplitude')

    fft_x_shifted = np.fft.fftshift(fft_x)
    freq_shifted = np.fft.fftshift(freq)

    plot_helper(freq_shifted, np.abs(fft_x_shifted),
                'FFT of signal, zero-frequency component to the center of the '
                'spectrum', 'Frequency (Hz)', 'Amplitude')

    # filter the Fourier transform
    def filter_rule(x, freq):
        band = 0.05
        if abs(freq) > f_signal + band or abs(freq) < f_signal - band:
            return 0
        else:
            return x

    # THEN: we can remove the offending noisy frequency
    fft_x_filtered = np.array([filter_rule(x, freq_s) for
                               x, freq_s in zip(fft_x_shifted, freq_shifted)])

    plot_helper(freq_shifted, np.abs(fft_x_filtered),
                'FFT of signal, with 100Hz component removed',
                'Frequency (Hz)', 'Amplitude')

    # THEN: when we perform the inverse FFT, we can see the original signal
    fft_x_filtered_inverted = np.fft.ifft(np.fft.ifftshift(fft_x_filtered))

    plot_helper(time, fft_x_filtered_inverted,
                'Sample signal with noise removed',
                'Time (sec)', 'Amplitude')

    plt.show()



def main():
    print('Demonstration of the removal of a 100Hz signal from a known '
          'sinusoidal signal')
    remove_known_interference_from_signal()
    print('Demonstration of the removal of a 100Hz signal from a noisy '
          'sinusoidal signal')
    remove_known_interference_from_noisy_signal()
    print('Demonstration of the extraction of a 2Hz signal from noise, with a '
          'signal to noise ratio of 1:2')
    extract_sinusoid_from_noise()


if __name__ == "__main__":
    main()