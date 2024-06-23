""" 
This module provides functions for data augmentation.

Author: Pan.LC <coreylin2023@outlook.com>
Date: 2024/6/21
License: MIT License
"""

# 数据增强模块

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# 时间滑动窗口数据扩充类      
class TimeWindowDataAugmentation(BaseEstimator, TransformerMixin):
    """Time window data augmentation class for EEG data.

    This class provides functions for time window data augmentation on EEG data. It takes an EEG array
    and splits it into samples with different window sizes and steps.

    Attributes:
        fs (int): The sampling rate of the EEG array in Hz.
        window_width (float): The width of the window in seconds.
        window_step (float): The step of the window in seconds.

    Methods:
        fit(X, y=None): This method is not used.
        transform(X, y=None): This method splits the EEG array into samples with different
        window sizes and steps.

    Example:
        >>> eeg = np.random.randn(90, 64, 1000) # generate a random EEG array
        >>> label = np.random.randint(0, 2, 90) # generate a random label array
        >>> fs = 250 # set the sampling rate to 250 Hz
        >>> window_width = 1.5 # set the window width to 1.5 seconds
        >>> window_step = 0.1 # set the window step to 0.1 seconds
        >>> da = DataAugmentation(fs, window_width, window_step) # initialize the DataAugmentation class
        >>> samples = da.transform(eeg) # split the EEG array into samples
        >>> print(samples.shape) # print the shape of the samples
        (4050, 64, 150)
    """
    
    def __init__(self, fs=250, window_width=2, window_step=0.2):
        self.fs = fs
        self.window_width = window_width
        self.window_step = window_step
    
    def __repr__(self):
        return "TimeWindowDataAugmentation(fs={}, window_width={}, window_step={})".format(
            self.fs, self.window_width, self.window_step
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """ Split the EEG array into samples with different window sizes and steps.

        This method takes an EEG array and splits it into samples with different window sizes
        and steps. The window size and step are specified in the constructor of the class.

        Args:
            X (numpy.ndarray): The EEG array to be split. Shape should be (n_samples, n_channels, n_timepoints).

        Returns:
            numpy.ndarray: The augmented samples.

        Raises:
            ValueError: If X is not a 3D array.
        """
        # check if X is a 3D array
        if len(X.shape)!= 3:
            raise ValueError("X must be a 3D array")
        # convert window_width and window_step from seconds to samples
        width = int(self.window_width * self.fs)
        step = int(self.window_step * self.fs)
        # get the shape of X
        n_samples, n_channels, n_timepoints = X.shape
        # initialize an empty list to store the augmented samples
        augmented_samples = []
        # loop through each sample
        for i in range(n_samples):
            # get the current sample
            sample = X[i]
            # initialize the start and end indices of the window
            start = 0
            end = width
            # loop until the end index exceeds the number of timepoints
            while end <= n_timepoints:
                # get the current window
                window = sample[:, start:end]
                # append the window to the augmented_samples list
                augmented_samples.append(window)
                # update the start and end indices by adding the step size
                start += step
                end += step
        # convert the list to a numpy array and return it
        return np.array(augmented_samples)

