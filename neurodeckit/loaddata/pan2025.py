"""
cross-session motor imagery dataset from Pan et al 2025
Authors: LC.Pan <panlincong@tju.edu.cn>
Date: 2025/7/1
License: BSD 3-Clause License
"""

import logging
import os
import mne
import numpy as np
from pooch import retrieve
from scipy.io import loadmat
from moabb.datasets.base import BaseDataset
from moabb.datasets.download import get_dataset_path

log = logging.getLogger(__name__)

ID_List = [11704410, 11704404, 11704420, 11704417, 11704412, 11704421, 11704402, 11704418, 
           11704405, 11704407, 11704419, 11704411, 11704413, 11704409, 11704415, 11704406, 
           11704414, 11704416, 11704403, 11704408] # MATLAB v7文件ID 

FILES = [f"https://dataverse.harvard.edu/api/access/datafile/{i}" for i in ID_List]

def eeg_data_path(subject, base_path=''):
    """Load EEG data for a given subject from the Pan2023 dataset.

    Parameters
    ----------
    subject : int
        Subject number, must be between 1 and 10.
    base_path : str, optional
        Base path where the EEG data files are stored. Defaults to current directory.

    Returns
    -------
    list of str
        Paths to the subject's EEG data files.
    """
    if not 1 <= subject <= 10:
        raise ValueError(f"Subject must be between 1 and 10. Got {subject}.")

    # 使用列表推导式和循环来简化文件下载逻辑
    file_paths = [
        os.path.join(base_path, f"S{str(sub).zfill(2)}D{day}.mat")
        for sub in (subject,) for day in (1, 2)
    ]

    # 下载缺失的文件
    for i, file_path in enumerate(file_paths, start=1):
        if not os.path.isfile(file_path):
            url = FILES[subject*2 - 3 + i]
            retrieve(url, None, file_path, base_path, progressbar=True)

    return file_paths


class Pan2025(BaseDataset):
    """Motor Imagery dataset from Pan et al 2023.

    .. admonition:: Dataset summary


        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Pan2025        10       28           2                ~180            4s            250Hz            2
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    Experimental Paradigm: Refer to the article *Riemannian geometric and ensemble 
    learning for decoding cross-session motor imagery electroencephalography 
    signals* [1]_.

    ## Abstract
    The Pan2025 dataset is a collection of electroencephalography (EEG) signals
    from 10 subjects performing motor imagery (MI) tasks across two sessions. 
    The dataset aims to facilitate the study of cross-session variability in 
    MI-EEG signals and to support the development of robust brain-computer 
    interface (BCI) systems.

    ## Dataset Composition
    The dataset encompasses EEG recordings from 10 subjects, each participating 
    in two sessions. The sessions involve MI tasks with visual cues for left-
    handed and right-handed movements. Data acquisition was performed using a 
    Neuroscan SynAmps2 amplifier, equipped with 28 scalp electrodes following 
    the international 10-20 system. The EEG signals were sampled at a frequency 
    of 250Hz, with a band-pass filter applied from 0.01 to 200Hz to mitigate 
    power line noise. The collected data is stored in Matlab format, labeled by 
    subject and session number.

    ## Participants
    The participant cohort includes 10 individuals (three females), aged 22 to 25, 
    with two reporting left-handedness. All subjects were screened for neurological 
    and movement disorders, ensuring a healthy participant profile for the study.

    ## Experimental Paradigm
    Each experimental session comprised ~180 trials, segmented into three distinct 
    phases: Rest, Preparation, and Task. During the Rest Period (4 seconds), 
    subjects were instructed to remain relaxed without engaging in mental tasks. 
    The Task Period (4 seconds) required subjects to perform the MI task, visualizing 
    the movement corresponding to the provided cues, either left or right-handed. 
    This paradigm was designed to occur in a controlled, distraction-free 
    environment.
    For session 2, the first 30 trials were used for training, while the remaining 
    trials were used for testing, with online feedback provided to the subject. 
    Note that the sample sizes are not consistent across sessions and classes, as 
    some subjects performed more or less than 180 trials.

    ## Data Acquisition and Preprocessing
    EEG signals were captured using a Neuroscan SynAmps2 amplifier and 28 scalp 
    electrodes positioned per the 10-20 system. The sampling rate was set at 
    1000Hz, and a band-pass filter from 0.01 to 200Hz and a notch filter at 50Hz 
    were employed to exclude power line interference. The signals were downsampled 
    to 250Hz and archived in Matlab format, systematically named by subject and 
    session identifiers.

    ## Data Structure
    The dataset's structure is encapsulated in a Matlab file, comprising a struct 
    with the following components:
    - `data`:   A 3D matrix (`[n_channels, n_samples, n_trials]`) containing the 
                recorded EEG signals.
    - `label`:  A vector (`[n_trials]`) denoting each trial's label 
                (1 for left-handed MI, 2 for right-handed MI).
    - 'fs':     The sampling frequency of the EEG signals.
    - 'period': The length of each trial in seconds. where index 0 corresponds 
                to the task start marker.
    - 'chaninfo': A cell array containing the channel information.

    ## References
    .. [1]	L. Pan et al. (2023). Riemannian geometric and ensemble learning for 
    decoding cross-session motor imagery electroencephalography signals. Journal 
    of Neural Engineering, 20(6), 066011. https://doi.org/10.1088/1741-2552/ad0a01

    """

    def __init__(self, **kwargs):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=2,
            events=dict(
                left_hand=1,
                right_hand=2,
            ),
            code="Pan2025",
            # Full trial w/ rest is 0-5.5
            interval=[1.5, 5.5],
            paradigm="imagery",
            doi="10.1088/1741-2552/ad0a01",
            **kwargs,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
           
        montage = mne.channels.make_standard_montage("standard_1005")

        # fmt: off
        ch_names = [
            "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
            "C5", "C3", "C1", "Cz", "C2", "C4", "C6", 
            "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
            "P5", "P3", "P1", "Pz", "P2", "P4", "P6",
        ]
        # fmt: on

        ch_types = ["eeg"] * 28

        info = mne.create_info(
            ch_names=ch_names + ["STIM014"], ch_types=ch_types + ["stim"], sfreq=250 # type: ignore
        )
        
        sessions = self.data_path(subject, path=self.path)
        out = {}
        for sess_ind, fname in enumerate(sessions):
            data = loadmat(fname, squeeze_me=True, struct_as_record=False, verify_compressed_data_integrity=False)
            event_ids = data["label"].ravel()
            raw_data = np.transpose(data["data"], axes=[2, 0, 1])
            # de-mean each trial
            raw_data = raw_data - np.mean(raw_data, axis=2, keepdims=True)
            raw_events = np.zeros((raw_data.shape[0], 1, raw_data.shape[2]))
            raw_events[:, 0, 0] = event_ids
            data = np.concatenate([raw_data * 1e-6, raw_events], axis=1) #uV to V
            # add buffer in between trials
            # log.warning(
            #     "Trial data de-meaned and concatenated with a buffer to create " "cont data"
            # )
            zeroshape = (data.shape[0], data.shape[1], 50)
            data = np.concatenate([np.zeros(zeroshape), data, np.zeros(zeroshape)], axis=2)
            
            trialnum = self.split_array(range(data.shape[0])) 
            out[str(sess_ind)] = {}
            for run_ind in range(len(trialnum)):
                
                raw = mne.io.RawArray(
                    # ~ 30 trials per run/block
                    data=np.concatenate(list(data[trialnum[run_ind], :, :]), axis=1), info=info, verbose=False
                )
                raw.set_montage(montage) 
            
                out[str(sess_ind)][str(run_ind)] = raw
            
        return out
    
    def split_array(self, arr, M = 6):
        n = len(arr)
        
        if M <= 0:
            raise ValueError("M must be a positive integer")
        
        if M > n:
            return [[x] for x in arr] + [[] for _ in range(M - n)]
        
        base_size = n // M
        remainder = n % M
        
        result = []
        start = 0
        
        for i in range(M):
            end = start + base_size + (1 if i < remainder else 0)
            result.append(arr[start:end])
            start = end
        
        return result

    def data_path(self, subject, path=None):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        
        path = get_dataset_path("PAN", path)
        basepath = os.path.join(path,"MNE-pan2025-data")
        if not os.path.isdir(basepath):
            os.makedirs(basepath)

        return eeg_data_path(subject, basepath)
