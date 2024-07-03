"""
cross-session motor imagery dataset from Pan et al 2023
Authors: Pan.LC <panlincong@tju.edu.cn>
Date: 2024/3/18
License: MIT License
"""

import logging
import os

import mne
import numpy as np
from pooch import retrieve
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.download import get_dataset_path

log = logging.getLogger(__name__)


ID_List = [10358857, 10358855, 10358853, 10358856, 10358859,
           10358858, 10358861, 10358854, 10358860] 

FILES = [f"https://dataverse.harvard.edu/api/access/datafile/{i}" for i in ID_List]

def eeg_data_path(subject, base_path=''):
    """Load EEG data for a given subject from the Pan2023 dataset.

    Parameters
    ----------
    subject : int
        Subject number, must be between 1 and 9.
    base_path : str, optional
        Base path where the EEG data files are stored. Defaults to current directory.

    Returns
    -------
    list of str
        Paths to the subject's EEG data files.
    """
    if not 1 <= subject <= 9:
        raise ValueError(f"Subject must be between 1 and 9. Got {subject}.")

    # 使用列表推导式和循环来简化文件下载逻辑
    file_path = os.path.join(base_path, f"Subject{str(subject).zfill(2)}.mat")

    # 下载缺失的文件
    if not os.path.isfile(file_path):
        url = FILES[subject-1]
        retrieve(url, None, file_path, base_path, progressbar=True)

    return [file_path]


class BCIC2024A(BaseDataset):
    """Motor Imagery dataset from Pan et al 2023.

    .. admonition:: Dataset summary


        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        BCIC2024A        9       64           3                 90  4s            1000Hz                     1
        =========  =======  =======  ==========  =================  ============  ===============  ===========


    ## Abstract
    The BCIC2024A dataset 

    """

    def __init__(self, **kwargs):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=1,
            events=dict(
                left_hand=1,
                right_hand=2,
                feet=3,
            ),
            code="BCIC2024A",
            # Full trial w/ rest is 0-7
            interval=[0, 4],
            paradigm="imagery",
            doi="",
            **kwargs,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
           
        montage = mne.channels.make_standard_montage("standard_1005")

        # fmt: off
        ch_names = [
            "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", 
            "Fz", "F1", "F2", "F3", "F4", "F5", "F6","F7", "F8", 
            "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7", "FT8",
            "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8", 
            "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8", 
            "Pz", "P3", "P4", "P5", "P6", "P7", "P8", 
            "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", 
            "Oz", "O1", "O2",
        ]
        # fmt: on

        ch_types = ["eeg"] * 59

        info = mne.create_info(
            ch_names=ch_names + ["STIM014"], ch_types=ch_types + ["stim"], sfreq=1000
        )
        
        sessions = self.data_path(subject, path=self.path)
        out = {}
        for sess_ind, fname in enumerate(sessions):
            data = loadmat(fname, squeeze_me=True, struct_as_record=False, verify_compressed_data_integrity=False)
            event_ids = data["label"].ravel()
            raw_data = data["data"]
            # de-mean each trial
            raw_data = raw_data - np.mean(raw_data, axis=2, keepdims=True)
            raw_events = np.zeros((raw_data.shape[0], 1, raw_data.shape[2]))
            raw_events[:, 0, 0] = event_ids
            data = np.concatenate([1e-6 * raw_data, raw_events], axis=1)
            # add buffer in between trials
            log.warning(
                "Trial data de-meaned and concatenated with a buffer to create " "cont data"
            )
            zeroshape = (data.shape[0], data.shape[1], 50)
            data = np.concatenate([np.zeros(zeroshape), data, np.zeros(zeroshape)], axis=2)
            
            trialnum = int(data.shape[0]/3)
            
            out[str(sess_ind)] = {}
            for run_ind in range(3):
                
                raw = mne.io.RawArray(
                    # 30/50 trials per run/block
                    data=np.concatenate(list(data[trialnum*run_ind:trialnum*(run_ind+1), :, :]), axis=1), info=info, verbose=False
                )
                raw.set_montage(montage)   
            
                out[str(sess_ind)][str(run_ind)] = raw
            
        return out

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        path = get_dataset_path("BCIC", path)
        basepath = os.path.join(path,"MNE-bcic2024a-data")
        if not os.path.isdir(basepath):
            os.makedirs(basepath)

        return eeg_data_path(subject, basepath)
