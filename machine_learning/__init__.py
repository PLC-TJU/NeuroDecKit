# from .ml_classifier import ML_Classifier

from metabci.brainda.algorithms.decomposition import (
    CSP,
    MultiCSP,
    FBCSP,
    FBMultiCSP,
    SPoC,
    SSCOR,
    FBSSCOR,
    FilterBankSSVEP,
    SCCA,
    FBSCCA,
    ItCCA,
    FBItCCA,
    MsCCA,
    FBMsCCA,
    ECCA,
    FBECCA,
    TtCCA,
    FBTtCCA,
    MsetCCA,
    FBMsetCCA,
    MsetCCAR,
    FBMsetCCAR,
    TRCA,
    FBTRCA,
    TRCAR,
    FBTRCAR,
    SKLDA,
    STDA,
    DSP,
    FBDSP,
    TDCA,
    FBTDCA,  
)

from metabci.brainda.algorithms.manifold import (
    MDRM,
    FgMDRM,
    Alignment,
    RecursiveAlignment,
)

from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from .dsp import DCPM

from pyriemann.estimation import Covariances, Kernels, Shrinkage
from pyriemann.utils.mean import mean_covariance

from pyriemann.classification import MDM#, FgMDM
from pyriemann.classification import SVC as RKSVM
# from pyriemann.classification import FgMDM
# from pyriemann.tangentspace import FGDA
# from pyriemann.tangentspace import TangentSpace as TS
from .ts import TSclassifier_online as TSclassifier
from .ts import FgMDM_online as FgMDM # lda使用'eigen'求解器
from .ts import FGDA_online as FGDA
from .ts import TS_online as TS

# from pyriemann.spatialfilters import CSP as RiemannCSP
from .csp import CSP_weighted as RiemannCSP
from pyriemann.spatialfilters import Xdawn




# from moabb.pipelines.csp import TRCSP
from .csp import TRCSP_weighted as TRCSP

from moabb.pipelines.utils import FilterBank as FB

from .sblest import SBLEST_model as SBLEST

from .feature_select import MutualInformationSelector as MIBIF
from .feature_select import LassoFeatureSelector as LassoSelector

from .base import recursive_reference_center, chesk_sample_weight

__all__ = [
    'CSP',
    'MultiCSP',
    'FBCSP',
    'FBMultiCSP',
    'SPoC',
    'SSCOR',
    'FBSSCOR',
    'FilterBankSSVEP',
    'SCCA',
    'FBSCCA',
    'ItCCA',
    'FBItCCA',
    'MsCCA',
    'FBMsCCA',
    'ECCA',
    'FBECCA',
    'TtCCA',
    'FBTtCCA',
    'MsetCCA',
    'FBMsetCCA',
    'MsetCCAR',
    'FBMsetCCAR',
    'TRCA',
    'FBTRCA',
    'TRCAR',
    'FBTRCAR',
    'SKLDA',
    'STDA',
    'DSP',
    'FBDSP',
    'TDCA',
    'FBTDCA',
    'DCPM',
    'Covariances',
    'Kernels',
    'Shrinkage',
    'MDM',
    'TSclassifier',
    'FgMDM',
    'RKSVM',
    'RiemannSVC',    
    'RiemannCSP',
    'Xdawn',
    'TS',
    'FGDA',
    'TRCSP',
    'FB',     
    'SBLEST', 
    'MIBIF',   
    'RSF',
    'ML_Classifier',
    'LassoSelector'
]   




