from moabb.pipelines.utils import FilterBank as FB
from pyriemann.estimation import Covariances, Kernels, Shrinkage
from pyriemann.utils.mean import mean_covariance
from pyriemann.classification import FgMDM, MDM
from pyriemann.classification import SVC as RKSVM
from pyriemann.spatialfilters import Xdawn

from .dsp import DCPM, DSP, FBDSP
from .cca import TRCA, FBTRCA

from .ts import TSclassifier_online as TSclassifier
from .ts import FgMDM_online as FgMDM
from .ts import FGDA_online as FGDA
from .ts import TS_online as TS

from .csp import CSP_weighted as RiemannCSP
from .csp import TRCSP_weighted as TRCSP
from .csp import FBCSP

from .sblest import SBLEST_model as SBLEST

from .feature_select import MutualInformationSelector as MIBIF
from .feature_select import LassoFeatureSelector as LassoSelector

from .base import recursive_reference_center, chesk_sample_weight






