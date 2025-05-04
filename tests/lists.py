from neurodeckit.ensemble_learning.el_classifier import EL_Classifier
from neurodeckit.transfer_learning.tl_classifier import TL_Classifier
from neurodeckit.deep_learning.dl_classifier import DL_Classifier
from sklearn.ensemble import AdaBoostClassifier as ABC
from neurodeckit.transfer_learning.mekt import MEKT
from neurodeckit.transfer_learning.kl import KEDA

# EL-EA-CSP
EL_Classifier(target_domain=None, dpa_method='EA', fee_method='CSP', fes_method=None, clf_method='LDA', end_method=None, ete_method=None)

# EL-RA-TS
EL_Classifier(target_domain=None, dpa_method='RA', fee_method='TS', fes_method=None, clf_method='LDA', end_method=None, ete_method=None)

# EL-RA-MDM-ABC (RAVE)
EL_Classifier(target_domain=None, dpa_method='RA', fee_method=None, fes_method=None, clf_method=None, end_method='ABC-MDM', ete_method=None)

# EL-RA-TS-LDA-ABC (RAVEplus)
EL_Classifier(target_domain=None, dpa_method='RA', fee_method=None, fes_method=None, clf_method=None, end_method='ABC-TS-LDA', ete_method=None)

# EL-RA-MEKT
EL_Classifier(target_domain=None, dpa_method='RA', fee_method=None, fes_method=None, clf_method=None, end_method='MEKT', ete_method=None)

# EL-RA-MEKT-LDA-ABC (RAVEplus2)
model = ABC(MEKT(target_domain=None, subspace_dim=10), n_estimators=100, algorithm='SAMME')
EL_Classifier(target_domain=None, dpa_method='RA', fee_method=None, fes_method=None, clf_method=None, end_method=model, ete_method=None)

# EL-RA-KEDA
model = KEDA(target_domain=None, subspace_dim=10)
EL_Classifier(target_domain=None, dpa_method='RA', fee_method=None, fes_method=None, clf_method=None, end_method=model, ete_method=None)

# EL-RA-KEDA-LDA-ABC (RAVEplus2)
model = ABC(KEDA(target_domain=None, subspace_dim=10), n_estimators=100, algorithm='SAMME')
EL_Classifier(target_domain=None, dpa_method='RA', fee_method=None, fes_method=None, clf_method=None, end_method=model, ete_method=None)

# SBLEST
TL_Classifier(target_domain=None, dpa_method='EA', fee_method=None, fes_method=None, clf_method=None, end_method=None, ete_method='SBLEST')

# EEGNet
model = DL_Classifier(model_name='EEGNet', n_classes=2, fs=128, device='cuda')
TL_Classifier(target_domain=None, dpa_method='EA', fee_method=None, fes_method=None, clf_method=None, end_method=None, ete_method=model)

# FBCNet
model = DL_Classifier(model_name='FBCNet', n_classes=2, fs=128, device='cuda')
TL_Classifier(target_domain=None, dpa_method='EA', fee_method=None, fes_method=None, clf_method=None, end_method=None, ete_method=model)

