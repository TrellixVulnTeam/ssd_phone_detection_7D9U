import numpy as np
priorbox=np.load("priorbox_class_stats.npy")
from scipy.io import savemat

savemat("priorbox_class_stats.mat",{"priorbox_class_stats":priorbox})