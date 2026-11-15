"""Study constants shared by the baseline and neural runners."""

from connpre.protocol import N_ALL

SEEDS = (0, 1, 2, 3, 4)
LABEL_SIZES = (50, 100, 300, N_ALL)
N_CAL = 100
MIN_SITE = 40
TARGETS = ("age", "fiq")
INSITE_SPLITS = 5
INSITE_N = 100
TANGENT_TOL = 1e-3
