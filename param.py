import numpy as np
import platform

VEC_DIM=10
VELOCITY_LIMIT=2
OMEGA_LIMIT=np.pi/4
TS=0.04
X_MAX = 8.08
Y_MAX = 4.48

SYSTEM="linux" if platform.system() != "Windows" else "win"
CONFRONTATION_NAME=f"{SYSTEM}_confrontation_v2.1/cog_confrontation_env.exe"
SIM2REAL_NAME=f"{SYSTEM}_v2.1/cog_sim2real_env.exe"