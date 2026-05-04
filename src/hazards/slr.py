from dataclasses import dataclass
import numpy as np


@dataclass
class LSLSim:
    sampleid: int
    years: np.ndarray
    lsl_ft: float
    rcp: float
    model: str


# model = {
#         "BRICK": {

#             "rcp2.6":
#             "rcp4.5":
#             "rcp6.0":
#             "rcp8.5":
#                  },

#         "K14": {

#             "rcp2.6":
#             "rcp4.5":
#             "rcp6.0":
#             "rcp8.5":
#             },

#         "DP16": {

#             "rcp2.6":
#             "rcp4.5":
#             "rcp6.0":
#             "rcp8.5":

#             }


#         }


def decidal_to_annual():

    return