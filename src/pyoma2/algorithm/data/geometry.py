# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:30:00 2024

@author: dpa
"""

import numpy as np
import pandas as pd
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

"""TODO fix type"""


class Geometry1(BaseModel):
    """Base class for output results."""
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    ## MANDATORY
    sens_names: list[str]
    sens_coord: pd.DataFrame | npt.NDArray[np.float64]# sensors' coordinates
    sens_dir: npt.NDArray[np.float64] # sensors' directions
    ## OPTIONAL
    sens_lines: npt.NDArray[np.int64] | None = None # lines connecting sensors
    bg_nodes: npt.NDArray[np.float64] | None = None # Background nodes
    bg_lines: npt.NDArray[np.int64] | None = None # Background lines
    bg_surf: npt.NDArray[np.float64] | None = None # Background surfaces

class Geometry2(BaseModel):
    """Base class for output results."""
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    ## MANDATORY
    sens_names: list[str] # sensors' names
    pts_coord: pd.DataFrame | npt.NDArray[np.float64] # points' coordinates
    sens_map: pd.DataFrame | npt.NDArray[np.float64] # mapping
    sens_sign: pd.DataFrame | npt.NDArray[np.float64] # sign

    ## OPTIONAL
    # Order reduction uno tra ["None", "xy","xz","yz","x","y","z"]
    order_red: None | str = None 
    sens_lines: npt.NDArray[np.int64] | None = None # lines connection sensors
    bg_nodes: npt.NDArray[np.float64] | None = None # Background nodes
    bg_lines: npt.NDArray[np.int64] | None = None # Background lines
    bg_surf: npt.NDArray[np.float64] | None = None # Background surfaces
    