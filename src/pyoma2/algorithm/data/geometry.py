# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:30:00 2024

@author: dpa
"""

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

"""TODO fix type"""


class Geometry1(BaseModel):
    """Base class for output results."""
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    ## MANDATORY
    sens_name: npt.NDArray[np.string] | list[str]
    # FIXME sens coor e meglio se lo facciamo come dataframe
    sens_coord: npt.NDArray[np.float32] # sensors' coordinates
    sens_dir: npt.NDArray[np.float32] # sensors' directions (array(n,3))
    ## OPTIONAL
    sens_lines: npt.NDArray[np.float32] # lines connection sensors (array(n,2))
    bg_nodes: npt.NDArray[np.float32] # Background nodes
    bg_lines: npt.NDArray[np.float32] # Background lines
    # bg_surf: npt.NDArray[np.float32] # Background surfaces

class Geometry2(BaseModel):
    """Base class for output results."""
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    ## MANDATORY
    order_red: None | str = None 
    # uno tra ["None", "xy","xz","yz","x","y","z"]
    sens_name: npt.NDArray[np.string] | list[str]# sensors' names (n, 1)
    # FIXME sens coor e meglio se lo facciamo come dataframe
    pts_coord: npt.NDArray[np.float32] # points' coordinates (n, 4or3)
    # sens_sign: npt.NDArray[np.float32] # sensors' sign (n, 1)
    sens_map: npt.NDArray[np.float32] # mapping (n, 3)
    ## OPTIONAL
    sens_dir: npt.NDArray[np.float32] # sensors' directions (array(n,3))
    sens_lines: npt.NDArray[np.float32] # lines connection sensors (array(n,2))
    bg_nodes: npt.NDArray[np.float32] # Background nodes
    bg_lines: npt.NDArray[np.float32] # Background lines
    # bg_surf: npt.NDArray[np.float32] # Background surfaces