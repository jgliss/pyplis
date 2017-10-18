# -*- coding: utf-8 -*-
"""
Pyplis test module for dataset.py base module of Pyplis
"""

from pyplis import Dataset, __dir__, Filter, Camera, Source, MeasSetup,\
    CellCalibEngine
from os.path import join, exists
from datetime import datetime
from numpy.testing import assert_almost_equal
import pytest

BASE_DIR = join(__dir__, "data", "testdata_minimal")
START_PLUME = datetime(2015, 9, 16, 7, 10, 00)
STOP_PLUME = datetime(2015, 9, 16, 7, 20, 00)

START_CALIB = datetime(2015, 9, 16, 6, 59, 00)
STOP_CALIB = datetime(2015, 9, 16, 7, 3, 00)

CALIB_CELLS = {'a37'    :   [8.59e17, 2.00e17],
               'a53'    :   [4.15e17, 1.00e17],
               'a57'    :   [19.24e17, 3.00e17]}

if exists(BASE_DIR):
       
    @pytest.fixture
    def setup():
        cam_id = "ecII"
        
        ### Define camera (here the default ecII type is used)
        img_dir = join(BASE_DIR, "images")
        
        ### Load default information for Etna
        source = Source("etna")
        
        #### Provide wind direction
        wind_info= {"dir"     : 0.0,
                    "dir_err"  : 1.0}
    
        #camera location and viewing direction (altitude will be retrieved automatically)                    
        geom_cam = {"lon"           :   15.1129,
                    "lat"           :   37.73122,
                    "elev"          :   20.0,
                    "elev_err"      :   5.0,
                    "azim"          :   270.0,
                    "azim_err"      :   10.0,
                    "alt_offset"    :   15.0,
                    "focal_length"  :   25e-3} 
    
        #the camera filter setup
        filters= [Filter(type="on", acronym="F01"),
                  Filter(type="off", acronym="F02")]
        
        cam = Camera(cam_id, filter_list=filters, **geom_cam)
    
        return MeasSetup(img_dir, 
                         camera=cam,
                         source=source, 
                         wind_info=wind_info,
                         cell_info_dict=CALIB_CELLS)
        
    @pytest.fixture(scope="module")
    def calib_dataset():
        stp = setup()
        stp.start = START_CALIB
        stp.stop = STOP_CALIB
        
        return CellCalibEngine(stp)
    
    @pytest.fixture(scope="module")
    def plume_dataset():
        """Initiates measurement setup and creates dataset from that"""
        
        stp = setup()
        stp.start = START_PLUME
        stp.stop = STOP_PLUME
        print stp.start, stp.stop
        ### Create analysis object (from BaseSetup)
        # The dataset takes care of finding all vali
        return Dataset(stp)   
        
    def test_geometry():
        geom = plume_dataset().meas_geometry
        res = geom.get_all_pix_to_pix_dists()
        assert_almost_equal([1.9032587, 1.9032587, 10232.567],
                            [res[0].mean(), res[1].mean(), res[2].mean()],
                            3)
    
    
    
        
if __name__=="__main__":
    ds = plume_dataset()
    #cell = calib_dataset()
# =============================================================================
#     cell.find_and_assign_cells_all_filter_lists()
#     cell.plot_cell_search_result()
#     
# =============================================================================
