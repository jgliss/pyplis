# -*- coding: utf-8 -*-
"""
PYGASCAM LOAD SETUP 

v0.1

Load setup (created in ex1 script and saved at 

/../TEMP/pygascam_setup_eval_20141122_1218_1222_Guallatiri_ecII_1106307/

and create :class:`pygascam.DataSets.DataSet` object and do some basic stuff 
with it.

Author: Jonas Gliß
Email: jg@nilu.no
Copyright: Jonas Gliß
"""

import pygascam as gc

createDataset=1

basepath='../data/TEMP/'
#the evaluation ID (automatically created in ex1)
evalID='pygascam_setup_eval_20141122_1218_1222_Guallatiri_ecII_1106307'

setupFile=basepath + evalID + "/" + evalID + ".stp"

#now load the setup
setup=gc.load_setup(setupFile)

ev=gc.Evaluation.SO2CamEval(setup)
if createDataset:
    ev.create_main_dataset(checkSettings=0)
    ev.save_dataset()
else:
    ev.load_dataset()
    
ds=ev.dataSet
"""In the following, the rectangles available in the Dataset object are registered
in the optical flow modules of the individual image lists, this is done in order
to reduce optical flow calculations to specific ROIs defined by rectangles
"""
#register the rectanglesds.add_roi_at_pcs("plume")
ds.set_rects_optflow_module() 
#automatically add one ROI to the plume cross section line with ID "plume" 
#(which was defined in the setup script ex1_..)
ds.add_roi_at_pcs("plume") 
#get the offband image list
offList=ds.get_list("Off") #equivalent: ds.sortedList.lists["Off"]
#activate the new ROI in optflow calculations (default ROI for optFlow is whole image)
offList.optFlowEdit.change_roi("plume") #offList.optFlowEdit.change_roi() sets back to whole image
#finally print available forms
print ds.forms.rects
print ds.forms.lines
ds.open_in_gui()