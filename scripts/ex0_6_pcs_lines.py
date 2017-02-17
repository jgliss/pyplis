# -*- coding: utf-8 -*-
"""
pyplis intorduction script no. 6 - LineOnImage objects and their orientation

This script introduces how to create LineOnImage objects within the image plane
and specify their orientation (i.e. the direction into which the normal vector
of the line points). This is mainly important for emission rate retrievals, 
where, for instance velcoity displacement vectors (e.g. from an optical flow
algorithm) have to be multiplied with the normal vector of such a line (using
the dot product).


"""
from pyplis.processing import LineOnImage
from os.path import join
from matplotlib.pyplot import show, subplots, close
from matplotlib.cm import get_cmap

### IMPORT GLOBAL SETTINGS
from SETTINGS import SAVEFIGS, SAVE_DIR, FORMAT, DPI, OPTPARSE

def create_example_lines():
    """Create some exemplary lines"""
    lines_r = [] #lines with orientation "right" are stored within this list
    lines_l = [] #lines with orientation "left" are stored within this list
    
    cmap = get_cmap("jet")
    # horizontal line, normal orientation to the top (0 deg)
    lines_r.append(LineOnImage(x0=0, y0=0, x1=1, y1=0, 
                               normal_orientation="right",
                               color=cmap(0),
                               line_id="Line 1"))
    
    # horizontal line, normal orientation to the bottom (180 deg)
    lines_l.append(LineOnImage(0, 0, 1, 0, normal_orientation="left",
                               color=cmap(0),
                               line_id="Line 1"))
                               
    # Vertical line normal to the right (90 deg)
    lines_r.append(LineOnImage(0, 0, 0, 1, normal_orientation="right",
                               color=cmap(50),
                               line_id="Line 2"))
    
    # Vertical line normal to the left (270 deg)
    lines_l.append(LineOnImage(0, 0, 0, 1, normal_orientation="left",
                               color=cmap(50),
                               line_id="Line 2"))
    
    # Slanted line 45 degrees
    lines_r.append(LineOnImage(.1, .3, .6, .8, normal_orientation="right",
                               color=cmap(100),
                               line_id="Line 3"))
    
    # Slanted line 45 degrees
    lines_l.append(LineOnImage(.1, .3, .6, .8, normal_orientation="left",
                               color=cmap(100),
                               line_id="Line 3"))
    
    # Slanted line 45 degrees
    lines_r.append(LineOnImage(1, 0, 0, 1, normal_orientation="right",
                               color=cmap(150),
                               line_id="Line 4"))
    
    # Slanted line 45 degrees
    lines_l.append(LineOnImage(1, 0, 0, 1, normal_orientation="left",
                               color=cmap(150),
                               line_id="Line 4"))
    
    lines_r.append(LineOnImage(0.6, .1, .8, .9, normal_orientation="right",
                               color=cmap(200),
                               line_id="Line 5"))
                               
    lines_l.append(LineOnImage(0.6, .1, .8, .9, normal_orientation="left",
                               color=cmap(200),
                               line_id="Line 5"))
    
    lines_r.append(LineOnImage(0.4, .1, .2, .9, normal_orientation="right",
                               color=cmap(250),
                               line_id="Line 6"))
    lines_l.append(LineOnImage(0.4, .1, .2, .9, normal_orientation="left",
                               color=cmap(250),
                               line_id="Line 6"))
    
    return lines_r, lines_l
if __name__ == "__main__":
    close("all")
    fig, ax = subplots(1, 2, figsize= (18,9))
    
    lines_r, lines_l = create_example_lines()
    
    for k in range(len(lines_r)):
        line = lines_r[k]
        #print "%d: %s" %(k, line.orientation_info)
        normal = line.normal_vector
        lbl = "%s" %line.line_id
        line.plot_line_on_grid(ax = ax[0], include_normal = 1, label = lbl)
    for k in range(len(lines_l)):
        line = lines_l[k]
        normal = line.normal_vector
        lbl = "%s" %line.line_id
        line.plot_line_on_grid(ax = ax[1], include_normal = 1, label = lbl)
    
    ax[0].set_title("Orientation right")
    ax[0].legend(loc = "best", fontsize=8, framealpha = 0.5)
    ax[0].set_xlim([-.1, 1.1])
    ax[0].set_ylim([1.1, -.1])
    
    ax[1].set_title("Orientation left")
    ax[1].legend(loc = "best", fontsize=8, framealpha = 0.5)
    ax[1].set_xlim([-.1, 1.1])
    ax[1].set_ylim([1.1, -.1])
    ### IMPORTANT STUFF FINISHED    
    if SAVEFIGS:
        fig.savefig(join(SAVE_DIR, "ex0_6_out_1.%s" %FORMAT),
                           format=FORMAT, dpi=DPI)
        
    
    # Display images or not    
    (options, args)   =  OPTPARSE.parse_args()
    try:
        if int(options.show) == 1:
            show()
    except:
        print "Use option --show 1 if you want the plots to be displayed"
