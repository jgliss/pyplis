# -*- coding: utf-8 -*-
"""
piscope intorduction script no. 6 - Orientation integration lines
"""

from piscope.processing import LineOnImage
from os.path import join
import matplotlib.pyplot as plt

from ex1_measurement_setup_plume_data import save_path

plt.close("all")
fig, ax = plt.subplots(1, 2, figsize= (18,9))

lines_r = []
lines_l = []
# horizontal line, normal orientation to the top (0 deg)
lines_r.append(LineOnImage(0, 0, 1, 0, normal_orientation="right"))

# Vertical line normal to the right (90 deg)
lines_r.append(LineOnImage(x0 = 0, y0 = 0, x1 = 0, y1 = 1, normal_orientation="right"))

# horizontal line, normal orientation to the bottom (180 deg)
lines_l.append(LineOnImage(0, 0, 1, 0, normal_orientation="left"))

# Vertical line normal to the left (270 deg)
lines_l.append(LineOnImage(0, 0, 0, 1, normal_orientation="left"))

# Slanted line 45 degrees
lines_r.append(LineOnImage(.1, .3, .6, .8, normal_orientation="right"))

# Slanted line 45 degrees
lines_l.append(LineOnImage(.1, .3, .6, .8, normal_orientation="left"))

# Slanted line 45 degrees
lines_r.append(LineOnImage(1, 0, 0, 1, normal_orientation="right"))

# Slanted line 45 degrees
lines_l.append(LineOnImage(1, 0, 0, 1, normal_orientation="left"))

lines_r.append(LineOnImage(0.6, .1, .8, .9, normal_orientation="right"))
lines_l.append(LineOnImage(0.6, .1, .8, .9, normal_orientation="left"))

lines_r.append(LineOnImage(0.4, .1, .2, .9, normal_orientation="right"))
lines_l.append(LineOnImage(0.4, .1, .2, .9, normal_orientation="left"))

for k in range(len(lines_r)):
    line = lines_r[k]
    #print "%d: %s" %(k, line.orientation_info)
    normal = line.normal_vector
    lbl = "%d: n=[%.2f, %.2f], n_a= %.1f, norm = %.2f" %(k, normal[0],\
                                normal[1], line.normal_theta, line.norm)
    line.plot_line_on_grid(ax = ax[0], include_normal = 1, label = lbl)
for k in range(len(lines_l)):
    line = lines_l[k]
    normal = line.normal_vector
    lbl = "%d: n=[%.2f, %.2f], n_a= %.1f, norm = %.2f" %(k, normal[0],\
                                normal[1], line.normal_theta, line.norm)
    line.plot_line_on_grid(ax = ax[1], include_normal = 1, label = lbl)

ax[0].set_title("Orientation right")
ax[0].legend(loc = "best", fontsize=8, framealpha = 0.5)
ax[0].set_xlim([-.1, 1.1])
ax[0].set_ylim([1.1, -.1])

ax[1].set_title("Orientation left")
ax[1].legend(loc = "best", fontsize=8, framealpha = 0.5)
ax[1].set_xlim([-.1, 1.1])
ax[1].set_ylim([1.1, -.1])

fig.savefig(join(save_path, "intro6_out_1.png"))