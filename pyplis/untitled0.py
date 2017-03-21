# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:39:27 2017

@author: jg
"""
import matplotlib.pyplot as plt

plt.close("all")
fig = plt.figure()#figsize=(10,6))
gs = plt.GridSpec(3, 1, height_ratios = [.4, .4, .2], hspace=0.05)

ax2 = fig.add_subplot(gs[2])
ax0 = fig.add_subplot(gs[0], sharex=ax2)
ax1 = fig.add_subplot(gs[1], sharex=ax2)
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position("right")
ax1.set_ylabel("Blaaa")

ax0.set_xticklabels([])
ax1.set_xticklabels([])

#fig.tight_layout()
