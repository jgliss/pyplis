# -*- coding: utf-8 -*-
"""
Collection of matplotlib.pyplot based widgets for interactive plotting
"""
from PyQt4.QtGui import *
from PyQt4.QtCore import pyqtSignal

from matplotlib.pyplot import Figure, Line2D, subplots, draw
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as\
                                                                FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as\
                                                            NavigationToolbar
from matplotlib import rcParams

from numpy import nan, linspace, sign

class FigureDispWidget(QWidget):
    """Display a list of figures in a VBoxlayout
    
    .. todo:: 
    
        there must be a more elegant solution to this (done in a hurry..)
    
    """
    def __init__(self, figs=[], parent = None):
        super(FigureDispWidget, self).__init__(parent)
        self.layout=QVBoxLayout()
        self.figs = figs
        self.canvases=[]
        self.setLayout(self.layout)
        self.sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.init_layout()
        
    def init_layout(self):
        for fig in self.figs:
            c=FigureCanvas(fig)
            self.canvases.append(c)
            c.setSizePolicy(self.sizePolicy)
            c.updateGeometry()
            self.layout.addWidget(c)            
        self.show()
                       
class DoubleGraphCanvasMain(QWidget):
    """This widget contains the two GraphCanvases shown in the MainTabView
    """
    def __init__(self, baseviewer,parent=None):
        super(DoubleGraphCanvasMain, self).__init__(parent)

        self.histoCanvas = HistoCanvas(parent=self)
        self.graphCanvas = GraphCanvas(parent=self)
        self.graphCanvas.main_view_adjust()
        self.baseViewer = baseviewer
        
        self.create_layout()
        
    def create_layout(self):
        layout=QVBoxLayout()
        hBoxCanvases=QHBoxLayout()
        hBoxCanvases.addWidget(self.histoCanvas)
        hBoxCanvases.addWidget(self.graphCanvas)
        layout.addLayout(hBoxCanvases)
        self.setLayout(layout) 
        
class GraphCanvas(QWidget):
    """Widget representing a Matplotlib canvas"""
    updateContrast=pyqtSignal()
    def __init__(self, parent = None, width = 3, height = 2, dpi = 80,\
                                    addToolbar = False, projection3D = False):
        super(GraphCanvas, self).__init__(parent)
        self.layout = QVBoxLayout()
        self.fig = Figure(figsize = (width, height), dpi = dpi, facecolor = 'w')
        self.canvas = FigureCanvas(self.fig)
        
        if projection3D:
            self.axes = self.fig.add_subplot(111, projection = "3d")
        else:
            self.axes = self.fig.add_subplot(111)
        
        self.toolbar = None
        #self.canvas.mpl_connect('button_press_event', self.onclick)

               
        self.setParent(parent)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#==============================================================================
#         sizePolicy.setHeightForWidth(True)
#==============================================================================
        self.canvas.setSizePolicy(sizePolicy)
                        
        self.canvas.updateGeometry()
        
        self.currentGraph = None
        self.currentBackground = self.fig.canvas.copy_from_bbox(self.axes.bbox)

                
        self.lines = {}
        if addToolbar:
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        self.setLayout(self.layout)
        print "successfully initiated GraphCanvas"
    
    def update_fontsize(self, fontsizeLabels):
        """Update the fontsize
        
        :param int fontsizeLabels: desired fontsize of labels
        """
        ax = self.axes
        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsizeLabels)        
        ax.title.set_fontsize(fontsizeLabels +2)
        
    def main_view_adjust(self, fontSize = 8):
        """Adjust for main view mode"""
        self.fig.subplots_adjust(top=0.95, bottom=0.2, left=0.05)
        rcParams.update({'font.size' : fontSize})
        
    def set_axes(self,axes):
        """Set the current axes object and draw"""
        self.axes = axes
        self.canvas.draw()
        
    def add_line(self, id, xdata, ydata, **kwargs):
        """add a line"""
        l = Line2D(xdata, ydata, picker=True, **kwargs)
        self.lines[id] = l
        self.axes.add_line(l)
        self.canvas.draw()
        
    def print_bla(self):
        print "Blaaaaa"

    def draw_graph(self,x = None, y = None, **kwargs):
        self.axes.cla()
        self.currentGraph = self.axes.plot(x,y,**kwargs)
        self.currentBackground = self.fig.canvas.copy_from_bbox(self.axes.bbox)
        self.fig.subplots_adjust(hspace = 0.4,top = 0.95, left = 0.05)
        self.canvas.draw()

#==============================================================================
#     def onclick(self,event):
#         print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
#         event.button, event.x, event.y, event.xdata, event.ydata)
#==============================================================================
        #self.draw_vline(event.xdata)
class HistoCanvas(GraphCanvas):
    """A subclass of :class:`GraphCanvas` for displaying a histogram in an image
    viewer
    """
    def __init__(self, parent = None, width = 3, height = 2, dpi = 80,\
                                            fontSize = 8, addToolbar = False):
        super(HistoCanvas, self).__init__(parent, width, height, dpi,\
                                                                addToolbar)
        
        #self.draggableLines = {}
        self.dragLines = None
        self.autoUpdate = True
        self.fixRangeMode = True
        
        self.currentGraph = None
        
        self.lowI = 0
        self.highI = 0
        self.main_view_adjust()
        self.init = False
    
    def init_axes_range(self, hist, bins):
        """Initiate the displayed x and y range of the axis object
        
        Check if bitDepth info is available and if so, then fix the x-axis 
        range of the histogram from 0 to the maximum possible count. If not
        (e.g. for tau images), the expected range of possible values is unknown
        and needs to be updated when a new image is displayed.
        """
        self.xArray = linspace(bins[0], bins[-2], len(bins)-1)
        self.axes.set_xlim(bins[0], bins[-2])
        self.axes.set_ylim(0, hist.max()*1.2)
        self.axes.grid()
            
    def update_x_axis(self, bins):
        """Update the x axis range"""
        i, f = bins[0] - abs(bins[0])*0.2, bins[-2] + bins[-2]*0.2
        self.xArray = linspace(i, f, len(bins) - 1)
        self.axes.set_xlim(i, f)
    
    def init_histogram(self, imgObj, includeAxisLabels = True):
        """Init the histogram
        
        :param Img imgObj: image object used for determining the init histogram
        :param bool includeAxisLabels: if false, the axes are displayed without
            labels
            
        Creates histo from image and initates the axis labels etc based on this,
        also the drag lines (:class:`TwoDragLinesHor` object) used for 
        interactively changing the contrast
        """
        self.axes.cla()
        hist, bins = imgObj.make_histogram()
        self.init_axes_range(hist, bins)
        maxC = hist.max()*1.2
        if includeAxisLabels:
            self.axes.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            self.axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        else:
            self.axes.get_xaxis().set_visible(False)
            self.axes.get_yaxis().set_visible(False)            
            
        self.currentBackground = self.fig.canvas.copy_from_bbox(self.axes.bbox)
        self.fig.subplots_adjust(top = 0.85, bottom = 0.2, left = 0.2)
        low = bins[0]
        high = bins[-1]
        self.add_line("lowI",[low,low],[0,maxC],color="g", linewidth = 3)
        self.add_line("highI",[high,high],[0,maxC],color="g", linewidth = 3)
        self.dragLines = TwoDragLinesHor(self.lines["lowI"],\
                    self.lines["highI"], parent = self)
        self.dragLines.connect()
        if imgObj._meta("bitDepth") is nan:
            self.fixRangeMode = False
        else:
            self.fixRangeMode = True
        self.init = True
            
    def draw_histogram(self, imgObj):
        """Draw histogram of input image into axes
        
        :param Img imgObj: image object
        """
        try:
            self.currentGraph.remove()
        except:
            pass
        if not self.init:
            self.init_histogram(imgObj)
        lowI, highI, hist, bins = imgObj.get_brightness_range()
        self.axes.set_ylim(0, hist.max()*1.2)
        self.dragLines._update_yrange(0, hist.max()*1.2)
        if not self.fixRangeMode:
            self.update_x_axis(bins)
        self.currentGraph = self.axes.fill_between(self.xArray,hist,0,\
                                                    color='#0000FF',alpha=0.5)                    
        if self.autoUpdate:
            self.dragLines.change_lines_position(lowI, highI)
            self.write_contrast()
#==============================================================================
#             self.lines["lowI"].set_xdata([lowI,lowI])
#             self.lines["highI"].set_xdata([highI,highI])
#             self.dragLines.update()
#==============================================================================
            
        #self.update_contrast()
                                                    
#==============================================================================
#     def update_contrast(self):        
#         if self.updateMode:
#==============================================================================
    def write_contrast(self):        
        self.lowI, self.highI = self.dragLines.xCoordinates
#==============================================================================
#         print "WRITE CONTRAST IN HISTOCANVAS"
#         print "CURRENT LOW/HIGH BORDERS INTENSITY: "
#         print str(self.lowI) + " / " + str(self.highI)
#==============================================================================
        
    def handle_brigthness_change(self):
        try:
            self.write_contrast()
            self.parent().baseViewer.set_disp_im(cmin=self.lowI,cmax=self.highI)
            self.updateContrast.emit()
        except:
            print "Unknown error updating the brightness range of diplayed img"

class DragLineX(object):
    """:class:`matplotlib.lines.Line2D` object draggable in X direction
    
    A class providing horziontal dragging of a (2DLine) line object of plt on
    a graph canvas. So far, only in x direction, easily expandable for all 
    directions, see commented code in :func:`motion_notify_event`
    
    """
    def __init__(self, l, parent=None):
        self.line = l
        self.press = None
        self.color = l.get_color()
        self.parent = parent
    
    def connect(self):
        """Connect this object to the current figure canvas"""
        self.cidpress = self.line.figure.canvas.mpl_connect(\
                        'button_press_event', self.button_press_event)
        self.cidrelease = self.line.figure.canvas.mpl_connect(\
                        'button_release_event', self.button_release_event)
        self.cidmotion = self.line.figure.canvas.mpl_connect(\
                        'motion_notify_event', self.motion_notify_event)
    
    def disconnect(self):
        'disconnect all the stored connection ids'
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)

    def button_press_event(self,event):
        print "button press event"
        if event.inaxes != self.line.axes:
            return
#==============================================================================
#         print self.line.get_xdata()[0]
#         print event.xdata
#==============================================================================
        
        if not self.line.contains(event)[0]: return
            
        self.press = self.line.get_xdata(), self.line.get_ydata(), event.xdata,\
            event.ydata

    def button_release_event(self,event):
        self.press = None
        self.line.figure.canvas.draw()
        if self.parent is not None:
            print self.parent.lines.keys()

    def motion_notify_event(self, event):
        if self.press is None: 
            contains = self.line.contains(event)[0]
            if not contains: 
                self.line.set_color(self.color)
                self.line.figure.canvas.draw()
                return
            self.line.set_color("r")
            self.line.figure.canvas.draw()
            return
        if event.inaxes != self.line.axes: return
        xdata, ydata, xpress, ypress = self.press
        dx = event.xdata - xpress
        #dy = event.ydata - ypress
        self.line.set_xdata([xdata[0]+dx,xdata[1]+dx])
        #self.line.set_ydata([ydata[0]+dy,ydata[1]+dy])
        self.line.figure.canvas.draw()

class DragLineX_new(Line2D):
    """:class:`matplotlib.lines.Line2D` object draggable in X direction
    
    A class providing horziontal dragging of a (2DLine) line object of plt on
    a graph canvas. So far, only in x direction, easily expandable for all 
    directions, see commented code in :func:`motion_notify_event`    
    """
            
    def __init__(self, *args, **kwargs):
        super(DragLineX_new, self).__init__(*args, **kwargs)
        self.press = None
        self.color = self.get_color()
        
        self.cidpress = None
        self.cidrelease = None
        self.cidmotion = None
    
    @property
    def canvas(self):
        """Try to access current figure canvas"""
        try:
            return self.figure.canvas
        except AttributeError:
            print "figure canvas not yet assigned, create new figure"
            fig, ax = subplots(1,1)
            ax.add_line(self)
            ax.autoscale()
            draw()
            return fig.canvas
        except:
            raise
    
    def get_axes(self):
        """Try to access axes object and create new one if applicable"""
        if not self.axes == None:
            return self.axes
        try:
            self.canvas
            return self.axes
        except:
            raise
            
    def connect(self):
        """Connect to canvas"""
        canvas = self.canvas

        self.cidpress = canvas.mpl_connect('button_press_event',\
                                                    self.button_press_event)
        self.cidrelease = canvas.mpl_connect('button_release_event',\
                                                    self.button_release_event)
        self.cidmotion = canvas.mpl_connect('motion_notify_event',\
                                                    self.motion_notify_event)
                                                    
    def disconnect(self):
        """Diconnect from canvas"""
        try:
            self.canvas.mpl_disconnect(self.cidpress)
            self.canvas.mpl_disconnect(self.cidrelease)
            self.canvas.mpl_disconnect(self.cidmotion)
        except:
            pass
        
    def button_press_event(self,event):
        """What to do when a button is pressed"""
        if event.inaxes != self.get_axes():
            return
        if event.button ==1:
            if not self.contains(event)[0]: return
            self.press = self.get_xdata(), self.get_ydata(), event.xdata,\
                event.ydata
            return
        elif event.button == 3:
            print "Here could be a pop up menu..."

    def button_release_event(self,event):
        """What to do when pressed button is released"""
        self.press = None
        try:
            self.canvas.draw()
        except:
            raise
    @property
    def x_lim(self):
        d=self.get_xdata()
        add=(d[1]-d[0])*.05
        return [d[0]-add, d[1]+add]
    def motion_notify_event(self, event):
        """What to do if motion is registered on the canvas"""
        if self.press is None: #plain motion, no button pressed currently
            if not self.contains(event)[0]: #mouse not on line => set default color
                self.set_color(self.color)
                self.canvas.draw()
                return
            self.set_color("r") #mouse on line => change color to red
            self.canvas.draw()
            return
        #The following is only accessed if a button is currently pressed
        if event.inaxes != self.get_axes(): 
            return        
        xdata, ydata, xpress, ypress = self.press
        dx = event.xdata - xpress
        #dy = event.ydata - ypress
        self.set_xdata([xdata[0]+dx,xdata[1]+dx])
        self.get_axes().set_xlim(self.x_lim)
        #self.line.set_ydata([ydata[0]+dy,ydata[1]+dy])
        self.canvas.draw()
        
class TwoDragLinesHor(object):
    """Two draggable horizontal lines specifiying x-axis range
    
    A class providing horziontal dragging of two (2DLine) line objects 
    in matplotlib figures. The lines have to be vertical, the 
    purpose of this class is e.g. for selection of an x-range in a plot.
    """  
    def __init__(self, l1, l2, parent = None):
        self.lines = {1     :   l1,
                      2     :   l2}
        
        self.x_coords = []
        
        self.changes_applied = 0 #flag set for changes applied (i.e. emission of self.changed signal)
        self.connected = 0
        self.move = None
        self.expand = None
        self.colors = {}
        self.parent = parent
        self.get_colors()
        self.cidpress = {}
        self.cidrelease = {}
        self.cidmotion = {}
        self.canvas = l1.figure.canvas
        
        if self._input_ok():
            self.vspan = self.lines[1].axes.axvspan(self.lines[1].get_xdata()\
                [0], self.lines[2].get_xdata()[0], alpha = 0.3, color = 'g')
            self._sign_init = self._get_sign()
            self.write_current_coordinates()
        else:
            raise Exception(":mod:`self` could not be initialised, check input")
            
    def _input_ok(self):
        """Check if lines are :class:`Line2D` objects"""
        for key, line in self.lines.iteritems():
            if not isinstance(line, Line2D):
                return 0
            xd=line.get_xdata()
            if not xd[0] == xd[1]:
                return 0
        return 1
            
    def _get_sign(self):
        """Is x pos of line2 > x pos of line 1"""
        return sign(self.lines[2].get_xdata()[0] - self.lines[1].get_xdata()[0])
        
    def get_colors(self):
        """Get the current line colors"""
        for key, line in self.lines.iteritems():
            self.colors[key] = line.get_color()
            
    def set_default_colors(self):
        """Change line colors to default color"""
        for key, line in self.lines.iteritems():
            line.set_color(self.colors[key])
        
    def connect(self):
        """Connect lines with canvas and activates event handling"""
        if self._input_ok():
            print "Connecting lines"
        #for key, line in self.lines.iteritems():
            self.cidpress = self.canvas.mpl_connect(\
                            'button_press_event', self.button_press_event)
            self.cidrelease = self.canvas.mpl_connect(\
                        'button_release_event', self.button_release_event)
            self.cidmotion = self.canvas.mpl_connect(\
                'motion_notify_event', self.motion_notify_event)
            self.connected=1
        else:
            print "Input error in TwoDragLinesHor: lines could not be connected"
            self.connected=0
        
    def disconnect(self):
        """disconnect all the stored connection ids"""
        if self.connected:
            self.canvas.mpl_disconnect(self.cidpress)
            self.canvas.mpl_disconnect(self.cidrelease)
            self.canvas.mpl_disconnect(self.cidmotion)
            self.connected = 0
    
    def grab_line(self, event):
        """Interactive grabbing of line
        
        :param event: a event (e.g. mouse click)
        
        Checks if this event is on one of the lines and returns the line and 
        its ID
        """
        for key, line in self.lines.iteritems():
            if line.contains(event)[0]:
                return line, key
        return None, None
        
    def grab_rect(self, event):
        """Interactive grabbing of rectangle between the two lines
        
        :param event: an event (e.g. mouse click)
        
        Checks if this event is on the rectangle, returns True or False
        """
        if self.vspan.contains(event)[0]:
            return True
        return False
            
    def write_current_coordinates(self):
        """Write the current x coordinates of both lines
        
        The coordinates are written into list ``self.x_coords``
        """
        self.x_coords = []
        for key, line in self.lines.iteritems():
            self.x_coords.append(line.get_xdata()[0])
        self.x_coords.sort()
    
    def _update_yrange(self, ymin, ymax):
        """Update y range of lines based on input values
        
        :param float ymin: lower value of y range
        :param float ymax: upper value of y range
        """
        for l in self.lines.values():
            l.set_ydata([ymin, ymax])
            
    def change_lines_position(self, x0, x1):
        """Update x positions of both lines
        
        :param float x0: new position (in axes coords) of line 1
        :param float x1: new position (in axes coords) of line 2
        """
        self.lines[1].set_xdata([x0,x0])
        self.lines[2].set_xdata([x1,x1])
        self.update()
        
    def update(self):
        """Update this object 
        
        Called in :func:`button_release_event`. Writes current coordinates,
        update the rect position, draws in canvas, resets flag 
        ``self.changes_applied``
        
        .. note::
            If ``self.parent`` is :class:`piscope.gui.PlotWidgets.HistoCanvas`` 
            then :func:`handle_brigthness_change` therein is called (e.g. in
            piscope main application)
            
        """
        print "Update double drag line object"
        #print self.vspan.get_xy()
        self.write_current_coordinates()
        self.update_rect()
        self.canvas.draw()
        print "Line 1 pos: " + str(self.lines[1].get_xdata()[0])
        print "Line 2 pos: " + str(self.lines[2].get_xdata()[0])
        self.changes_applied = 0
        try:
            self.parent.handle_brigthness_change()
        except:
            print ("Error in TwoDragLinesHor.update(): could not emit signal "
            "in parent.")
        
    def button_press_event(self,event):
        """What to do when a button is pressed while on the canvas"""
        #print "button press event"
        if event.inaxes != self.lines[1].axes:
            return
        #print self.line.get_xdata()[0]
        #print event.xdata
        line, key=self.grab_line(event)
        #print "Line found: " + str(line) + "key: " + str(key)
        if line is not None: 
            self.expand = line, line.get_xdata(), event.xdata
            return
        if self.grab_rect(event):
            self.move=event.xdata,self.lines[1].get_xdata(),\
                self.lines[2].get_xdata()
            return
            
    def button_release_event(self,event):
        """What to do when a button is released"""
        self.expand = None
        self.move = None
        self.update()
        
    def update_rect(self):
        """Update the rectangle between the two lines
        
        Changes x position of left and right end of this rect        
        """
        x0, x1 = [self.lines[1].get_xdata()[0], self.lines[2].get_xdata()[0]]
        xy = self.vspan.get_xy()
        for k in range(len(xy)):
            if k not in [2,3]:
                xy[k][0] = x0
            else:
                xy[k][0] = x1
        self.vspan.set_xy(xy)

    def motion_notify_event(self, event):
        """What to do when there is mouse movement on the canvas"""
        if event.inaxes != self.lines[1].axes: return
        if self.move is not None:
            xpress,xL1,xL2=self.move
            dx = event.xdata - xpress
            self.lines[1].set_xdata([xL1[0]+dx,xL1[1]+dx])
            self.lines[2].set_xdata([xL2[0]+dx,xL2[1]+dx])
            self.update_rect()
            self.canvas.draw()
            self.changes_applied=1
            return
        if self.expand is None: 
            line,key = self.grab_line(event)
            if line is None: 
                self.set_default_colors()
                self.canvas.draw()
                return
            line.set_color("r")
            self.canvas.draw()
            return
        line, x0, xpress = self.expand
        if not self._sign_init == self._get_sign() :
            msg="Line was drawn too far and is reset to initial value"
            print msg
            line.set_xdata([x0[0],x0[1]])
            self.update_rect()  
            #self.canvas.draw()
            self.move=self.expand=None
            return
        dx = event.xdata - xpress
        #dy = event.ydata - ypress
        line.set_xdata([x0[0]+dx,x0[1]+dx])
        self.update_rect()        
        #self.line.set_ydata([ydata[0]+dy,ydata[1]+dy])
        self.canvas.draw()
        self.changes_applied=1

if __name__ == "__main__":
    l=DragLineX_new([5,15],[7,23])
    l.connect()     
"""Sorted out stuff
"""
#==============================================================================
# class MultiPlotWidget(QWidget):
#     """
#     
#     A widget with NXM number of :class:`GraphCanvas`widgets specified by
#     
#     :param int rowNum: Number of rows
#     :param int colNum: Number of columns
#     
#     """
#     def __init__(self, rowNum, colNum,graphSize=(8,4),parent=None):
#         super(MultiPlotWidget, self).__init__(parent)
#         super(GraphCanvas, self).__init__(parent)
#         self.layout=QVBoxLayout()
#         self.fig = Figure(figsize=(width, height), dpi=dpi,facecolor='w')
#         self.axes = self.fig.add_subplot(111)
#         
#         self.canvas=FigureCanvas(self.fig)
#         #self.canvas.mpl_connect('button_press_event', self.onclick)
# 
#                
#         self.setParent(parent)
#         sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
# #==============================================================================
# #         sizePolicy.setHeightForWidth(True)
# #==============================================================================
#         self.canvas.setSizePolicy(sizePolicy)
#                         
#         self.canvas.updateGeometry()
#         
#         self.currentGraph=None
#         self.currentBackground = self.fig.canvas.copy_from_bbox(self.axes.bbox)
# 
#         
#         self.lines={}
#         
#         self.layout.addWidget(self.canvas)
# 
#         self.setLayout(self.layout)
#         self.setMinimumSize(800,400)
#         self.layout=QVBoxLayout()
#         self.rowNum=rowNum
#         self.colNum=colNum
#         self.canvases=[]
#         self.init_graph_canvases(graphSize)
# #==============================================================================
# #         grid=QGridLayout()
# #         grid.addWidget(GraphCanvas(),0,0)
# #         grid.addWidget(GraphCanvas(),0,1)
# #==============================================================================
#         self.setLayout(self.layout)
#         
#     def init_graph_canvases(self,graphSize=(8,4)):
#         grid=QGridLayout()
#         positions=[(i,j) for i in range(self.rowNum) for j in range(self.colNum)]
#         for k in range(len(positions)):
#             self.canvases.append(GraphCanvas(self,graphSize[0],graphSize[1]))
#             print positions[k]
#             grid.addWidget(self.canvases[k],positions[k][0],positions[k][1])
#         self.layout.addLayout(grid)
#         w=self.colNum*graphSize[0]+graphSize[0]*0.5
#==============================================================================
#==============================================================================
#         h=self.rowNum*graphSize[1]+graphSize[1]*0.5
#         self.resize(w,h)
#==============================================================================