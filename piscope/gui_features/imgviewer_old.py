# -*- coding: utf-8 -*-
from PyQt4.QtGui import QWidget, QColor, QVBoxLayout, QHBoxLayout, QPalette,\
    QScrollArea, QSplitter, QLabel, QAction, QPushButton, QLineEdit, QIcon,\
    QMessageBox, QPen, QPainter, QPixmap, QRubberBand
from PyQt4.QtCore import pyqtSignal, QSize, Qt, QObject, QLine, QPoint, QRect
from numpy import pi, sin, cos
from scipy.misc.pilutil import toimage
#from scipy.misc import bytescale 
from PIL.ImageQt import ImageQt
from time import time

from ..forms import LineCollection, RectCollection
from ..imagelists import BaseImgList
from ..image import Img
from ..inout import get_icon

from .PlotWidgets import DoubleGraphCanvasMain
from .EditWidgets import DispImagePrepSettings
from .ObjectWidgets import AllFormsPopup

class ImgViewer(QWidget):
    """Image list viewer with basic edit features
    
    Widget providing an image display pixmap embedded in :class:`QLabel`
    (for details see :class:`ImgLabel`) embedded into a :class:`QScrollArea` 
    widget to handle Zoom. Furthermore, 2 graph display widgets (for details see
    :class:`GraphCanvas`) are displayed below the image next to each other.
    Furthermore standard buttons for image handling are provided (e.g. next / 
    previous image, zoom in /  out, goto file number, fit to screen).
    """
    newIm = pyqtSignal()
    corrDark = pyqtSignal()
    reloadList=pyqtSignal()
    
    def __init__(self, id, imgList = None, buttonMode = "icons", parent = None):
        """Initialise image viewer
        
        :param str id: Display ID of this viewer (can be different from id of list displayed)
        :param BaseImgList imgList: the image list object to be displayed
        :param str buttonMode ("icons"): choose from `["icons", "normal"]`
        :param parent:
        """
        super(ImgViewer,self).__init__(parent)
        print "\nInit ImgViewer %s\n" %id
        self.id = id
        self.imgList = None        
        
        
        #icons for standard buttons (displayed in black)
        
        self.buttonModes=["icons","normal"]
          
        self.iconSize=QSize(25,25)
        
        self.menuIconNames =  {"prevIm"        :   "Arrow3 Left", 
                               "nextIm"        :   "Arrow3 Right", 
                               "zoomIn"        :   "Zoom In",
                               "zoomOut"       :   "Zoom Out",
                               "fit"           :   "Full Size",
                               "goto"          :   "Player FastFwd",
                               "add"           :   "Plus",
                               "currentEdit"   :   "Gear"}    

        self.modeButtonIconNames = {"drawMode"      :   "Write2",
                                    "editMode"      :   "Tool",
                                    "optFlow"       :   "myFlow",
                                    "tau"           :   "tau"}  
        #the 2 icons for each mode buttons will be loaded in this dictionary
        self.modeButtonIcons={True  :   {},
                              False :   {}}
                              
        self.modeButtonStyles={True     :   "background-color: #B2FFB2;"\
                                                        "font-weight: bold",
                               False    :   "background-color: #FF4D4D"}
                              
        self.drawIconNames= {"rect"    :   "myRect",
                             "line"    :   "myLine"}

        #the icons for the standard buttons will be loaded in this dictionary
        self.buttonIcons={}
        
        #the draw icons will be loaded in this dictionary
        self.drawIcons={}
    
        #button collection
        self.buttons={}
        #icons for buttons with 2 different modi (active / inactive)
        #will be displayed in green / red respectively
        self.modeButtons = {}
        
        #the popup window for the drawmode selection
        self.drawModePopup = AllFormsPopup(self.drawIconNames.keys(), self)
        self.drawModePopup.changed.connect(self.update_drawmode)
        
        #all actions are created within this dictionary
        self.actions={}
        
        #flag indicating whether data display is active
        self.active = False
        self.editActive = False
        
        self.connectedWidgets={}
        #self.simpleViewer=SimpleViewer()
        
        #Image management
        self.currentDispIm=None        

        self.zoomFac = 0.1
        #helper for string conversion of intensities at mouse position
        #if tauMode==0 then display 0 fractional digits
        #if tauMode==1 then display two fractional digits
        self.formatIntensity=["{:.0f}".format,
                              "{:.2f}".format]
        #init default colors for images, note that the image display is
        #performed as 8 Bit image
        self.colors=[QColor(i,i,i).rgb() for i in range(0,255)]
        
        #Additional own widgets
        self.imgLabel = ImgLabel(parent = self)
        
        self.graphs = DoubleGraphCanvasMain(baseviewer = self)
        self.histoCanvas = self.graphs.histoCanvas
        
        #create all actions +  corresponding buttons etc.
        self.create_actions()
        self.create_edit_menu()
        self.set_button_mode(buttonMode)
        
        #create layout widgets for header information
        self.create_header_layout()
        #
        self.create_toolbar_layout()
        self.create_edit_toolbar_layout()
        
        
        self.create_layout()
        
        self.enable_buttons(False)
        
        self.connect_to_img_label()
        self.set_list(imgList)
        #self.set_list(imgList)
#==============================================================================
#         self.palette().setColor(self.backgroundRole(),Qt.white)
#         self.setAutoFillBackground(True)
#==============================================================================
        #self.imgLabel.mouseReleaseEvent(self.print_viewport_alignment)
        
#==============================================================================
#     def print_viewport_alignment(self):
#         print "Mouse release event"
#         print "Viewport alignment scrollarea"
#         print self.scrollArea.alignment()
#==============================================================================
        
    def create_layout(self):
        """Create the layout of this widget"""
        self.layout=QVBoxLayout()
        
        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
#==============================================================================
#         self.scrollArea.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#==============================================================================
#==============================================================================
#         self.scrollArea.setMinimumHeight(600)
#==============================================================================
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.imgLabel)
        
        self.hBoxImDisp=QHBoxLayout()
        self.vBoxImDispSub=QVBoxLayout()
        self.vBoxImDispSub.addWidget(self.scrollArea)
        self.vBoxImDispSub.addLayout(self.hBoxTools)
        
#==============================================================================
#         self.hBoxGraphs=QHBoxLayout()
#         self.hBoxGraphs.addWidget(self.histoCanvas)
#         self.hBoxGraphs.addWidget(self.graphCanvas)
#==============================================================================
        
        self.layout.addLayout(self.hBoxHeader)
        self.layout.addLayout(self.hBoxSubHeader)
        #self.layout.addLayout(self.hBoxSelect)
        if self.buttonMode == "icons":
            self.hBoxImDisp.insertLayout(0,self.editLayout)
        else:
            self.layout.addLayout(self.editLayout)
        
        self.hBoxImDisp.addLayout(self.vBoxImDispSub)
        #self.layout.addWidget(self.imgLabel)
        
        self.viewer=QWidget()
        self.viewer.setLayout(self.hBoxImDisp)
        self.vSplitter=QSplitter(Qt.Vertical)
        
        
#==============================================================================
#         self.layout.addLayout(self.hBoxImDisp)
#         self.layout.addWidget(self.graphs)
#==============================================================================
        self.vSplitter.addWidget(self.viewer)
        self.vSplitter.addWidget(self.graphs)
        self.layout.addWidget(self.vSplitter)
        #self.layout.addLayout(self.hBoxEditMenu)  
        self.setLayout(self.layout)
        
        #self.load_standard_icons()
#==============================================================================
#         
#         self.imgList.import_filelist(fileList)
#         self.disp_list()
#==============================================================================
    def create_header_layout(self):
        """Create the header info layout (:class:`QHBoxLayout`)"""
        self.hBoxHeader = QHBoxLayout()
        self.headerLabel = QLabel(self.id + " images")
        self.headerLabel.setStyleSheet("font-weight: bold")#;font-size: 20pt")
        self.hBoxHeader.addWidget(self.headerLabel)
        
        self.mousePosition = QLabel("(  |  )")
        self.imgLabel.moved.connect(self._write_mouse_position)
        
        self.hBoxSubHeader = QHBoxLayout()
        self.subHeaderLabel = QLabel("")
        self.hBoxSubHeader.addWidget(self.subHeaderLabel)
        self.hBoxSubHeader.addStretch(1)
        self.hBoxSubHeader.addWidget(self.mousePosition)
    
    def create_actions(self):
        """Create all actions"""
        self.actions["prev"]=QAction("&Previous image...",self,\
            triggered=self.prev)
        self.actions["next"]=QAction("&Next image...",self,\
            triggered=self.next)
        self.actions["gotonum"]=QAction("&Go to image...", self,\
            triggered=self.goto_num)
        self.actions["zoom_in"]=QAction("&Zoom in...", self,\
            triggered=self.zoom_in)
        self.actions["zoom_out"]=QAction("&Zoom out...", self,\
            triggered=self.zoom_out)
        self.actions["fit"]=QAction("&Fit to screen...", self,\
            triggered=self.fit_to_screen)
        self.actions["chgActive"]=QAction("&Disable/Enable active mode...",\
            self, triggered=self.change_activemode)
        self.actions["currentEdit"]=QAction("&Current edit...",\
            self, triggered=self.show_current_edit)
        self.actions["chgDraw"]=QAction("&Disable/Enable drawing mode...",\
            self, triggered=self.change_drawmode)
        self.actions["addForm"]=QAction("&Add form to DataSet...",\
            self, triggered=self.add_form)     

    def create_edit_menu(self):
        """Create the edit menu
        
        This function creates all buttons, dropdown menus etc for the widget.
        Note, that in this function they are only created but not yet placed 
        in the widget.
        """
        self.buttons["prevIm"]=QPushButton("Previous")
        self.buttons["prevIm"].clicked.connect(self.prev)
        self.buttons["prevIm"].setToolTip("Click to display previous image")
        
        self.buttons["nextIm"]=QPushButton("Next")#U+21E8
        self.buttons["nextIm"].clicked.connect(self.next)
        self.buttons["nextIm"].setToolTip("Click to display the next image")
        
        self.editCurrentFileNum=QLineEdit()
        w=self.editCurrentFileNum.fontMetrics().boundingRect("4000").width() + 15
        self.editCurrentFileNum.setFixedWidth(w)
        self.buttons["goto"]=QPushButton("Go to file")
        self.buttons["goto"].clicked.connect(self.goto_num)
        
        self.buttons["zoomIn"]=QPushButton("+")
        self.buttons["zoomIn"].setStyleSheet("font-size: 22; font-weight: bold")
        self.buttons["zoomIn"].clicked.connect(self.zoom_in)
        
        self.buttons["zoomOut"]=QPushButton("-")
        self.buttons["zoomOut"].setStyleSheet("font-size: 22; font-weight: bold")
        self.buttons["zoomOut"].clicked.connect(self.zoom_out)
        
        zoomBtnWidth=self.buttons["zoomIn"].fontMetrics().boundingRect("+").width() + 15
        self.buttons["zoomIn"].setMaximumWidth(zoomBtnWidth)
        self.buttons["zoomOut"].setMaximumWidth(zoomBtnWidth)
        
        self.buttons["fit"]=QPushButton("Fit")
        self.buttons["fit"].setToolTip("Fit to screen")
        self.buttons["fit"].clicked.connect(self.fit_to_screen)
        

        self.modeButtons["editMode"]=QPushButton("Edit")
        self.modeButtons["editMode"].setToolTip("Activate or deactivate edit"
            " mode: Image editing (e.g. blurring etc, see edit menu) only "
            " applies to active image lists")
        self.modeButtons["editMode"].clicked.connect(self.change_activemode)
        
        self.modeButtons["drawMode"]=QPushButton("Draw")
        self.modeButtons["drawMode"].setToolTip("Activate or deactivate draw mode")
        self.modeButtons["drawMode"].clicked.connect(self.change_drawmode)
        
        self.modeButtons["optFlow"]=QPushButton("Flow")
        self.modeButtons["optFlow"].setToolTip("Overlay optical flow between "
            "this image and the next image in the time series")
        self.modeButtons["optFlow"].clicked.connect(self.change_optflow_mode)
        
        self.modeButtons["tau"]=QPushButton("Tau")
        self.modeButtons["tau"].setToolTip("Display tau images")
        self.modeButtons["tau"].clicked.connect(self.change_tau_mode)
        
        strForm=self.drawModePopup.current["id"]
        self.drawFormButton=QPushButton(strForm)
        w=self.drawFormButton.fontMetrics().boundingRect(strForm).width() + 5
        #self.modeButtons["currentForm"].setFixedWidth(w)
        self.drawFormButton.clicked.connect(self.show_drawmode_popup)
        
#==============================================================================
#         self.plotMeanButton=QPushButton("Plot mean")
#         w=self.plotMeanButton.fontMetrics().boundingRect().width() + 5
#         #self.modeButtons["currentForm"].setFixedWidth(w)
#         self.drawFormButton.clicked.connect(self.show_drawmode_popup)
#         
#==============================================================================
        self.buttons["add"]=QPushButton("Add")
        self.buttons["add"].setToolTip("Add the latest drawn form to the dataset")
        self.buttons["add"].setEnabled(False)
        self.buttons["add"].clicked.connect(self.add_form)
        
        self.buttons["currentEdit"]=QPushButton("Current edit")
        self.buttons["currentEdit"].setToolTip("Show current edit mode of img list")
        self.buttons["currentEdit"].clicked.connect(self.show_current_edit)
        
    def create_edit_toolbar_layout(self):
        """Create layout of edit toolbar"""
        if self.buttonMode == "icons":
            self.editLayout=QVBoxLayout()
        else:
            self.editLayout=QHBoxLayout()
        self.editLayout.addWidget(self.modeButtons["editMode"])
        self.editLayout.addWidget(self.buttons["currentEdit"])
        self.editLayout.addWidget(self.modeButtons["optFlow"])
        self.editLayout.addWidget(self.modeButtons["tau"])
        #self.editLayout.addSpacing(15)
        self.editLayout.addWidget(self.modeButtons["drawMode"])
        self.editLayout.addWidget(self.drawFormButton)
        self.editLayout.addWidget(self.buttons["add"])
        self.editLayout.addStretch(1)
        
    def create_toolbar_layout(self):
        """Create layout for base toolbar"""
        self.hBoxTools=QHBoxLayout()
        
        self.hBoxTools.addWidget(self.buttons["prevIm"])     
        self.hBoxTools.addWidget(self.buttons["nextIm"])
        self.hBoxTools.addSpacing(30)
        self.hBoxTools.addWidget(self.editCurrentFileNum)
        self.hBoxTools.addWidget(self.buttons["goto"])
            
        self.hBoxTools.addWidget(self.buttons["zoomOut"])
        self.hBoxTools.addWidget(self.buttons["zoomIn"])
        self.hBoxTools.addWidget(self.buttons["fit"])
        self.hBoxTools.addStretch(1)
        
    def set_button_mode(self,modeStr):
        """Set button mode (i.e. classic buttons with text or icons)"""
        if modeStr in self.buttonModes:
            self.buttonMode=modeStr
            if modeStr == "icons":
                if not self.load_all_icons():
                    self.buttonMode="normal"
                    self.modeButtons["editMode"].setStyleSheet(\
                        "background-color: #FF4D4D")
                    self.modeButtons["drawMode"].setStyleSheet(\
                        "background-color: #FF4D4D")
                else:
                    self.init_icons()
                    
            
    def load_all_icons(self):
        """Load all icons for this widget"""
        allGood=1
        for key,val in self.menuIconNames.iteritems():
            try:
                self.buttonIcons[key]=QIcon(get_icon(val,"k"))
            except:
                allGood=0
                msg="Could not load icon " + val
                QMessageBox.warning(self,"Error",msg, QMessageBox.Ok)
        for key, val in self.modeButtonIconNames.iteritems():
            try:
                self.modeButtonIcons[True][key]=QIcon(\
                    get_icon(val,"g"))
                self.modeButtonIcons[False][key]=QIcon(\
                    get_icon(val,"r"))
            except:
                allGood=0
                msg="Could not load icon " + val 
                QMessageBox.warning(self,"Error",msg, QMessageBox.Ok)
        for key,val in self.drawIcons:
            try:
                self.drawIcons[key]=QIcon(get_icon(val,"k"))
            except:
                allGood=0
                msg="Could not load icon " + val
                QMessageBox.warning(self,"Error",msg, QMessageBox.Ok)

        return allGood
        
    def init_icons(self):
        """Initiate all icons"""
        for key,val in self.buttonIcons.iteritems():
            if key in self.buttons.keys():
                try:
                    self.buttons[key].setFlat(True)
                    self.buttons[key].setIcon(val)
                    self.buttons[key].setIconSize(self.iconSize)
                    self.buttons[key].setText('')
                except:
                    msg="Could not load icon " + val + ", color: " + key
                    QMessageBox.warning(self,"Error",msg, QMessageBox.Ok)
        palette = QPalette()
        palette.setColor(palette.Background, Qt.transparent)
        for key, val in self.modeButtonIcons[False].iteritems():
            if key in self.modeButtons.keys():
                self.modeButtons[key].setFlat(True)
                self.modeButtons[key].setPalette(palette)
                self.modeButtons[key].setIcon(val)
                self.modeButtons[key].setIconSize(self.iconSize)
                self.modeButtons[key].setText('')
                
        self.drawFormButton.setFlat(True)
        self.drawFormButton.setIcon(self.drawModePopup.current["icon"])
        self.drawFormButton.setIconSize(self.iconSize)
        self.drawFormButton.setText("")
        
                
    def set_list(self, lst):
        """Set image list
        
        :param ImgList lst: the image list to be displayed        
        """
        if not isinstance(lst, BaseImgList):
            return
        print "Setting list %s in ImgViewer\n" %lst.id
#==============================================================================
#         if lst.bgModel is None or not lst.bgModel.ready_2_go():
#             self.modeButtons["tau"].setEnabled(False)
#==============================================================================
        self.imgList = lst
        lst.load()#lst.reload_current()
        if isinstance(lst.current_img(), Img):
            self.histoCanvas.init_histogram(self.imgList.loadedImages["this"])
        self.editCurrentFileNum.setText(str(self.imgList.currentFileNum+1))
        self.check_state()
    
    def check_state(self):
        """Check current state of image list"""
        print ("Checking state of image list:\nTauMode: " + str(self.imgList.tauMode)
         + "\nOptFlowMode: " + str(self.imgList.optFlowEdit.active))
        self.update_tau_mode(self.imgList.tauMode)
        self.update_optflow_mode(self.imgList.optFlowEdit.active)
            
    def connect_img_viewer(self, widget):
        """Connect another image to this one 
        
        :param ImgViewer widget: image viewer to be linked
        
        The displayed image in the connected image viewer is then updated 
        whenever the index is changed here.
        
        """
        id=widget.imgList.id
        if id not in self.imgList.linkedLists.keys():
            self.imgList.link_imglist(widget.imgList)
        if id in self.connectedWidgets.keys():
            id=id+str(len(self.connectedWidgets.keys()))
        self.connectedWidgets[id]=widget
    
    def disconnect_img_viewer(self,widget):
        """Disconnect a connected ImgViewer"""
        id=widget.imgList.id
        self.imgList.disconnect_linked_imglist(id)
        del self.connectedWidgets[id]
        
    def connect_to_img_label(self):
        """Connect this viewer to image label"""
        self.imgLabel.formChanged.connect(self.activate_add_button)
#==============================================================================
#         self.drawFormsCombo.addItems(self.imgLabel.drawForms)
#         self.drawFormsCombo.activated.connect(self.update_drawform)  
#         self.update_drawform()
#==============================================================================
        
    def show_current_edit(self):
        dial=DispImagePrepSettings(self.imgList.imgPrep, parent = self)
        dial.exec_()
    
    def show_drawmode_popup(self):
        globalP = self.mapToGlobal(self.sender().pos())
        relP=self.sender().rect().topRight()
        self.drawModePopup.move(globalP+relP)
        self.drawModePopup.show()
        
    def add_form(self):
        """Add current form to image label"""
        self.imgLabel.add_current_form()
        
    def disp_list(self):
        """If the imgList object contains files, load the one corresponding to the 
        current file number (see ImgList object) and displays it.
        """
        if self.imgList.files:
            #flag as active
            self.active = True 
            self.disp_image("this") #the currentImg variable will be set here 
            self.write_header()
            self.enable_buttons(True)
            if self.imgList.numberOfFiles < 2:
                self.buttons["nextIm"].setEnabled(False)
                self.buttons["prevIm"].setEnabled(False)
                self.buttons["goto"].setEnabled(False)
        else:
            print "could not display image list, no self.files is empty"
                         
    def enable_buttons(self, Bool):
        """Enable or disable the main buttons of this widget"""
        excludeKeys=["add"]
        for key, button in self.buttons.iteritems():
            if not key in excludeKeys:
                button.setEnabled(Bool)
        
        for key, button in self.modeButtons.iteritems():
            if not key in excludeKeys:
                button.setEnabled(Bool)

    def create_cmap(self,iMin8Bit,iMax8Bit):            
        return [QColor(i,i,i).rgb() for i in range(iMin8Bit,iMax8Bit)]
    
    def disp_image(self, key):
        """Display image
        
        If image (:mod:`piscope.Image.Img` object) is available in imgList, 
        convert the image data into QImage object using the variables lowI and 
        highI of the image object for upper and lower brightness limits of the 
        display (and of the histogram) and print the image on the ImageLabel.
        Write header of display and set currentDispIm variable (which can be 
        used for online editing, e.g. changing the displayed brightness range).
        
        :param key: choose, from "prev","this","next"
        
        .. todo::
        
            The conversion to 8 bit (done in toImage() is the current bottleneck
             
        """
        t0 = time()
        if self.imgList.loadedImages[key] is not None:            
            self.draw_histogram(key)
            self.currentDispIm = self.imgList.loadedImages[key]
            self.set_disp_im(self.histoCanvas.lowI, self.histoCanvas.highI)
#==============================================================================
#             lowI=im._to_8bit_int_pix(im.lowI)
#             highI=im._to_8bit_int_pix(im.highI)
#==============================================================================
            #self.currentHistogram=self.loadedHistograms[key]
        else:
            print "Could not display image, no data available for key: " + key
        
        print "DISP_IMAGE " + key + " , time elapsed: " + str(time()-t0)
        
    def set_disp_im(self, cmin = None, cmax = None):
        """Set the current image display in ImgLabel
        
        :param float cmin (None): minimum intensity for displayed contrast range
            (if None, use minimum brightness in image)
        :param float cmax (None): maximum intensity for displayed contrast range
            (if None, use maximum brightness in image)
        
        Steps:
        
            1. Set the current contrast (using cmin, cmax)
            2. Convert numpy array into PIL image (using toimage() func)
            3. Convert this PIL image into a Qt image (using ImageQt() func)
            4. Update and print this in ``self.imgLabel``
            
        """
        if self.currentDispIm is not None:
            if None in [cmin, cmax]:
                cmin = self.currentDispIm.img.min()
                cmax = self.currentDispIm.img.max()
            #t0=time()
            im = self.currentDispIm
            pilIm = toimage(im.img, cmin = cmin,cmax = cmax)
            #print "Elapsed time (toimage) (s): " + str(time()-t0)
            self.imgLabel.currentIm = ImageQt(pilIm)
            #print "Elapsed time (to QImage) (s): " + str(time()-t0)
            self.imgLabel.print_img()
        
        
    def invert_pixels(self):
        """Negative image inverting pixels"""
        for key in self.loadedQImages:
            if self.loadedQImages[key] is not None:
                self.loadedQImages[key].invertPixels()
        self.disp_image("this")
        
    def draw_histogram(self,key):
        """Draw the histogram of the current image in the histoCanvas"""   
        #print "NOT drawing HISTOGRAM right now"
        self.histoCanvas.draw_histogram(self.imgList.loadedImages[key])

    """Resizing, zooming, etc
    """
    def adjust_scroll_bar(self, scrollBar, factor):
        """Adjust the current scrollbar"""
        scrollBar.setValue(int(factor * scrollBar.value()
                                + ((factor - 1) * scrollBar.pageStep()/2)))
    
    def zoom(self,fac):                            
        self.imgLabel.scaleFactor=self.imgLabel.scaleFactor*fac
        self.imgLabel.print_img()
        self.adjust_scroll_bar(self.scrollArea.horizontalScrollBar(), fac)
        self.adjust_scroll_bar(self.scrollArea.verticalScrollBar(), fac)
        
    def zoom_in(self):
#==============================================================================
#         print self.imgLabel.pixmap().rect()
#==============================================================================
        fac = (1.0 + self.zoomFac)
        self.zoom(fac)        
    
    def zoom_out(self):
        fac = (1.0 - self.zoomFac)
        self.zoom(fac)        
    
    def fit_to_screen(self):
        self.imgLabel.my_resize(self.scrollArea.size())
    
    def write_header(self):
        self.subHeaderLabel.setText(self.imgList.make_header())
    """Button handles definitions
    """
    def goto_num(self):
        num = int(self.editCurrentFileNum.text()) - 1
        if num < 0 or num > self.imgList.numberOfFiles - 1:
            QMessageBox.warning(self, "Error",\
                "Chosen number out of range, retry...", QMessageBox.Ok)
            return
        self.imgList.goto_im(num)
        self.write_header()
#==============================================================================
#         self.selectButton.setChecked(bool(
#             self.imgList.selectedFilesIndex[self.imgList.currentFileNum]))
#==============================================================================
        #self.create_8bit_images()
        self.disp_image("this")
        self.imgLabel.update_flow(self.imgList.optFlowEdit.flowLinesInt)
        for key, widget in self.connectedWidgets.iteritems():
            widget.disp_image("this")
            widget.write_header()
            widget.imgLabel.update_flow(widget.imgList.optFlowEdit.flowLinesInt)
                    
    def prev(self):
        self.disp_image("prev")
        self.imgList.prev_im()
        self.write_header()
#==============================================================================
#         self.selectButton.setChecked(bool(self.imgList.selectedFilesIndex[
#             self.imgList.currentFileNum]))
#==============================================================================
        self.imgLabel.update_flow(self.imgList.optFlowEdit.flowLinesInt)        
        for key, widget in self.connectedWidgets.iteritems():
            widget.disp_image("this")
            widget.write_header()
            widget.imgLabel.update_flow(widget.imgList.optFlowEdit.flowLinesInt)
        #self.create_8bit_images()
        
        
    def next(self):
        """First, display the next image which should already be loaded in 
        `self.loadedImages` and then reload the images in the `imgList` object 
        calling the function :func:`next_im` in the list. 
        """
        self.disp_image("next")
        self.imgList.next_im()
        self.write_header()
        self.imgLabel.update_flow(self.imgList.optFlowEdit.flowLinesInt)
        for key, widget in self.connectedWidgets.iteritems():
            widget.disp_image("this")
            widget.write_header()
            widget.imgLabel.update_flow(widget.imgList.optFlowEdit.flowLinesInt)
            
        #self.create_8bit_images()
        
#==============================================================================
#     def handle_select_button(self):
#         if self.selectButton.isChecked():
#             self.imgList.selectedFilesIndex[self.imgList.currentFileNum]=1
#         else:
#             self.imgList.selectedFilesIndex[self.imgList.currentFileNum]=0
#         num=str(int(sum(self.imgList.selectedFilesIndex)))
#         self.selectInfoNum.setText(num)
#         if num>0:
#             self.reloadSelectionButton.setEnabled(True)
#         else:
#             self.reloadSelectionButton.setEnabled(False)
#==============================================================================
#==============================================================================
#     
#     def handle_reload_button(self):
#         if any(a for a in self.imgList.selectedFilesIndex):
#             text=("List will be reloaded, only selected files will prevail.\n "
#                 "Please confirm")
#             answer=QMessageBox.information(self,"Info",\
#                 text, QMessageBox.Ok,\
#                 QMessageBox.Cancel)
#             if answer == QMessageBox.Ok:
#                 """emitting the reload signal, such that the current selection
#                 can be loaded in another tab
#                 """
#                 self.reloadList.emit()
#                 self.reload_list(self.imgList.selectedFilesIndex)
#                 self.reloadSelectionButton.setEnabled(False)
#             else:
#                 print "Cancel button was pushed, no reload"
#         else:
#             print "Reload not possible, no images are selected"
#==============================================================================
            
    def change_activemode(self):
        newMode= not self.editActive
        self.editActive = newMode
        if self.buttonMode == "icons":
            icon=self.modeButtonIcons[newMode]["editMode"]
            self.modeButtons["editMode"].setIcon(icon)
        else:
            self.modeButtons["editMode"].setStyleSheet(self.modeButtonStyles[newMode])
            
    def change_drawmode(self):
        newMode = not self.imgLabel.drawMode
        self.imgLabel.drawMode=newMode
        if self.buttonMode == "icons":
            icon=self.modeButtonIcons[newMode]["drawMode"]
            self.modeButtons["drawMode"].setIcon(icon)
        else:
            self.modeButtons["drawMode"].setStyleSheet(self.modeButtonStyles[newMode])
    
    def change_tau_mode(self):
        newMode = not self.imgList.tauMode
        try:
            self.update_tau_mode(newMode)
        except Exception as e:
            msg = ("Error changing tau mode: %s" %repr(e))
            QMessageBox.warning(self,"Error", msg, QMessageBox.Ok)
    
    def update_tau_mode(self, newMode):
        
        self.imgList.activate_tau_mode(newMode)
        if self.buttonMode == "icons":
            self.modeButtons["tau"].setIcon(self.modeButtonIcons[newMode]["tau"])
        else:
            self.modeButtons["tau"].setStyleSheet(self.modeButtonStyles[newMode])                
        self.histoCanvas.init_histogram(self.imgList.current_img())
        
        print "Histocanvas autoupdate: " + str(self.histoCanvas.autoUpdate)
        print "Histocanvas fixRange: " + str(self.histoCanvas.fixRangeMode)
        self.disp_image("this")
        print "Tau mode changed: "
        print "New mode: " + str(newMode)
        
    def change_optflow_mode(self):
        newMode = not self.imgList.optFlowEdit.active
        self.update_optflow_mode(newMode)
        
    def update_optflow_mode(self,newMode):
        """Update optical flow mode of viewer
        
        """
        if self.buttonMode == "icons":
            self.modeButtons["optFlow"].setIcon(self.modeButtonIcons[newMode]["optFlow"])
        else:
            self.modeButtons["optFlow"].setStyleSheet(self.modeButtonStyles[newMode])        
        self.imgList.optFlowEdit.active = newMode
        self.imgList.set_flow_images()
        self.imgList.optFlowEdit.calc_flow()
        self.imgList.optFlowEdit.calc_flow_lines()
        self.imgLabel.optFlowOverlay = newMode
        #if newMode:
        self.imgLabel.update_flow(self.imgList.optFlowEdit.flowLinesInt)
        print "Newmode: " + str(newMode)
        
    def activate_add_button(self):
        """Activate the button to add a (just) drawn form"""
        if self.imgLabel.currentForm:
            self.buttons["add"].setEnabled(True)
        else:
            self.buttons["add"].setEnabled(False)
        
    
    """Low level code
    """
#==============================================================================
#     def init_edit(self):
#         """Reset image edit"""
#         self.imgList.init_edit()
#         self.disp_image("this")
#         
#==============================================================================
    def reload_list(self, boolList):
        newList=[d for (d, remove) in\
            zip(self.imgList.files, boolList) if remove]
        self.imgList.import_filelist(newList)
        self.imgList.load_images()
        self.selectInfoNum.setText("0")
        self.disp_list()
        
    def update_drawmode(self):
        info=self.drawModePopup.current
        if self.buttonMode == "icons":
            self.drawFormButton.setIcon(info["icon"])
        else:
            self.drawFormButton.setText(info["id"])
        self.imgLabel.drawForm=info["id"]
        print "Changed drawForm to ", self.imgLabel.drawForm
        
    def _write_mouse_position(self):
        if self.active and self.imgList.loadedImages["this"].img.ndim==2:
            x, y = self.imgLabel.map(self.imgLabel.mousePosX,\
                                                self.imgLabel.mousePosY)
            z = self.formatIntensity[self.imgList.tauMode](self.imgList.\
                                            loadedImages["this"].img[y, x])
            text="(" + str(x) + "|" + str(y) + "): I=" + z
            self.mousePosition.setText(text)

#==============================================================================
#     def mousePressEvent(self,event):
#         if event.button()==Qt.LeftButton:
#             self.drawOrigin = QPoint(event.pos())
#             print "Button pressed at: " + str(self.drawOrigin)
# #==============================================================================
# #             self.rubberBand.setGeometry(QRect(self.drawOrigin, QSize()))
# #             self.rubberBand.show()
# #==============================================================================
#==============================================================================
     
class ImgLabel(QLabel, QObject):
    moved = pyqtSignal()
    formChanged=pyqtSignal()
    newForm=pyqtSignal()
    drawMeanEvolution=pyqtSignal()
    def __init__(self,width=600,parent=None):
        super(ImgLabel, self).__init__(parent)
        self.resize(width,width)
#==============================================================================
#         sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         sizePolicy.setHeightForWidth(True)
#         self.setSizePolicy(sizePolicy)
#==============================================================================
        #the main dataset, where drawn forms will be written to
        #call self.set_dataSet to allocate
        self.dataSet=None     
        self.currentIm=None
        self.scaleFactor=0.3
        self.pixmap=QPixmap(256,256) #will be set in set_image
        self.pixmapScaled=QPixmap(256,256)
        
        """Paint an mouse move variables
        """
        self.drawMode=False
        self.drawForms=["rect","line"]
        self.drawForm="line"
        self.forms=Bunch({"lines"     :   LineCollection(),
                          "rects"     :   RectCollection()})
        self.rectsTemp={}
        
        self.actions={}
                          
        self.currentForm = None
        self.showAllForms = 1
        
        self.painter = QPainter()
        self.pens = dict(lime    =   QPen(QColor(0, 255 , 0)),
                         red     =   QPen(QColor(255, 0 , 0)),
                         yellow  =   QPen(QColor(255, 255 , 0)),
                         aqua    =   QPen(QColor(0, 255 , 255)),
                         blue    =   QPen(QColor(0, 0 , 255)))
                        
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.drawLine = QLine()
        #self.rubberBand=QRubberBand(QRubberBand.Line,self)
        self.drawOrigin=QPoint()
        self.drawLine=QLine(self.drawOrigin,QPoint())
        
        self.setMouseTracking(True)
        self.mousePosX=0
        self.mousePosY=0
        
        self.optFlowOverlay=False
        self.currentFlowLines=None 
        self.currentFlowOffset=[0,0] #offset if optical flow was calculated in ROI
        self.optFlowOnlySignificant=None    
    
    def add_current_form(self):
        """Add the currently drawn form (displayed in red) to the form coll"""
        print self.currentForm
        if isinstance(self.currentForm, QRect):
            tl = self.currentForm.topLeft()
            br = self.currentForm.bottomRight()
            x0, y0 = self.map(tl.x(),tl.y())
            x1, y1 = self.map(br.x(),br.y())
            #if self.dataSet:
            self.forms["rects"].add(x0, y0, x1, y1)
                
        elif isinstance(self.currentForm,QLine):
            l = self.currentForm
            x0, y0 = self.map(l.x1(),l.y1())
            x1, y1 = self.map(l.x2(),l.y2())
            self.forms["lines"].add(x0, y0, x1, y1)
        self.newForm.emit()
        

    def print_img(self):
        #print "Printing image on ImgLabel"
        #t0=time()
        try:
            #print str(type(self.currentIm))
            self.pixmap=QPixmap.fromImage(self.currentIm) #the actual image in original size
            #print "Elapsed time convert image (s): " + str(time()-t0)
            size=self.pixmap.size()*self.scaleFactor
            #print "Scale fact: ", self.scaleFactor
            self.setFixedSize(size)
            #print "New label size: " , self.size()
            #the scaled pixmap according to current zoom factor (self.scaleFactor)
            self.pixmapScaled=self.pixmap.scaled(size,Qt.KeepAspectRatio)
            #print "Elapsed time scale pixmap (s): " + str(time()-t0)
    
            self.setPixmap(self.pixmapScaled) 
            #print "Elapsed time set pixmap (s): " + str(time()-t0)
            self.currentForm=None
            self.formChanged.emit()
            self.print_all_forms()
            self.draw_optical_flow(self.optFlowOverlay)
            #print "Elapsed time print all forms (s): " + str(time()-t0)
            #print "Elapsed time total (s): " + str(time()-t0)
        except:
            pass

    def update_flow(self, flowLines):
        try:
            self.currentFlowLines = flowLines[0]
            self.currentFlowOffset = [flowLines[1], flowLines[2]]
        except:
            self.currentFlowLines=None
            self.currentFlowOffset=[0,0]
        self.print_img()
        
    def my_resize(self,QSize):
        #print "myResize"
        self.pixmapScaled=self.pixmap.scaled(QSize,Qt.KeepAspectRatio)
        self.setFixedSize(self.pixmapScaled.size())
        self.scaleFactor=self.width()/float(self.pixmap.width())
        self.print_all_forms()
        self.draw_optical_flow(self.optFlowOverlay)
        self.setPixmap(self.pixmapScaled)
        
#==============================================================================
#         print "pixmap size: ", self.pixmap.size()
#         print "pixmapScaled size:", self.pixmapScaled.size()
#         print "label size:", self.size()
#         print "Scale factor: ", self.scaleFactor
#==============================================================================
    
    def print_all_forms(self):
        """Draw all forms on current image pixmap"""
        if self.showAllForms:
            self.painter.begin(self.pixmapScaled)
            self.painter.setPen(self.pens["lime"])
            self.rectsTemp = {}
            for rId, r in self.forms["rects"].forms.iteritems():
                if not rId == "whole_img":
                    tlx, tly = self.map(r[0], r[1], inverse = 1)
                    brx, bry = self.map(r[2], r[3], inverse = 1)
                    p1, p2 = QPoint(tlx, tly), QPoint(brx, bry)
                    self.rectsTemp[rId] = rect = QRect(p1, p2)
                    self.painter.drawRect(rect)
            for l in self.forms["lines"].forms.values():
                ix, iy = self.map(l[0], l[1], inverse = 1)
                fx, fy = self.map(l[2], l[3], inverse = 1)
                p1, p2 = QPoint(ix, iy), QPoint(fx, fy)
                self.painter.drawLine(p1, p2)
            self.painter.end()
            self.setPixmap(self.pixmapScaled)
                    
    def map(self, x, y, inverse = 0):
        """Map coordinate on pixmap into original image coordinates"""
        if not inverse:
            x, y = int(round(x/self.scaleFactor)), int(round(y/self.scaleFactor))
        else:
            x, y=int(round(x*self.scaleFactor)), int(round(y*self.scaleFactor))
        return x, y
        
    def draw_form(self,event):    
        """Draw one form on the image pixmap"""
        self.painter.begin(self.pixmapScaled)
        self.painter.setPen(self.pens["red"])
        if self.drawForm == "rect":
            self.currentForm = QRect(self.drawOrigin, event.pos()).normalized()
            self.painter.drawRect(self.currentForm)
        elif self.drawForm == "line":
            self.currentForm = QLine(self.drawOrigin, event.pos())
            self.painter.drawLine(self.currentForm)
        #self.draw_arrow()
        self.painter.end()
        print "EMITTED FORM CHANGED IN IMGLABEL"
        self.formChanged.emit()
        
    def draw_optical_flow(self,bool):
        """Draw optical flow overlay
            
            (x1,y1),(x2,y2) in lines:
            line(vis,(x1+dx,y1+dy),(x2+dx,y2+dy),(0,255,255),1)
            circle(vis,(x2+dx,y2+dy),1,(255,0,0), -1)
            
        """
        if bool and self.currentFlowLines is not None:
            self.painter.begin(self.pixmapScaled)
            self.painter.setPen(self.pens["aqua"])
            for (x1,y1),(x2,y2) in self.currentFlowLines:
                dx, dy = self.currentFlowOffset
    #==============================================================================
    #             lineLen=sqrt((x2-x1)**2+(y2-y1)**2)
    #             if lineLen > 0:
    #                 sigLineCounter+=1
    #                 print str(sigLineCounter)
    #                 print "Length in pix: " + str(lineLen)
    #                 print "x, y: " + str(x1) + ", " + str(y1)
    #                 print "Scale factor: " + str(self.scaleFactor)
    #                 print self.map(x1,y1,inverse=1)
    #                 if lineLen > lmax[0]:
    #                     lmax=[lineLen,x1,y1,x2,y2]
    #==============================================================================
                x1,y1=self.map(x1+dx,y1+dy,inverse=1)            
                x2,y2=self.map(x2+dx,y2+dy,inverse=1)
                l=QLine(QPoint(x1,y1),QPoint(x2,y2))
                if not l.isNull():
                    self.painter.drawLine(l)
                    self.painter.setPen(self.pens["red"])
                    self.painter.drawEllipse(QPoint(x2,y2), 1,1)
                    self.painter.setPen(self.pens["aqua"])
            self.painter.end()
            self.setPixmap(self.pixmapScaled)
        
    
    def draw_arrow(self):
        """Draw an arrow
        
        .. note::
        
            written for test purposes, currently not in use
            
        """
        from random import randint
        angle=randint(0,360)
        a=angle*pi/180
        l=30
        x=self.drawOrigin.x()
        y=self.drawOrigin.y()
        delx=l*sin(a)
        dely=l*cos(a)
        x1=x+delx
        y1=y+dely
        ps=QPoint(x1,y1)
        print "Angle: " + str(angle)
        print "p0: " + str(self.drawOrigin)
        print "p1: " + str(ps)
        
        self.painter.drawLine(QLine(self.drawOrigin,ps))
        
    def mousePressEvent(self,event):
        """What to do when the mouse is pressed on this object"""
        if event.button()==Qt.LeftButton:
            print "activate paint mode"
            self.drawOrigin = QPoint(event.pos())
            print self.drawOrigin
            self.rubberBand.setGeometry(QRect(self.drawOrigin, QSize()))
            self.rubberBand.show()
        if event.button()==Qt.RightButton:
            print "Right mouse click detected.."
            for key, val in self.rectsTemp.iteritems():
                if val.contains(event.pos()):
                    print "On rect: ", key
            
    def mouseMoveEvent(self, event):
        """Define what's supposed to happen whenever a mouse move is detected"""
        self.mousePosX=event.x()
        self.mousePosY=event.y()
        self.moved.emit()
        if not self.drawOrigin.isNull():
            self.rubberBand.setGeometry(QRect(self.drawOrigin, event.pos()).normalized())
        
#==============================================================================
#         elif event.button() == Qt.LeftButton:
#             print "Left click drag"
#         elif event.button() == Qt.RightButton:
#             print "Right click drag"
#==============================================================================
        
    def mouseReleaseEvent(self,event):
        """Define what's supposed to happen whenever a mouse button is released"""
        self.rubberBand.hide()
        self.print_img()
        if event.button() == Qt.LeftButton and self.drawMode:            
            self.draw_form(event)
            self.setPixmap(self.pixmapScaled)
        elif event.button() == Qt.LeftButton and not self.drawMode:
            print "I WOULD REALLY LIKE TO HAVE A ZOOM IN HERE IN THIS SPECIFIC RECTANGLE"
            