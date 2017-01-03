# -*- coding: utf-8 -*-
from PyQt4.QtGui import QWidget, QGroupBox, QPushButton, QVBoxLayout,\
    QHBoxLayout, QToolBar, QAction, QMessageBox, QInputDialog, QDialog, QIcon,\
    QFormLayout, QLabel, QSpinBox, QComboBox, QLineEdit, QCheckBox
from PyQt4.QtCore import Qt, pyqtSignal
from collections import OrderedDict as od
from traceback import format_exc
#from matplotlib.pyplot import hist

from ..inout import get_icon
from ..plumespeed import OpticalFlowFarnebackSettings

from .plot_widgets import FigureDispWidget

class DrawMenuWidget(QWidget):
    """Widget for managing draw form collections 
    
    By default, :class:`piscope.Datasets.Dataset` objects include a buncdict) 
    of different draw form collections (e.g. 
    :class:`piscope.Forms.RectCollection`, :class:`piscope.Forms.LineCollection`)
    
    These are managed within this widget which provides basic editing such as
    renaming or deleting or assigning them to default forms which are used for
    certain processes such as background image modelling.
    
    """
    def __init__(self, drawWidgetCollection, parent = None):
        super(DrawMenuWidget,self).__init__(parent)
        self.setWindowTitle("piSCOPE: Edit forms")
        self.setWindowModality(Qt.ApplicationModal)
        
        self.objectWidgets = drawWidgetCollection
        
        self.layout = QVBoxLayout()
        
        hBoxButtons = QHBoxLayout()
        
        self.closeButton = QPushButton("Close")
        self.closeButton.clicked.connect(self.close)
        
        hBoxButtons.addStretch(1)
        hBoxButtons.addWidget(self.closeButton)
        
        for widget in self.objectWidgets.values():
            group = QGroupBox(str(widget.formColl.type))
            group.setLayout(widget.layout)
            self.layout.addWidget(group)
            #self.layout.addWidget(widget)
            #self.layout.addWidget(QLabel())
            
        self.layout.addLayout(hBoxButtons)
        
        self.setLayout(self.layout)
        
    def update_forms(self):
        """update forms collection"""
        for key,widget in self.objectWidgets.iteritems():
            print "Updating forms in widget " + key
            widget.update_forms_combo()
            widget.fill_set_as_combo()
        
    def closeEvent(self,e):
        """Hide widget on close"""        
        self.hide()        
        
class MainEditManagement(QWidget):
    """Central object for management of image editing operations 
        
    This widget creates all necessary actions for image editing and is rather 
    a collection of actions and connected methods and in order 
    to work, it needs to be embedded within a QWidget, QApplication etc (the
    actions need to be assigned to a QObject, which must be the parent of this
    widget)
    """

    def __init__(self, parent):
        """Init class
        
        :param parent: parent widget (:class:`piscope.gui.MainApp.MainApp`) 
        
        """
        super(MainEditManagement,self).__init__(parent)
        
        self.iconNames = od([("Correct dark"     ,   "myDarkCorr"),
                             ("Add blurring"     ,   "myBlurring"),
                             ("Optical flow"     ,   "myOptFlow"),
                             ("Reset all"        ,   "Refresh"),
                             ("Disp widgets"     ,   "Info2")])
        
        self.icons = od()
                 
        self.parent = parent
        self.actions = od()
        
        self.toolbar = None
        
        self.widgetsCounter = 0              
        self.widgets = od()
        
        self.blurMenu = BlurMenuWidget()
        self.blurMenu.gaussianBlurringButton.clicked.connect(self.add_blurring)
        
        #this is the central optical flow object for editing etc.
        self.opticalFlowColl = {}
        #create dialog to edit the previously created flow object
        self.optFlowDialog = OpticalFlowDialog(self.opticalFlowColl,\
                                                            parent = self)
        self.optFlowDialog.update.connect(self.redraw_flow)
        
        self.create_actions()
        self.create_toolbar()
        try:
            self.load_icons()
        except WindowsError as e:
            print "Windows sucks..." + repr(e)
        
    def create_actions(self):
        """Create all image edit actions"""                
        self.actions["Correct dark"] = QAction("&Correct dark...", self.parent,\
            triggered=self.correct_dark)
        self.actions["Add blurring"] = QAction("&Add gaussian blurring...",\
            self.parent, triggered = self.blurMenu.show)
        self.actions["Optical flow"] = QAction("&Optical flow...",\
            self.parent, triggered = self.show_optflow_dial)
        self.actions["Reset all"] = QAction("&Reset all...", self.parent,\
            triggered = self.reset_all)
        self.actions["Disp widgets"] = QAction("&Connection info...", self.parent,\
            triggered = self.show_info)
        self.write_tooltips()
    
    def create_toolbar(self):
        """Create :class:`QToolBar` and add all actions in ``self.actions``"""
        tb = QToolBar()
        tb.setMovable(True)
        tb.setOrientation(Qt.Horizontal)
        for action in self.actions.values():
            tb.addAction(action)
        self.toolbar = tb
        
    def write_tooltips(self):
        """Add tooltips to all actions in ``self.actions``"""
        self.actions["Correct dark"].setToolTip("Subtract dark frame from all"
            " active image viewers")
        self.actions["Add blurring"].setToolTip("Add gaussian blurring to all"
            " active image viewers")
        self.actions["Reset all"].setToolTip("Reset currentEdit dictionary in"
        " all image viewers which are curretnly connected")
        self.actions["Disp widgets"].setToolTip("Show info about all connected"
        " widgets")
    
    def add_blurring(self):
        """Add blurring to all active widgets
        
        Reads the value (width of gaussian blurring kernel) from 
        ``self.blurMenu.sigmaAdd`` and applies this (!) additional blurring
        to all active image viewers
        
        """
        for widget in self.widgets.values():
            if widget and widget.editActive:
                sigmaAdd = int(self.blurMenu.sigmaEdit.text())
                widget.imgList.add_gaussian_blurring(sigmaAdd)
                widget.disp_image("this")
        self.blurMenu.close()    
    
    def correct_dark(self):
        """Activate dark correction in all active image viewers"""
        for widget in self.widgets.values():
            if widget and widget.editActive:
                widget.imgList.activate_dark_corr()
                #widget.imgList.apply_current_edit()
                widget.disp_image("this")      
    
    def show_optflow_dial(self):
        """Show optical flow dialog"""
        #self.optFlowDialog.fill_widget_combo()
        self.optFlowDialog.show()
        self.optFlowDialog.raise_()
                
    def redraw_flow(self):
        """Redraw optical flow
        
        Recalculate the optical flow in the current imgviewer and display reload
        the results
        """
        id = self.optFlowDialog.currentEditId
        for widget in self.widgets.values():
            if widget.id == id:
                widget.imgLabel.update_flow(\
                    widget.imgList.optFlowEdit.flowLinesInt) 
        
    def reset_all(self):
        """Reset image edit in all active widgets"""
        for key in self.widgets:
            widget = self.widgets[key]
            if widget.active:
                self.widgets[key].imgList.reset_edit()
                self.widgets[key].disp_image("this")
    
    def show_info(self):
        """Display :class:`QMessageBox` with current connection info"""
        s = "All connected widgets in MainEditManagement:\n\n"
        if not self.widgets.keys():
            s = s + "No widgets detected, check connection"
        for key, widget in self.widgets.iteritems():
            sub1 = "Image viewer number:\t" + str(key) + "\n"
            sub2 = "Image box ID:\t" + str(widget.id) + "\n\n"
            s = s + sub1 + sub2
        QMessageBox.information(self.parent,"Info",s, QMessageBox.Ok)
        
    """
    Connect to other widgets / objects functions
    """    
    def delete_all_connections(self):
        """Remove all currently connected widgets"""
        keys=self.widgets.keys()
        for k in range(len(keys)):
            del self.widgets[keys[k]]
        keys=self.opticalFlowColl.keys()
        for key in keys:
            del self.opticalFlowColl[key]
            
    def connect_widget(self, widget):
        """Connect a widget to this editbar, 
        
        All image edit operations are controlled via this working environment. A speciality is the central
        optical flow object, which is connected to the imgLists in the img 
        display widgets. 
        """
        counter = len(self.widgets.keys())+1
        if not counter in self.widgets.keys():
            #connect to optical flow widget in image list
            self.widgets[counter] = widget
            if widget.imgList is not None:
                self.opticalFlowColl[widget.imgList.id] =\
                                            widget.imgList.optFlowEdit
        else:
            msg=("Widget could not be importet into editMenu, id %s " 
             "already exists in this application.\n\n"
            "Please enter another id:" %counter)
            text, ok = QInputDialog.getText(self, 'piSCOPE Error', msg)
            self.connect_widget(widget, int(text))
            
    def get_widget_num(self,id):
        """Get number of widget with input id"""
        for key,widget in self.widgets.iteritems():
            if widget.id == id:
                return key
        return None
        
    """
    Printing etc.
    """
    def print_all_connected_widgets(self):
        print 
        print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print "+++++++++ ALL CONNECTED WIDGETS IN EDIT MENU ++++++++++++++++++"
        print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print
        for key in self.widgets:
            print "ImViewer num: " + str(key)
            print "Image box ID: " + str(self.widgets[key].id)
    """
    Other stuff
    """
    
    def load_icons(self):
        for key, val in self.iconNames.iteritems():
            if not key in self.actions.keys():
                print "ERROR IN LOAD ICONS: COULD NOT LOAD ICONS"
                return 0
            else:
                #try:
                print "LOAD ICON: " + str(id)
                self.icons[key]=QIcon(get_icon(val,"k"))
                self.actions[key].setIcon(self.icons[key])
        return 1
        
class BlurMenuWidget(QDialog):
    """Small widget for editing blur settings"""
    def __init__(self, parent=None):
        super(BlurMenuWidget,self).__init__(parent)
        self.layout=QVBoxLayout()
        self.setWindowTitle("piSCOPE: Edit optical flow")
        groupGaussianBlurring=QGroupBox()
        groupGaussianBlurring.setTitle("Gaussian blurring")
        
        formGaussianBlurring=QFormLayout()
        
#==============================================================================
#         hBoxCurrentBlurring=QHBoxLayout()
#         hBoxCurrentBlurring.addWidget(QLabel("Current blurring: "))
#         self.labelCurrentBlurInfo=QLabel()
#         hBoxCurrentBlurring.addWidget(self.labelCurrentBlurInfo)        
#==============================================================================
        
        self.sigmaLabel=QLabel(u"\u03C3")
        self.sigmaLabel.setStyleSheet("font-size: 15px; font-weight: bold")
        self.sigmaLabel.setToolTip("Width of gaussian blurring filter")
        self.sigmaEdit=QSpinBox()
        self.sigmaEdit.setValue(2)
        
        self.gaussianBlurringButton=QPushButton("Blur")
        self.gaussianBlurringButton.setToolTip("Gaussian blurring")
        
        hBoxClose=QHBoxLayout()
        self.closeButton=QPushButton("Close")
        self.closeButton.clicked.connect(self.close)
        hBoxClose.addStretch(1)
        hBoxClose.addWidget(self.closeButton)
        
        formGaussianBlurring.addRow(self.sigmaLabel, self.sigmaEdit)
        formGaussianBlurring.addRow(QLabel(),self.gaussianBlurringButton)
    
        groupGaussianBlurring.setLayout(formGaussianBlurring)
        
        self.layout.addWidget(groupGaussianBlurring)
        self.layout.addLayout(hBoxClose)
        self.setLayout(self.layout)

class DispImagePrepSettings(QDialog):
    """Widget to visualise an :class:`pyscope.Utils.ImagePrepSettings` object
    """  
    def __init__(self, prepSettings, parent=None):
        """Init
        
        :param OpticalFlowFarnbackColl: collection of optical flow objects
        
        """
        super(DispImagePrepSettings,self).__init__(parent)
        self.layout=QVBoxLayout()
        self.setWindowTitle("piSCOPE: view image preparation settings")
        
        self.imgPrepSettings = prepSettings
        
        self.labels={}
        self.buttons={}
        
        self.create_buttons()
        self.create_layout()
        self.write_values()
    
    def create_buttons(self):
        """Create buttons for this widget"""
        self.buttons={}
        b0=QPushButton("Reset edit")
        b0.clicked.connect(self.reset)
        b0.setToolTip("Reset edit settings")
        self.buttons["reset"]=b0
        
        b1=QPushButton("Close")
        b1.clicked.connect(self.close)
        self.buttons["close"]=b1
        
    def create_layout(self):
        """Create the display layout"""
        settingsGroup=QGroupBox("Current settings")
        form=QFormLayout()
        self.labels={}
        keys=self.imgPrepSettings.make_settings_dict().keys()
        for key in keys:
            self.labels[key]=lbl=QLabel("")
            form.addRow(QLabel(key),lbl)
            
        hBoxReset=QHBoxLayout()
        hBoxReset.addStretch(1)
        hBoxReset.addWidget(self.buttons["reset"])
        hBoxReset.addWidget(self.buttons["close"])
        
        settingsGroup.setLayout(form)
        self.layout=QVBoxLayout()
        self.layout.addWidget(settingsGroup)        
        self.layout.addLayout(hBoxReset)
        #self.layout.addLayout(self.buttons)
        self.setLayout(self.layout)
    
    def write_values(self):
        d = self.imgPrepSettings.make_settings_dict()
        for k, v in self.labels.iteritems():
            if d.has_key(k):
                v.setText(str(d[k]))
                
    def reset(self):
        """reset settings
        """
        self.imgPrepSettings.reset_settings()
        self.write_values()
        try:
            self.parent().imgList.load()
            self.parent().disp_image("this")
        except:
            raise IOError(format_exc())
        
        
class OpticalFlowDialog(QDialog):
    """Dialog window for interactive edit of optical flow settings
    """
    update = pyqtSignal()
    changeHistoLink = pyqtSignal()
    def __init__(self, OpticalFlowFarnbackColl, parent=None):
        """Init
        
        :param OpticalFlowFarnbackColl: collection of optical flow objects
        
        """
        super(OpticalFlowDialog,self).__init__(parent)
        self.layout=QVBoxLayout()
        self.setWindowTitle("piSCOPE: Edit optical flow input")
        
        #an empty settings mask for creating the layout etc..
        self.settingsMask=OpticalFlowFarnebackSettings("","default")

        self.flowObjColl=OpticalFlowFarnbackColl
        self.currentEditId=None
        
        self.currentPreEditSettings=od()
        self.currentSettings=od()
        self.currentDrawSettings=od()
        self.currentHistoSettings=od()
        self.typeSettings=od()
        self.buttons={}
        
        
        self.linkedToCanvas={}
        self.connectedCanvas={}
        
        self.create_layout()
        
    def create_layout(self):
        self.layout=QVBoxLayout()

        self.selectWidgetCombo=QComboBox()
        self.selectWidgetCombo.activated[str].connect(self.reload_settings)  
        self.layout.addWidget(self.selectWidgetCombo)
        
        self.create_presettings_layout()
        self.create_settings_layout()
        self.create_display_layout()
        self.create_plotting_layout()

        hBoxShow=QHBoxLayout()
        
        self.buttons["showCurrent"]=QPushButton("Show current settings")
        self.buttons["showCurrent"].clicked.connect(self.show_current_settings)
        self.buttons["close"]=QPushButton("Close")
        self.buttons["close"].clicked.connect(self.close)
        hBoxShow.addStretch(1)
        hBoxShow.addWidget(self.buttons["showCurrent"])
        hBoxShow.addWidget(self.buttons["close"])
        self.layout.addLayout(hBoxShow)
        self.setLayout(self.layout)
    
    def create_presettings_layout(self):
        settingsLayout=QVBoxLayout()
        settingsGroup=QGroupBox()
        settingsGroup.setTitle("Edit image pre-settings")
        formSettings=QFormLayout()
        hBoxUpdateSettings=QHBoxLayout()
        for key in self.settingsMask.preEditSettings.keys():
            self.currentPreEditSettings[key]=QLineEdit()            
            formSettings.addRow(QLabel(key),self.currentPreEditSettings[key])
        self.roiCombo=QComboBox()
        self.roiCombo.activated[str].connect(self.set_roi)
        #formSettings.addRow(QLabel("Select ROI"), self.roiCombo)
        self.linkToCanvasBox=QCheckBox()
        self.linkToCanvasBox.setChecked(False)
        self.linkToCanvasBox.clicked.connect(self.toggle_canvas_link)
        self.linkToCanvasBox.setTristate(False)
        
        formSettings.addRow(QLabel("Link to Histo canvas"),self.linkToCanvasBox)
        
        self.buttons["updatePreSettings"]=QPushButton("Update")
        self.buttons["updatePreSettings"].setToolTip("Confirm and update "
            "settings for image preparation before running the flow algo")
        self.buttons["updatePreSettings"].clicked.connect(\
            self.handle_update_presettings_button)

        hBoxUpdateSettings.addStretch(1)
        hBoxUpdateSettings.addWidget(self.buttons["updatePreSettings"])
        
        settingsLayout.addLayout(formSettings)
        settingsLayout.addLayout(hBoxUpdateSettings)
        settingsGroup.setLayout(settingsLayout)
        
        self.layout.addWidget(settingsGroup)
        
    def create_settings_layout(self):
        settingsLayout=QVBoxLayout()
        formSettingsGroup=QGroupBox()
        formSettingsGroup.setTitle("Edit settings")
        formSettings=QFormLayout()
        hBoxUpdateSettings=QHBoxLayout()
        for key in self.settingsMask.flowSettings.keys():
            self.currentSettings[key]=QLineEdit()            
            formSettings.addRow(QLabel(key),self.currentSettings[key])
        
        self.buttons["update"]=QPushButton("Update")
        self.buttons["update"].setToolTip("Confirm and update current flow settings")
        self.buttons["update"].clicked.connect(self.update_flow_settings)

        hBoxUpdateSettings.addStretch(1)
        hBoxUpdateSettings.addWidget(self.buttons["update"])
        
        settingsLayout.addLayout(formSettings)
        settingsLayout.addLayout(hBoxUpdateSettings)
        formSettingsGroup.setLayout(settingsLayout)
        
        self.layout.addWidget(formSettingsGroup)
    
    def create_display_layout(self):
        drawSettingsLayout=QVBoxLayout()
        formDrawSettingsGroup=QGroupBox()
        formDrawSettingsGroup.setTitle("Flow display options")
        formDrawSettings=QFormLayout()
        hBoxUpdateDrawSettings=QHBoxLayout()
        for key in self.settingsMask.drawSettings.keys():
            self.currentDrawSettings[key]=QLineEdit()
            formDrawSettings.addRow(QLabel(key),self.currentDrawSettings[key])
        
        self.buttons["updateDraw"]=QPushButton("Update")
        self.buttons["updateDraw"].setToolTip("Confirm and update current "
            "settings for visualisation of optical flow")
        self.buttons["updateDraw"].clicked.connect(self.update_draw_settings)

        hBoxUpdateDrawSettings.addStretch(1)
        hBoxUpdateDrawSettings.addWidget(self.buttons["updateDraw"])
        
        drawSettingsLayout.addLayout(formDrawSettings)
        drawSettingsLayout.addLayout(hBoxUpdateDrawSettings)
        formDrawSettingsGroup.setLayout(drawSettingsLayout)
        
        self.layout.addWidget(formDrawSettingsGroup)
    
    def create_plotting_layout(self):
        plotSettingsLayout=QVBoxLayout()
        plotSettingsGroup=QGroupBox()
        plotSettingsGroup.setTitle("Flow analysis (Histograms)")
        plotSettingsForm=QFormLayout()
#==============================================================================
#         for key,val in self.settingsMask.dispHistoSettings.iteritems():
#             self.currentHistoSettings[key]=QLineEdit()
#             
#             plotSettingsForm.addRow(QLabel(key),self.currentHistoSettings[key])
#==============================================================================
        self.buttons["updateHisto"]=QPushButton("Update")
        self.buttons["updateHisto"].setToolTip("Not working currently")
        #self.buttons["updateHisto"].clicked.connect(self.update_histo_settings)
        self.buttons["updateHisto"].setEnabled(False)
        self.buttons["flowHistograms"]=QPushButton("Show Histograms")
        self.buttons["flowHistograms"].setToolTip("Show two histograms (length"
            " and angle distribution) of the current flow fiels")
        self.buttons["flowHistograms"].clicked.connect(self.show_flow_histos)
        self.buttons["flowHistograms"].setEnabled(False)
        
        plotSettingsForm.addRow(self.buttons["updateHisto"], self.buttons["flowHistograms"])
        plotSettingsLayout.addLayout(plotSettingsForm)
        plotSettingsGroup.setLayout(plotSettingsLayout)
        self.layout.addWidget(plotSettingsGroup)
        
    def load_settings_datatypes(self):
        try:
            settings=self.settingsMask
            for key,val in settings.flowSettings.iteritems():
                self.typeSettings[key]=type(val)
            for key,val in settings.drawSettings.iteritems():
                self.typeSettings[key]=type(val)
            for key, val in settings.preEditSettings.iteritems():
                self.typeSettings[key]=type(val)
#==============================================================================
#             for key, val in settings.dispHistoSettings.iteritems():
#                 self.typeSettings[key]=type(val)
#==============================================================================
        except:
             msg="Could not load optical flow settings datatypes"
             QMessageBox.warning(self,"Unknown Error",msg, QMessageBox.Ok)
    
    def init_canvas_link(self):
        self.linkedToCanvas={}
        for key in self.flowObjColl:
            self.linkedToCanvas[key]=False
    
    def set_widget_combo_id(self,id):
        index = self.selectWidgetCombo.findText(id, Qt.MatchFixedString)
        if index >= 0:
            self.selectWidgetCombo.setCurrentIndex(index)
            
    def fill_widget_combo(self):
        self.selectWidgetCombo.clear()
        keys=self.flowObjColl.keys()
        self.selectWidgetCombo.addItems(keys)
        if keys:
            self.currentEditId=keys[0]
            self.load_settings_datatypes()
            self.disp_myflow_settings(self.currentEditId)
    
    def fill_rect_combo(self):
        """fill ROI selection the combo with all available rectangles
        
        :param str id: id of current optical flow object
        """
        self.roiCombo.clear()
        rectIds=[]
        try:
            rectIds.extend(self.flowObjColl[self.currentEditId].rectsCollection.forms.keys())
        except:            
            print format_exc()
        self.roiCombo.addItems(rectIds)
        id=self.flowObjColl[self.currentEditId].currentRoiId
        idx= self.roiCombo.findText(id, Qt.MatchFixedString)
        if idx >= 0:
            self.roiCombo.setCurrentIndex(idx)
    
    def set_roi(self, text):
        """Change ROI for optical flow display"""
        print "Changing ROI for optical flow, new ID: " + str(text)
        if str(text) == "None":
            self.flowObjColl[self.currentEditId].change_roi()
        else:
            self.flowObjColl[self.currentEditId].change_roi(str(text))
        
    def reload_settings(self, text):
#==============================================================================
#         msg="Index changed, now displaying optical flow settings of " + str(text)
#         QMessageBox.information(self,"Info",msg, QMessageBox.Ok)
#==============================================================================
        self.currentEditId=str(text)
        self.disp_myflow_settings(self.currentEditId)
        self.set_widget_combo_id(self.currentEditId)
        self.buttons["flowHistograms"].setEnabled(self._flow_active())
        
    def disp_myflow_settings(self, id):
        self.currentEditId=id
        print "Display flow settings for image viewer " + str(id)
        settings=self.flowObjColl[id].settings
        for key,val in settings.flowSettings.iteritems():
            self.currentSettings[key].setText(str(val))
        for key,val in settings.drawSettings.iteritems():
            self.currentDrawSettings[key].setText(str(val))
        for key, val in settings.preEditSettings.iteritems():
            self.currentPreEditSettings[key].setText(str(val))
        #self.fill_rect_combo()
        
#==============================================================================
#         for key, val in settings.dispHistoSettings.iteritems():
#             self.currentHistoSettings[key].setText(str(val))
#==============================================================================
        self.linkToCanvasBox.setChecked(self.linkedToCanvas[self._id()])
    
    def handle_update_presettings_button(self):
        self.update_presettings()
        id=self.currentEditId
        if self.linkedToCanvas[self._id()]:
            x0=self.flowObjColl[id].settings.preEditSettings["iMin"]
            x1=self.flowObjColl[id].settings.preEditSettings["iMax"]
            self.connectedCanvas[self._id()].dragLines.\
                change_lines_position(x0,x1)
                
    def update_presettings(self):
        #create a backup of the last settings
        id=self.currentEditId
        print "Updating presettings in " + id
        for key, val in self.currentPreEditSettings.iteritems():
            print "Current key: " + str(key)
            print "Type: " + str(self.typeSettings[key])
            print "Value: " + val.text()
            self.flowObjColl[id].settings.preEditSettings[key]=\
                                            self.typeSettings[key](val.text())
        self.flowObjColl[id].prepare_flow_images()
        self.flowObjColl[id].calc_flow()
        self.flowObjColl[id].calc_flow_lines()

        self.update.emit()
            #self.flowObjColl.preEditSettings[key]=np.int(val.text())
            
    def update_flow_settings(self):
        #create a backup of the last settings
        id=self.currentEditId
        print "Updating flow settings in " + id
        for key, val in self.currentSettings.iteritems():
           self.flowObjColl[id].settings.flowSettings[key]=\
                                           self.typeSettings[key](val.text())
        self.flowObjColl[id].calc_flow()
        self.flowObjColl[id].calc_flow_lines()
        self.update.emit()    
        
    def update_draw_settings(self):
        id=self.currentEditId
        print "Updating draw settings in " + id
        for key, val in self.currentDrawSettings.iteritems():
           self.flowObjColl[id].settings.drawSettings[key]=\
                                           self.typeSettings[key](val.text())
        self.flowObjColl[id].calc_flow_lines()
        self.update.emit()
        
#==============================================================================
#     def update_histo_settings(self):
#         id=self.currentEditId
#         print "Updating optflow histogram display settings in " + id
#         for key, val in self.currentHistoSettings.iteritems():
#             print key, ", ", val.text()
#             self.flowObjColl[id].settings["default"].dispHistoSettings[key]=\
#                                            self.typeSettings[key](val.text())
#==============================================================================
        #self.flowObjColl[id].calc_flow_lines()
        #self.update.emit()
        
    def show_flow_histos(self):
        """Open figure display widget to show optical flow diagrams"""
        id = self.currentEditId
        flow = self.flowObjColl[id]
        fig = flow.plot_flow_histograms(drawGauss = 1, forApp = 1)
        self.widget=FigureDispWidget([fig])
        self.widget.setWindowTitle("piSCOPE: MPL Figure display interface")
        self.widget.setWindowModality(Qt.WindowModal)        
#==============================================================================
#         
#         canvas=self.widget.canvases[0]
#         canvas.update_fontsize(12)
#         n,b,_=canvas.axes.hist(lens,100)
#         maxPosInd=argmax(n)
#         maxPos=(b[maxPosInd+1]-b[maxPosInd])/2+b[maxPosInd]
#         canvas.axes.set_title("Optical flow: lengths distribution (Max @ " +"{:.1f}".format(maxPos) + ")")        
#         
#         canvas=self.widget.canvases[1]
#         canvas.update_fontsize(12)
#         n,b,_=canvas.axes.hist(angles,bins=Bins)
#         maxPosInd=argmax(n)
#         maxPos=(b[maxPosInd+1]-b[maxPosInd])/2+b[maxPosInd]
#         canvas.axes.set_title("Optical flow: angle distribution (Max @ " +"{:.1f}".format(maxPos) + ")")        
#==============================================================================
        
        self.widget.show()
        
    def show_current_settings(self):
        id=self.currentEditId
        string=self.flowObjColl[id].settings.print_current_settings()
        QMessageBox.information(self,"Current flow settings",string,QMessageBox.Ok)
        
    def closeEvent(self,e):
        self.hide()
        
    def toggle_canvas_link(self):
        print "Check box TOGGLE"
        isLinked=not self.linkedToCanvas[self._id()]
        self.linkedToCanvas[self._id()]=isLinked
        num=self.parent().get_widget_num(self._id())
        print "isLinked: "+ str(isLinked)
        print "widget num: " + str(num)
        if num:
            canvas=self.parent().widgets[num].histoCanvas
            self.connectedCanvas[self._id()]=canvas
            if isLinked:
                canvas.updateContrast.connect(self.update_contrast)
                canvas.autoUpdate=False
                canvas.updateContrast.emit()
            else:
                canvas.updateContrast.disconnect()
                canvas.autoUpdate=True
            
        
#==============================================================================
#         msg="Oaans nachm annern! Geduld, Geduld mein Bester"
#         QMessageBox.warning(self,"Error",msg, QMessageBox.Ok)
#==============================================================================
    def update_contrast(self):
        """Update contrast in a displayed image
        
        This function is for instance called when the brigthness range in the 
        histogram of a connected :class:`ImgViewer` is interactively changed.
        
        Steps:
        
            1. get the :mod:`piscope.gui.ImgDispWidget.ImgDispWidget` object
                where the brightness change was applied and it's id (e.g. "On")
            
            #. get the current iMin and iMax values from the histogram canvas
                object

            #.Write those into the input settings of the corresponding 
                :mod:`piscope.PlumeSpeed.OpticalFlowFarneback`object 

            #. reload the settings in the dialog and swap the displayed settings
                (i.e. show the optical flow settings of the image viewer, where the
                change in histogram was applied)
                Note that if the dialog was currently displaying the settings of 
                one of the other image viewers, all changes are lost, which were not
                applied by pressing the corresponding update button.

            #. redisplay the optical flow
            
        """
        print "CHANGE IN HISTOGRAM DETECTED, UPDATE CONTRAST"
        widget=self.sender().parent().baseViewer
        print "ImageViewer: " + widget.id
        iMin, iMax=widget.histoCanvas.dragLines.xCoordinates
        self.flowObjColl[widget.id].settings["iMin"]=float(iMin)
        self.flowObjColl[widget.id].settings["iMax"]=float(iMax)
        self.reload_settings(widget.id)
        self.update_presettings()
    """
    Some more convenience helpers
    """
    def _id(self):
        return self.currentEditId
    
    def _flow_active(self):
        return self.flowObjColl[self.currentEditId].active