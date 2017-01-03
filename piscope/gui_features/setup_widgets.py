# -*- coding: utf-8 -*-
from PyQt4.QtGui import QDialog, QVBoxLayout, QTabWidget, QPushButton,\
    QHBoxLayout, QWidget, QFormLayout, QGroupBox, QLineEdit, QDateTimeEdit,\
    QLabel, QRadioButton, QIcon, QMessageBox, QValidator, QDoubleValidator,\
    QFileDialog, QCheckBox, QComboBox, QGridLayout#, QTimeEdit
from PyQt4.QtCore import QSize

#from numpy import nan
from copy import deepcopy
from os.path import join, exists
#from datetime import datetime
from collections import OrderedDict as od

from ..inout import get_source_ids, get_cam_ids, get_source_info, get_icon,\
                                                    get_source_info_online
from ..utils import Filter, DarkOffsetInfo
from ..setupclasses import FilterSetup, Camera, MeasSetup, Source


class MeasSetupEdit(QDialog):
    """Tab Widget for base setup of camera data (class:`MeasSetup`)"""
    def __init__(self, setup = None, parent = None):
        """Initialisation
        
        :param MeasSetup setup: setup class
        :param parent: parent widget
        """
        
        super(MeasSetupEdit, self).__init__(parent)
        self.setWindowTitle("piSCOPE: Measurement setup dialog")
        
        if isinstance(setup, MeasSetup):
            self.setup = setup
        else:
            self.setup = MeasSetup()

        self.last_setup = deepcopy(self.setup)
        
        self.changes_accepted = 0
        
        self.tab_cam = None
        self.tab_source = None
        
        self.icon_size = QSize(25,25)
        self.init_ui()
    
    def init_ui(self):
        """Create dialog tab layout"""
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        stp = self.setup
        self.tabs.addTab(self.init_base_tab(), "Basic information")
        
        tab_source = SourceEdit(stp.source)
        self.tabs.addTab(tab_source,"Source")
        self.tab_source = tab_source
        
        tab_cam = CameraEdit(stp.camera)
        self.tabs.addTab(tab_cam,"Camera")
        self.tab_cam = tab_cam
        
        buttonCancel = QPushButton("Cancel")
        buttonCancel.clicked.connect(self.close)
        
        buttonApply = QPushButton('Apply and close')
        buttonApply.clicked.connect(self.handle_apply_button)        
        buttonApply.setDefault(True)
        
        hBoxButtons = QHBoxLayout()
        hBoxButtons.addStretch(1)
        hBoxButtons.addWidget(buttonApply)
        hBoxButtons.addWidget(buttonCancel)
                
        layout.addWidget(self.tabs)
        layout.addLayout(hBoxButtons)
        self.setLayout(layout)
    
    
    def init_base_tab(self):
        """Initiation of first tab for base information"""
        tab = QWidget()
        loupIconFile = get_icon("myLoupe", "k")
        
        layout = QVBoxLayout()
        
        baseGroup = QGroupBox("Basic information")
        base_form = QFormLayout()
        #All the import IO widgets
        self.base_path_edit = QLineEdit(self.setup.base_path)
        button1 = QPushButton("Browse")
        button1.clicked.connect(self.browse_base_path)
        
        self.save_path_edit = QLineEdit(self.setup.save_path)
        button2 = QPushButton("Browse")
        button2.clicked.connect(self.browse_save_path)
        
        for button in (button1, button2):
            try:
                button.setFlat(True)
                button.setIcon(QIcon(loupIconFile))
                button.setIconSize(self.icon_size)
                button.setText("")
            except:
                pass
                
        self.start_edit = QDateTimeEdit(self.setup.start)
        self.start_edit.setMaximumWidth(350)
        self.start_edit.setCalendarPopup(1)
        self.stop_edit = QDateTimeEdit(self.setup.stop)
        self.stop_edit.setCalendarPopup(1)
        self.stop_edit.setMaximumWidth(350)
        
        self.option_buttons = {}        
        for key, val in self.setup.options.iteritems():
            self.option_buttons[key] = bt = QRadioButton()
            bt.setChecked(val)
        
        self.option_buttons["USE_ALL_FILES"].toggled.connect(\
                                        self.handle_all_files_toggle)
        
        #Descriptive labels for For
        allFilesLabel = QLabel("Use all files")
        allFilesLabel.setToolTip("If active, then all files in the base folder"
            " are included, independent of datetime")
        
        hBoxSelectBasePath = QHBoxLayout()
        hBoxSelectBasePath.addWidget(self.base_path_edit)
        hBoxSelectBasePath.addWidget(button1)
        
        hBoxSelectSavePath = QHBoxLayout()
        hBoxSelectSavePath.addWidget(self.save_path_edit)
        hBoxSelectSavePath.addWidget(button2)
        
        base_form.addRow(QLabel("Base path"), hBoxSelectBasePath)
        base_form.addRow(QLabel("Save path"), hBoxSelectSavePath)
    
        base_form.addRow(allFilesLabel, self.option_buttons["USE_ALL_FILES"])
        base_form.addRow(QLabel("Start (UTC)"), self.start_edit)
        base_form.addRow(QLabel("Stop (UTC)"), self.stop_edit)
        
        baseGroup.setLayout(base_form)
        
        meteoGroup = QGroupBox("Meteorology info")
        meteoForm = QFormLayout()
        self.meteo_io = {}
        for key,val in self.setup.wind_info.iteritems():
            self.meteo_io[key] = edit = QLineEdit(str(val))
            edit.setValidator(QDoubleValidator())
            edit.setMaximumWidth(100)
            edit.textChanged.connect(self.check_meteo_input_state)
            edit.textChanged.emit(edit.text())

            meteoForm.addRow(QLabel(key),edit)
            
        meteoGroup.setLayout(meteoForm)
        
        hBoxButtons = QHBoxLayout()
        hBoxButtons.addStretch(1)
        confirm_button = QPushButton("Confirm")
        confirm_button.clicked.connect(self.update_base_setup)
        confirm_button.setEnabled(True)
        hBoxButtons.addWidget(confirm_button)
        
        
        layout.addWidget(baseGroup)
        layout.addSpacing(10)
        layout.addWidget(meteoGroup)
        layout.addSpacing(10)
        layout.addLayout(hBoxButtons)
        
        if self.option_buttons["USE_ALL_FILES"].isChecked():
            self.enable_time_edit(1)
        tab.setLayout(layout)
        self.base_form = base_form
        
        return tab
    

    def update_base_setup(self):
        """Read labels and update ``self.setup``"""
        bp = join(self.base_path_edit.text())
        sp = join(self.save_path_edit.text())
        if exists(bp):
            self.setup.base_path = bp
        else:
            QMessageBox.warning(self, "Warning",
                    "The entered BasePath location does not exist, retry...",
                    QMessageBox.Ok)
        if exists(sp):
            self.setup.save_path = sp
        else:
            QMessageBox.warning(self, "Warning",
                    "The entered SavePath location does not exist, retry...",
                    QMessageBox.Ok)
        self.setup.options["USE_ALL_FILES"] = self.option_buttons[\
                                                "USE_ALL_FILES"].isChecked()
        self.setup.start = self.start_edit.dateTime().toPyDateTime()
        self.setup.stop = self.stop_edit.dateTime().toPyDateTime()
        if not self.setup.USE_ALL_FILES and self.setup.stop <= self.setup.start:
            msg = ("Start time equals or exceeds stop time")
            QMessageBox.warning(self, "Warning", msg, QMessageBox.Ok)
        
        for key, val in self.meteo_io.iteritems():
            try:
                self.setup.wind_info[key] = float(val.text())
            except:
                pass
                
    def closeEvent(self,event):
        """What to do on close"""
        if not self.changes_accepted:
            self.setup = self.last_setup
        event.accept()
    
    def handle_apply_button(self):
        """Apply changes and check if all necessary info is available"""
        ok,s = self.setup.base_info_check()
        if not ok:
            box=QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Warning")
            box.setText("Basic information is missing, do you want to proceed?")
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            box.setDefaultButton(QMessageBox.No)
            box.setDetailedText(s)
            reply = box.exec_()
            if reply == QMessageBox.No:
                return
                
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setText("Apply changes (please confirm)?")
        msg.setWindowTitle("piSCOPE information")
        msg.setDetailedText(str(self.setup))
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Abort |\
                                                            QMessageBox.Apply)
        msg.setDefaultButton(QMessageBox.Apply)
        reply=msg.exec_()
        if reply == QMessageBox.Apply:
            self.setup.set_camera(self.tab_cam.cam_setup)
            self.setup.camera.filter_setup.check_default_filters()
            self.changes_accepted = 1
        if reply == QMessageBox.Abort:
            reply1 = QMessageBox.question(self, "Confirm", "No changes will "\
                "be applied", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply1 == QMessageBox.Cancel:
                reply = QMessageBox.Cancel
        if reply != QMessageBox.Cancel:        
            self.close()
    
    """
    Helpers, convenience stuff, et cetera
    """
    def check_meteo_input_state(self, *args, **kwargs):
        """Check input fields for meteorological data
        
        Credit to 
        
        http://snorf.net/blog/2014/08/09/validating-user-input-in-
        pyqt4-using-qvalidator/

        """
        sender = self.sender()
        validator = sender.validator()
        state = validator.validate(sender.text(), 0)[0]
        
        if state == QValidator.Acceptable:
            color = '#e6ffe6' # green
        
        elif state == QValidator.Intermediate:
            color = '#ffffff' # white
        else:
            color = '#ffe6e6' # red
        sender.setStyleSheet('QLineEdit { background-color: %s }' % color)
        
    def handle_all_files_toggle(self):
        """What to do if ``self.option_buttons["USE_ALL_FILES"]`` is clicked"""
        self.enable_time_edit(self.option_buttons["USE_ALL_FILES"].isChecked())

    def enable_time_edit(self, bool):
        """Enable / disable start / stop time edit fields"""
        b = not bool
        self.start_edit.setEnabled(b)        
        self.stop_edit.setEnabled(b)        
    
    def browse_base_path(self):
        """Opens a dialog` for choosing the image basepath"""
        path = join(str(QFileDialog.getExistingDirectory(self, 
                               "Select Directory")) + '\\')
        self.base_path_edit.setText(path)
        if not exists(str(self.save_path_edit.text())):
            self.save_path_edit.setText(path)
            
    def browse_save_path(self):        
        """Opens a QFileDialog to choose the path where results are saved"""
        self.save_path_edit.setText(join(str(QFileDialog.getExistingDirectory(\
                                            self, "Select Directory")) + '\\'))
                                                             
class AutoCellCalibEdit(MeasSetupEdit):
    """Inherits from MeasSetupEdit including editing of calibration cell specs
    """
    def __init__(self, *args,**kwargs):
        super(AutoCellCalibEdit, self).__init__(*args,**kwargs)
        self.setWindowTitle("piSCOPE: Cell calibration setup (Auto)")
        self.cell_edit_dial = CellEdit(self.setup.cell_info_dict, parent = self)
        self.update_ui()
        
    def update_ui(self):
        """Add cell edit to UI"""
        editCellsButton = QPushButton("Edit")
        editCellsButton.setToolTip("Open dialog to edit gas cells used for "
            "camera calibration")
        editCellsButton.setFixedWidth(70)
        editCellsButton.clicked.connect(self.show_cell_edit_dialog)
        self.base_form.addRow(QLabel("Calibration cells"), editCellsButton)
        
        createBgCheckBox=QCheckBox()
        createBgCheckBox.setChecked(True)
        createBgCheckBox.setTristate(False)
        tt=("If checked, then a background image"
            " Dataset is automatically created from all images covered by "
            " the specified time window which are NOT cell calibration images")
        createBgCheckBox.setToolTip(tt)
        createBgLabel=QLabel("Create background image DataSet ")
        createBgLabel.setToolTip(tt)
        self.base_form.addRow(createBgLabel, createBgCheckBox)
        self.createBgCheckBox=createBgCheckBox
    def update_base_setup(self):
        super(AutoCellCalibEdit, self).update_base_setup()
        self.setup.createBgOption=bool(self.createBgCheckBox.checkState())
        print "HEEEEEERE: "  +  str(self.setup.createBgOption)
        
    def show_cell_edit_dialog(self):
        self.cell_edit_dial.show()
    
class CameraEdit(QDialog): 
    """
    Widget to setup (or choose from default) emission sources
    
    .. todo::
    
        Drop down with default sources
        
    """
    def __init__(self, cam_setup):
        super(CameraEdit, self).__init__()
        self.setWindowTitle("piSCOPE: Edit camera setup")
        if isinstance(cam_setup, Camera):
            self.cam_setup = cam_setup
        else:
            self.cam_setup = Camera()
        
        self._cam_setup_temp = deepcopy(self.cam_setup) #store a copy
        
        self.width_edit_lbl = 150
        self._type_info = {}
        
        self.user_io = None
        self.load_settings_datatypes()
        self.cam_ids = get_cam_ids()
        self.init_ui()
        self.write_current()
    
    def _tool_tips(self):
        """Some information on the file name conventions"""
        return self.cam_setup._info_dict
        
    def init_ui(self):
        """Initiate the layout"""
        self.current_setup_label = QLabel(self._cam_setup_temp.__str__())
        self.current_setup_label.setMinimumWidth(250)
        self.user_io = {}
        layout=QHBoxLayout()
        #The basic information
        groupBase = QGroupBox("Basic information and optics")
        baseLayout = QVBoxLayout()
        base_form = QFormLayout()
        #Fill the form layout for the basic setup 
        
        camCombo=QComboBox()
        camCombo.addItem("None")
        camCombo.addItems(self.cam_ids)
        camCombo.activated[str].connect(self.update_camera)
        
        base_form.addRow(QLabel("Choose"), camCombo)
        
        self.user_io["serNo"] = edit = QLineEdit()
        base_form.addRow(QLabel("Serial no. "), edit)
        
        for k in self._cam_setup_temp._type_dict:
            if not k in ["filterAcronyms","default_filters", "dark_info"]:
                self.user_io[k] = edit = QLineEdit()
                lbl = QLabel(k)
                try:
                    lbl.setToolTip(self._tool_tips[k])
                    edit.setToolTip(self._tool_tips[k])
                except:
                    pass
                base_form.addRow(lbl, edit)
            
        filterEditButton = QPushButton("Edit")
        bw = filterEditButton.fontMetrics().\
                                    boundingRect("Filter setup").width() + 30
        filterEditButton.setFixedWidth(bw)
        filterEditButton.clicked.connect(self.edit_filters)
        base_form.addRow(QLabel("Filters"), filterEditButton)
        
        darkEditButton = QPushButton("Edit")
        darkEditButton.setFixedWidth(bw)
        darkEditButton.clicked.connect(self.edit_dark_offset)
        base_form.addRow(QLabel("Dark & offset"), darkEditButton)
        
        hBoxButtons = QHBoxLayout()
        hBoxButtons.addStretch(1)
        confirm_button = QPushButton("Confirm")
        confirm_button.clicked.connect(self.update_setup)
        confirm_button.setEnabled(True)
        hBoxButtons.addWidget(confirm_button)
        
        baseLayout.addLayout(base_form)
        baseLayout.addSpacing(10)
        baseLayout.addLayout(hBoxButtons)
        
        groupBase.setLayout(baseLayout)
        layout.addWidget(groupBase)
        
        groupGeom=QGroupBox("Geometry setup")
        geomLayout=QVBoxLayout()
        geomForm=QFormLayout()
        for key in self._cam_setup_temp.geomData.keys():
            self.user_io[key] = edit = QLineEdit()
            edit.setFixedWidth(self.width_edit_lbl)
            geomForm.addRow(QLabel(key), edit)
        hBoxButtons2=QHBoxLayout()
        hBoxButtons2.addStretch(1)
        confirm_button2=QPushButton("Confirm")
        confirm_button2.clicked.connect(self.update_setup)
        confirm_button2.setEnabled(True)
        hBoxButtons2.addWidget(confirm_button2)
        geomLayout.addLayout(geomForm)
        geomLayout.addSpacing(15)
        geomLayout.addLayout(hBoxButtons2)
        groupGeom.setLayout(geomLayout)
        layout.addSpacing(15)
        layout.addWidget(groupGeom)
        
        layout.addSpacing(15)
    
        vBoxInfo=QVBoxLayout()
        vBoxInfo.addWidget(self.current_setup_label)
        vBoxInfo.addStretch(1)
        groupInfo=QGroupBox("Current setup")
        groupInfo.setLayout(vBoxInfo)
        
        layout.addWidget(groupInfo)
        
        self.setLayout(layout)
    
    def update_camera(self, text):
        """Change current camera"""
        if text in self.cam_ids:
            stp = self._cam_setup_temp.change_camera(str(text))
            #self.filterEdit = FilterEdit(stp.filter_setup, parent = self)
            for key, lEdit in self.user_io.iteritems():
#==============================================================================
#                 print key
#                 print stp[key]
#==============================================================================
                lEdit.setText(str(stp[key]))
                
    def write_current(self):
        """Write current settings from ``self._cam_setup_temp into input fields"""
        for key, lEdit in self.user_io.iteritems():
            lEdit.setText(str(self._cam_setup_temp[key]))
    
    def update_setup(self):
        """Update ``self._cam_setup_temp`` """
        errKeys=[]
        for key, lEdit in self.user_io.iteritems():
            try:
                lEdit.setStyleSheet("background-color: #ffffff;")
                self._cam_setup_temp[key] = self._type_info[key](lEdit.text())
            except:
                lEdit.setStyleSheet("background-color: #ffe6e6;")
                errKeys.append(key)
                print "Something wrong: " + str(key)
        if bool(errKeys):
            msg="Please check the following fields:\n"
            for item in errKeys:
                msg = msg  + "%s\n" %item
            QMessageBox.warning(self, "Input error", msg)
        self.current_setup_label.setText(self._cam_setup_temp.__str__())
        self.cam_setup = self._cam_setup_temp
        #self.cam_setup.load_default(self.cam_setup.camId)
        
    def edit_filters(self):
        """Open filter setup dialog"""
        self.filterEdit = FilterEdit(self._cam_setup_temp.filter_setup, parent = self)
        self.filterEdit.show()
    
    def edit_dark_offset(self):
        """Open filter setup dialog"""
        self.darkEdit = DarkOffsetEdit(self._cam_setup_temp.dark_info, parent = self)
        self.darkEdit.show()    
        
    def load_settings_datatypes(self):
        self._type_info = self._cam_setup_temp._type_dict
        self._type_info["serNo"] = int
        for key in self.cam_setup.geomData:
            self._type_info[key] = float

class SourceEdit(QDialog):
    """Widget to setup (or choose from default) emission sources"""
        
    def __init__(self, source_setup):
        super(SourceEdit, self).__init__()
        self.setWindowTitle("piSCOPE: Edit emission source")
        self.source_setup = source_setup
        self.all_sources = od()
        
        self.source_search_online = SourceSearchOnline(self)
        
        self.width_edit_lbl = 150
        
        self._type_info = source_setup._type_dict   
        self.user_io = None
        
        self.init_layout()
        
        self.load_all_sources()
        self.fill_combo()
        self.write_current()
        
    def update_sources(self, source_dict):
        """Add sources comboBox"""
        print "Updating sources in combo box, all sources: "
        for name, source in source_dict.iteritems():
            self.all_sources[name] = source
    
        self.fill_combo()
        
    def init_layout(self):
        """Initiate the dialog layout"""
        self.user_io = {}
        layout = QHBoxLayout()
        ioLayout = QVBoxLayout()
        
        #The input form
        ioForm = QFormLayout()
        
        self.source_combo = QComboBox()
        
        self.source_combo.activated[str].connect(\
                            self.handle_source_combo_change)
        
        ioForm.addRow(QLabel("Choose"), self.source_combo)
        
        self.user_io["name"] = edit = QLineEdit()
        edit.setFixedWidth(self.width_edit_lbl)
        #ioForm.addRow(QLabel("Source Name"), edit)
        for key in self._type_info.keys():
            self.user_io[key] = edit = QLineEdit()
            edit.setFixedWidth(self.width_edit_lbl)
            ioForm.addRow(QLabel(key), edit) 
        
        hBoxButtons = QHBoxLayout()
        hBoxButtons.addStretch(1)
        confirm_button = QPushButton("Confirm")
        confirm_button.clicked.connect(self.update_setup)
        confirm_button.setEnabled(True)
        hBoxButtons.addWidget(confirm_button)
        
        ioLayout.addLayout(ioForm)
        ioLayout.addLayout(hBoxButtons)
        
        groupSetup=QGroupBox("Setup Source")
        groupSetup.setLayout(ioLayout)
        
        self.current_setup_label = QLabel(self.source_setup.__str__())
        self.current_setup_label.setMinimumWidth(200)
        vBoxInfo = QVBoxLayout()
        vBoxInfo.addWidget(self.current_setup_label)
        vBoxInfo.addStretch(1)
        
        groupInfo = QGroupBox("Current setup")
        groupInfo.setLayout(vBoxInfo)
        
        layout.addWidget(groupSetup)
        layout.addSpacing(15)
        layout.addWidget(groupInfo)
        layout.addStretch(1)
        self.setLayout(layout)
    
    def load_all_sources(self):
        """Load all piSCOPE default sources"""
        for id in get_source_ids():
            try:
                self.all_sources[id] = get_source_info(id, 0).values()[0]
            except:
                print "Failed to load source %s" %id
            
    def fill_combo(self):
        """Fukk the current source combo box"""
        self.source_combo.clear()
        self.source_combo.addItem("None")        
        self.source_combo.addItem("_online_search")
        #self.source_combo.addItems(get_source_ids())
        self.source_combo.addItems(self.all_sources.keys())
 
    def handle_source_combo_change(self, text):
        """What to do when the user changes the current selectin in the combo"""
        if text == "None":
            print "Text is None in Combo...return"
            return
        elif text == "_online_search":
            self.source_search_online.show()
        else:
            self.write_source(text)
        
    def write_source(self, text):
        """Writes the source info into the current source display label"""
        if self.all_sources.has_key(text):
            source = self.all_sources[text]#.values()[0]
            for key, lEdit in self.user_io.iteritems():
                if source.has_key(key):
                    lEdit.setText(str(source[key]))
                else:
                    lEdit.setText("")
                
    def write_current(self):
        """Write the currently selected source"""
        for key, lEdit in self.user_io.iteritems():
            lEdit.setText(str(self.source_setup[key]))
            
    def update_setup(self):
        """Update the current :class:`piscope.Setup.Source` object"""
        for key, val in self.user_io.iteritems():
            self.source_setup[key] = self._type_info[key](val.text())

        self.current_setup_label.setText(self.source_setup.__str__())

class SourceSearchOnline(QDialog):
    """Interface for online source search"""
    def __init__(self, parent = None):
        super(SourceSearchOnline, self).__init__(parent)
        self.setWindowTitle("Online source search")
        
        self.search_results = {}
        
        self.input_edit = QLineEdit("")
        self.input_edit.setFixedWidth(160)
        
        self.search_button = QPushButton("Search")
        self.search_button.setFixedWidth(60)
        self.search_button.clicked.connect(self.search_sources)
        
        self.source_combo = QComboBox()
        self.source_combo.addItem("No results available")
        self.source_combo.currentIndexChanged[str].connect(self.check_current)
        
        self.show_info_button = QPushButton("?")
        self.show_info_button.setMaximumWidth(25)
        self.show_info_button.setEnabled(False)
        self.show_info_button.setToolTip("Display current source")
        self.show_info_button.clicked.connect(self.disp_source_details)
        self.confirm_button = QPushButton("Confirm and close")
        self.confirm_button.clicked.connect(self.confirm_and_close)
        
        self.init_layout()
        
    def init_layout(self):
        """Make the layout"""
        layout = QGridLayout()
        texts=["Enter name", "Search results"]
        for k in range(len(texts)):
            l=QLabel(texts[k])
            l.setFixedWidth(80)
            layout.addWidget(l,k,1)
        
        layout.addWidget(self.input_edit, 0, 2)
        layout.addWidget(self.search_button,0,3)
        layout.addWidget(self.source_combo,1,2)
        layout.addWidget(self.show_info_button,1,3)
        layout.addWidget(self.confirm_button, 2,3)
        self.setLayout(layout)
        
    def search_sources(self):
        """Search for sources online 
        
        Uses the current string in ``self.input_edit``
        """
        res=get_source_info_online(str(self.input_edit.text()))
        if bool(res):
            self.search_results = res
            self.fill_result_combo()
    
    def fill_result_combo(self):
        """Write search results into combo"""
        self.source_combo.clear()
        self.source_combo.addItems(self.search_results.keys())
    
    def check_current(self, text):
        """Check if string of current combo item corresponds to available source"""
        val=False
        if self.search_results.has_key(text):
            val=True
        self.show_info_button.setEnabled(val)
        
    def disp_source_details(self):
        """Display infos about current item in source Combo box"""
        text=str(self.source_combo.currentText())
        s=Source(infoDict=self.search_results[text])
        QMessageBox.information(self, "Source details", str(s))
    
    def confirm_and_close(self):
        """Try to write the current results into parent"""
        self.parent().update_sources(self.search_results)
        self.close()
        
    def closeEvent(self, e):
        """Print some very important info on closing"""
        print "Chocolate is fucking weird..."
        self.close()

class CellEdit(QDialog):
    """
    Dialog to set calibration cell informationf
    """
    def __init__(self, cellSpecs={}, parent=None):
        super(CellEdit, self).__init__(parent)     
        self.setWindowTitle('piSCOPE: Define filter setup')
        self.accepted=0
        #variables for interactive management
        self.cell_info_dict=cellSpecs
        self.stringFormat="{0:.2e}".format
        self.checkBoxes={}
        
        self.cellCounter=0
        
        self.init_ui()
        self.disp_current_setup()
        
        self.setLayout(self.layout)
        
    def init_ui(self):
        self.layout=QVBoxLayout()
        
        self.groupCells=QGroupBox()
        self.groupCells.setTitle("Edit calibration cells")
        
        self.vBoxCells=QVBoxLayout()
        self.gridCells = QGridLayout()
        self.gridCells.setHorizontalSpacing(10)

        self.gridCells.addWidget(QLabel("Cell ID"),0,0)
        self.gridCells.addWidget(QLabel("SO2-SCD [cm-2]"),0,1)
        self.gridCells.addWidget(QLabel("SO2-SCD Err [cm-2]"),0,2)
        self.gridCells.addWidget(QLabel("Add"),0,3)
        
        self.addCellButton=QPushButton("+")
        self.addCellButton.setToolTip("Add one cell")
        self.addCellButton.clicked.connect(self.add_cell)
        
        self.hBoxAddCell=QHBoxLayout()
        self.hBoxAddCell.addWidget(self.addCellButton)
        self.hBoxAddCell.addStretch(1)
        
        self.confirmCellButton=QPushButton("Confirm and close")
        self.confirmCellButton.setToolTip("Confirm the current setup")
        self.confirmCellButton.clicked.connect(self.confirm_cell_setup)
        
        self.cancelButton=QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.close)
        
        self.hBoxConfirm=QHBoxLayout()
        self.hBoxConfirm.addStretch(1)
        self.hBoxConfirm.addWidget(self.confirmCellButton)
        self.hBoxConfirm.addWidget(self.cancelButton)
        
        self.vBoxCells.addLayout(self.gridCells)
        self.vBoxCells.addLayout(self.hBoxAddCell)
        self.vBoxCells.addLayout(self.hBoxConfirm)
        self.vBoxCells.addStretch(1)
        
        #self.layout.addWidget(self.rectsWidget) 
        self.groupCells.setLayout(self.vBoxCells)
        self.layout.addWidget(self.groupCells)
    
    def disp_current_setup(self):
        f=self.stringFormat
        for key, cell in self.cell_info_dict.iteritems():
            self.cellCounter+=1
            row=self.cellCounter
            self.gridCells.addWidget(QLineEdit(key),row,0)
            self.gridCells.addWidget(QLineEdit(f(cell[0])),row,1)
            self.gridCells.addWidget(QLineEdit(f(cell[1])),row,2)
            
            self.checkBoxes[row]=QCheckBox()
            self.checkBoxes[row].setCheckState(1)
            self.checkBoxes[row].setTristate(False)
            self.gridCells.addWidget(self.checkBoxes[row],row,3)
                
    def remove_all(self):
        for k in range(self.gridCells.rowCount()-1):
            for i in range(self.gridCells.columnCount()):
                item=self.gridCells.itemAtPosition(k+1,i)
                if item is not None:
                    widget=item.widget()
                    self.gridCells.removeWidget(widget)
                    widget.deleteLater()
                    del widget
                self.gridCells.removeItem(item)
            #self.grid.removeItem(item)
            
#==============================================================================
#     def handle_delete_row(self):
#         print "Delete row"
#         print "i: " + str(self.cells)
#         sender=self.sender()
#         for k in range(self.gridCells.rowCount()):
#             item=self.gridCells.itemAtPosition(k,3)
#             if item is not None:
#                 if item.widget() is sender:
#                     print item.widget()
#                     cellID=self.gridCells.itemAtPosition(k,0).widget().text()
#                     del self.cells[str(cellID)]
#                     del self.deleteButtons[str(cellID)]
#                     
#         print "ii: " + str(self.cells)
#         #elf.remove_all()
#         self.disp_current_setup()
#==============================================================================
    def confirm_cell_setup(self):
        msg="The old setup will be overwritten. Please confirm"
        reply=QMessageBox.information(self, "piSCOPE Information", msg,\
            QMessageBox.Cancel, QMessageBox.Ok,QMessageBox.NoButton)
        if reply==QMessageBox.Ok:
            self.empty_cell_dict()
            for row in range(1,self.gridCells.rowCount()):
                print "Num of rows: " + str(self.gridCells.rowCount())
                print "Current row: " + str(row)
                ok=self.check_input_row(row)
                if ok and self.gridCells.itemAtPosition(row,3).widget().checkState() != 0:
                    id=self.gridCells.itemAtPosition(row,0).widget().text()
                    scd=self.gridCells.itemAtPosition(row,1).widget().text()
                    scdErr=self.gridCells.itemAtPosition(row,2).widget().text()
                    self.cell_info_dict[id]=[float(scd),float(scdErr)]
                else:
                    print "Entry in row " + str(row) + " invalid or inactive"
        self.accepted=1 
        print self.cell_info_dict
        self.close()
        
    
    def empty_cell_dict(self):
        keys=self.cell_info_dict.keys()
        for i in range(len(keys)):
            del self.cell_info_dict[keys[i]]            
            
    def check_input_row(self,rowNum):
        ok=1
        
        for k in range(3):
            item=self.gridCells.itemAtPosition(rowNum,k)
            print str(item)
            if item is None:
                ok=0
        return ok
        
    def add_cell(self):
        row=self.gridCells.rowCount()+1
        for k in range(3):
            self.gridCells.addWidget(QLineEdit(),row,k)
        self.checkBoxes[row]=QCheckBox()
        self.checkBoxes[row].setCheckState(1)
        self.checkBoxes[row].setTristate(False)
        self.gridCells.addWidget(self.checkBoxes[row],row,3)
                
class FilterEdit(QDialog):
    """Widget used to setup the camera filters"""   
    def __init__(self, filterCollection = None, parent = None):
        super(FilterEdit, self).__init__(parent)     
        self.setWindowTitle('piSCOPE: Define filter setup')
        
        self.accepted = 0
        #variables for interactive management
        if filterCollection is None:
            self.collection = FilterSetup()
        else:
            self.collection = filterCollection
        
        
        self.checkBoxes = {}
        self.defaultRadioButtons = {}
        
        self.filterCounter = 0
        
        self.init_ui()
        self.disp_current_setup()
        
        self.setLayout(self.layout)
    
    def init_ui(self):
        """Initiate the layout"""
        self.layout = QVBoxLayout()
        
        self.groupFilters = QGroupBox()
        self.groupFilters.setTitle("Edit filter specs")
        
        self.vBoxFilters = QVBoxLayout()
        self.gridFilters = QGridLayout()
        self.gridFilters.setHorizontalSpacing(10)

        self.gridFilters.addWidget(QLabel("Filter ID"),0,0)
        self.gridFilters.addWidget(QLabel("Type"),0,1)
        self.gridFilters.addWidget(QLabel("Acronym"),0,2)
        self.gridFilters.addWidget(QLabel("Acronym (measType)"),0,3)
        self.gridFilters.addWidget(QLabel("Central wavelength"),0,4)
        #self.gridFilters.addWidget(QLabel("Peak transmission"),0,4)
        self.gridFilters.addWidget(QLabel("Add"),0,5)
        self.gridFilters.addWidget(QLabel("Default filter"),0,6)
        
        
        self.addFilterButton = QPushButton("+")
        self.addFilterButton.setToolTip("Add one filter")
        self.addFilterButton.clicked.connect(self.add_filter)
        
        self.hBoxAddFilter = QHBoxLayout()
        self.hBoxAddFilter.addWidget(self.addFilterButton)
        self.hBoxAddFilter.addStretch(1)
        
        self.confirmFilterButton = QPushButton("Confirm and close")
        self.confirmFilterButton.setToolTip("Confirm the current setup")
        self.confirmFilterButton.clicked.connect(self.confirm_filter_setup)
        
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.close)
        
        self.hBoxConfirm=QHBoxLayout()
        self.hBoxConfirm.addStretch(1)
        self.hBoxConfirm.addWidget(self.confirmFilterButton)
        self.hBoxConfirm.addWidget(self.cancelButton)
        
        self.vBoxFilters.addLayout(self.gridFilters)
        self.vBoxFilters.addLayout(self.hBoxAddFilter)
        self.vBoxFilters.addLayout(self.hBoxConfirm)
        self.vBoxFilters.addStretch(1)
        
        #self.layout.addWidget(self.rectsWidget) 
        self.groupFilters.setLayout(self.vBoxFilters)
        self.layout.addWidget(self.groupFilters)
    
    def _new_type_combo(self, text = None):
        """create and return combo for filter type selection"""
        cb = QComboBox()
        cb.addItems(["on", "off"])
        cb.setFixedWidth(60)
        idx = cb.findText(text)
        if idx >= 0:
            cb.setCurrentIndex(idx)
        return cb
        #cb.activated[str].connect(self.update_camera)
        
    def disp_current_setup(self):
        """Display the current filter setup"""
        #self.remove_all()
        for key, filter in self.collection.filters.iteritems():
            self.filterCounter += 1
            row = self.filterCounter
            self.gridFilters.addWidget(QLineEdit(filter.id), row, 0)
            cb = self._new_type_combo(filter.type)
            #self.gridFilters.addWidget(QLineEdit(filter.type),row,1)
            self.gridFilters.addWidget(cb, row, 1)
            self.gridFilters.addWidget(QLineEdit(filter.acronym), row, 2)
            self.gridFilters.addWidget(QLineEdit(filter.measTypeAcronym), row, 3)
            self.gridFilters.addWidget(QLineEdit(str(filter.centralWL)),\
                row, 4)
#==============================================================================
#             self.gridFilters.addWidget(QLineEdit(\
#                 str(filter.maxTransmission)), row,4)
#==============================================================================
            self.checkBoxes[row] = QCheckBox()
            self.checkBoxes[row].setCheckState(1)
            self.checkBoxes[row].setTristate(False)
            self.gridFilters.addWidget(self.checkBoxes[row], row, 5)
            self.defaultRadioButtons[row] = QRadioButton()
            self.gridFilters.addWidget(self.defaultRadioButtons[row], row, 6)
            if key == self.collection.defaultKey:
                self.defaultRadioButtons[row].setChecked(1)
            
#==============================================================================
#     def remove_all(self):
#         """Remove all current filters in dialog"""
#         for k in range(self.gridFilters.rowCount()-1):
#             for i in range(self.gridFilters.columnCount()):
#                 item=self.gridFilters.itemAtPosition(k+1,i)
#                 if item is not None:
#                     widget=item.widget()
#                     self.gridFilters.removeWidget(widget)
#                     widget.deleteLater()
#                     del widget
#                 self.gridFilters.removeItem(item)
#             #self.grid.removeItem(item)
#==============================================================================
            
#==============================================================================
#     def handle_delete_row(self):
#         print "Delete row"
#         print "i: " + str(self.filters)
#         sender=self.sender()
#         for k in range(self.gridFilters.rowCount()):
#             item=self.gridFilters.itemAtPosition(k,3)
#             if item is not None:
#                 if item.widget() is sender:
#                     print item.widget()
#                     filterID=self.gridFilters.itemAtPosition(k,0).widget().text()
#                     del self.filters[str(filterID)]
#                     del self.deleteButtons[str(filterID)]
#                     
#         print "ii: " + str(self.filters)
#         #elf.remove_all()
#         self.disp_current_setup()
#==============================================================================
    def confirm_filter_setup(self):
        """Update the current filter setup based on current dialog settings"""
        msg = "The old setup will be overwritten. Please confirm"
        reply=QMessageBox.information(self, "piSCOPE Information", msg,\
            QMessageBox.Cancel, QMessageBox.Ok,QMessageBox.NoButton)
        if reply == QMessageBox.Ok:
            self.empty_filter_dict()
            for row in range(1,self.gridFilters.rowCount()):
                ok = self.check_input_row(row)
                if ok and self.gridFilters.itemAtPosition(row, 5).\
                                                widget().checkState() != 0:
                    id = str(self.gridFilters.itemAtPosition(row, 0).\
                                                            widget().text())
                    type = str(self.gridFilters.itemAtPosition(row, 1).\
                                                        widget().currentText())
                    acro = str(self.gridFilters.itemAtPosition(row, 2).\
                                                            widget().text())
                    mtAcro = str(self.gridFilters.itemAtPosition(row, 3).\
                                                            widget().text())
                    cWl = None
                    try:
                        cWl = int(self.gridFilters.itemAtPosition(row, 4).
                                                            widget().text())
                    except:
                            print ("Input (central wavelength or peak "
                                " transmission) missing in row " + str(row))

                    self.collection.filters[id] = Filter(id, type, acro,\
                                                                mtAcro, cWl)
                    if self.gridFilters.itemAtPosition(row, 6).widget().isChecked():
                        self.collection.defaultKey = id
                else:
                    print "Entry in row " + str(row) + " invalid or inactive"
        
        #self.collection.get_ids_on_off()
        if not bool(self.collection.idsOn):
            msg=("Filter specification missing. Please create at least one "
                "on band filter")
            QMessageBox.warning(self, "Error", msg, QMessageBox.Ok)
            return
               
        self.collection.check_default_filters()
        self.print_filter_overview()
        self.accepted = 1        
        self.close()
        
    
    def empty_filter_dict(self):
        """Delete all filters in current filter collection"""
        keys=self.collection.filters.keys()
        for i in range(len(keys)):
            del self.collection.filters[keys[i]]
            
    def print_filter_overview(self):
        """Print current filter overview"""
        for id,filter in self.collection.filters.iteritems():
            print 
            print "Filter: " + id
            print "Type: " + filter.type
            print "Acronym: " + filter.acronym
            print "Central wavelength: " + str(filter.centralWL) + " nm"
            print
            
            
    def check_input_row(self, rowNum):
        """Check if information in input row is ok
        
        :param int rowNum: the number of the row in the dialog
        """
        ok = 1
        for k in range(3):
            item = self.gridFilters.itemAtPosition(rowNum, k)
            if item is None:
                ok = 0
        return ok
        
    def add_filter(self):
        """Add one row to define (add) a new filter to the collection"""
        row=self.gridFilters.rowCount() + 1
        self.gridFilters.addWidget(QLineEdit(), row, 0)
        self.gridFilters.addWidget(self._new_type_combo(), row, 1)
        self.gridFilters.addWidget(QLineEdit(),row, 2)
        self.gridFilters.addWidget(QLineEdit(),row, 3)
        self.gridFilters.addWidget(QLineEdit(),row, 4)
        
        self.checkBoxes[row] = QCheckBox()
        self.checkBoxes[row].setCheckState(1)
        self.checkBoxes[row].setTristate(False)
        self.gridFilters.addWidget(self.checkBoxes[row], row, 5)
        self.defaultRadioButtons[row] = QRadioButton()
        self.gridFilters.addWidget(self.defaultRadioButtons[row], row, 6)

class DarkOffsetEdit(QDialog):
    """Widget used to setup the camera filters"""   
    def __init__(self, darkOffsetInfo = [], parent = None):
        super(DarkOffsetEdit, self).__init__(parent)     
        self.setWindowTitle('piSCOPE: Define dark / offset input information')
        
        self.accepted = 0
        #variables for interactive management
        if not isinstance(darkOffsetInfo, list):
            darkOffsetInfo = []
        
        self.dark_info = darkOffsetInfo
                
        self.checkBoxes = {}
        
        self.counter = 0
        
        self.init_ui()
        self.disp_current_setup()
        
        self.setLayout(self.layout)
    
    def init_ui(self):
        """Initiate the layout"""
        self.layout = QVBoxLayout()
        
        self.group = QGroupBox()
        self.group.setTitle("Edit dark / offset specs")
        
        self.vBox = QVBoxLayout()
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(10)
        
        self.grid.addWidget(QLabel("ID"), 0, 0)
        self.grid.addWidget(QLabel("Type"), 0, 1)
        self.grid.addWidget(QLabel("Acronym"), 0, 2)
        self.grid.addWidget(QLabel("Acronym (measType)"), 0, 3)
        self.grid.addWidget(QLabel("Read gain"), 0, 4)
        self.grid.addWidget(QLabel("Add"), 0, 5)
        
        self.addButton = QPushButton("+")
        self.addButton.setToolTip("Add one DarkOffsetInfo object")
        self.addButton.clicked.connect(self.add)
        
        self.hBoxAdd = QHBoxLayout()
        self.hBoxAdd.addWidget(self.addButton)
        self.hBoxAdd.addStretch(1)
        
        self.confirm_button = QPushButton("Confirm and close")
        self.confirm_button.setToolTip("Confirm the current setup")
        self.confirm_button.clicked.connect(self.confirm_setup)
        
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.close)
        
        self.hBoxConfirm = QHBoxLayout()
        self.hBoxConfirm.addStretch(1)
        self.hBoxConfirm.addWidget(self.confirm_button)
        self.hBoxConfirm.addWidget(self.cancelButton)
        
        self.vBox.addLayout(self.grid)
        self.vBox.addLayout(self.hBoxAdd)
        self.vBox.addLayout(self.hBoxConfirm)
        self.vBox.addStretch(1)
        
        #self.layout.addWidget(self.rectsWidget) 
        self.group.setLayout(self.vBox)
        self.layout.addWidget(self.group)
    
    def _new_gain_combo(self, text = None):
        """create and return combo for gain value selection"""
        cb = QComboBox()
        cb.addItems(["LOW", "HIGH"])
        cb.setFixedWidth(80)
        idx = cb.findText(text)
        if idx >= 0:
            cb.setCurrentIndex(idx)
        return cb
        
    def _new_type_combo(self, text = None):
        """create and return combo for type selection"""
        cb = QComboBox()
        cb.addItems(["dark", "offset"])
        cb.setFixedWidth(80)
        idx = cb.findText(text)
        if idx >= 0:
            cb.setCurrentIndex(idx)
        return cb
        #cb.activated[str].connect(self.update_camera)
        
    def disp_current_setup(self):
        """Display the current filter setup"""
        #self.remove_all()
        for info in self.dark_info:
            self.counter += 1
            row = self.counter
            self.grid.addWidget(QLineEdit(info.id), row, 0)
            cb = self._new_type_combo(info.type)
            self.grid.addWidget(cb, row, 1)
            self.grid.addWidget(QLineEdit(info.acronym), row, 2)
            self.grid.addWidget(QLineEdit(info.measTypeAcronym), row, 3)
            cb = self._new_gain_combo(info.gain)
            self.grid.addWidget(cb, row, 4)

            self.checkBoxes[row] = QCheckBox()
            self.checkBoxes[row].setCheckState(1)
            self.checkBoxes[row].setTristate(False)
            self.grid.addWidget(self.checkBoxes[row], row, 5)
        
    def confirm_setup(self):
        """Update the current filter setup based on current dialog settings"""
        msg = "The old setup will be overwritten. Please confirm"
        reply = QMessageBox.information(self, "piSCOPE Information", msg,\
            QMessageBox.Cancel, QMessageBox.Ok,QMessageBox.NoButton)
        if reply == QMessageBox.Ok:
            self.dark_info = []
            for row in range(1, self.grid.rowCount()):
                ok = self.check_input_row(row)
                if ok and self.grid.itemAtPosition(row, 5).widget().checkState() != 0:
                    id = str(self.grid.itemAtPosition(row, 0).widget().text())
                    type = str(self.grid.itemAtPosition(row, 1).\
                                                        widget().currentText())
                    acro = str(self.grid.itemAtPosition(row, 2).widget().text())
                    mtAcro = str(self.grid.itemAtPosition(row, 3).widget().text())
                    
                    gain = str(self.grid.itemAtPosition(row, 4).\
                                                        widget().currentText())


                    self.dark_info.append(DarkOffsetInfo(id, type, acro,\
                                                                mtAcro, gain))
                else:
                    print "Entry in row " + str(row) + " invalid or inactive"
        self.print_overview()
        self.accepted = 1        
        self.close()
        
    def print_overview(self):
        """Print current filter overview"""
        for info in self.dark_info:
            print info
        
    def check_input_row(self, rowNum):
        """Check if information in input row is ok
        
        :param int rowNum: the number of the row in the dialog
        """
        if self.grid.itemAtPosition(rowNum, 0) == None:
            return False
        return True
        
    def add(self):
        """Add one row to define (add) a new dark / offset object"""
        row = self.grid.rowCount() + 1
        self.grid.addWidget(QLineEdit(), row, 0)
        cb = self._new_type_combo()
        self.grid.addWidget(cb, row, 1)
        self.grid.addWidget(QLineEdit(), row, 2)
        self.grid.addWidget(QLineEdit(), row, 3)
        cb = self._new_gain_combo()
        self.grid.addWidget(cb, row, 4)

        self.checkBoxes[row] = QCheckBox()
        self.checkBoxes[row].setCheckState(1)
        self.checkBoxes[row].setTristate(False)
        self.grid.addWidget(self.checkBoxes[row], row, 5)
        
#==============================================================================
# class BgModelEdit(QDialog):
#     """
#     Dialog to edit an :class:`BackGroundModelSetup` object
#     """
#     def __init__(self, bgModelSetup, parent=None):
#         super(BgModelEdit, self).__init__(parent)
#         self.setWindowTitle("piSCOPE: Background model settings")
#         self.setup=bgModelSetup
#         self.width_edit_lbl=150
#         self._type_info={}
#         self.changes_accepted=0
#         self.user_io=None
# 
#         self.init_ui()
# 
#         self.write_current()
#         
#         
#     def init_ui(self):
#         self.user_io={}
#         outerLayout=QVBoxLayout()
#         layout=QHBoxLayout()
#         ioLayout=QVBoxLayout()
#         
#         settingsForm=QFormLayout()
#         for key, val in self.setup.settings.iteritems():
#             self.user_io[key]=cBox=QCheckBox()
#             cBox.setTristate(0)
#             settingsForm.addRow(QLabel(key), cBox)
#         
#         
#         choosePolysButton=QPushButton("Select")
#         choosePolysButton.clicked.connect(self.get_poly_files)
#         resetPolysButton=QPushButton("Reset")
#         resetPolysButton.clicked.connect(self.reset_poly_files)
#         hBoxPolyButtons=QHBoxLayout()
#         hBoxPolyButtons.addWidget(choosePolysButton)
#         hBoxPolyButtons.addWidget(resetPolysButton)
#         settingsForm.addRow(QLabel("Background poly files [.bgp]"), hBoxPolyButtons)
#         
#         setTimeIvalsButton=QPushButton("Setup")
#         setTimeIvalsButton.clicked.connect(self.show_tival_edit)
#         settingsForm.addRow(QLabel("Set poly time windows"), setTimeIvalsButton)
#         
#         hBoxConfirm=QHBoxLayout()
#         hBoxConfirm.addStretch(1)
#         confirm_button=QPushButton("Confirm")
#         confirm_button.clicked.connect(self.update_setup)
#         confirm_button.setEnabled(True)
#         hBoxConfirm.addWidget(confirm_button)
#         
#         ioLayout.addLayout(settingsForm)
#         ioLayout.addLayout(hBoxConfirm)
#         
#         groupIO=QGroupBox("Settings")
#         groupIO.setLayout(ioLayout)
#         
#         self.current_setup_label=QLabel(str(self.setup))
#         self.current_setup_label.setMinimumWidth(200)
#         vBoxInfo=QVBoxLayout()
#         vBoxInfo.addWidget(self.current_setup_label)
#         vBoxInfo.addStretch(1)
#         
#         groupInfo=QGroupBox("Current setup")
#         groupInfo.setLayout(vBoxInfo)
#         
#         layout.addWidget(groupIO)
#         layout.addSpacing(15)
#         layout.addWidget(groupInfo)
#         layout.addStretch(1)
#         
#         outerLayout.addLayout(layout)
#         
#         buttonCancel=QPushButton("Cancel")
#         buttonCancel.clicked.connect(self.close)
#         
#         buttonApply = QPushButton('Apply and close')
#         buttonApply.clicked.connect(self.handle_apply_button)        
#         buttonApply.setDefault(True)
#         
#         hBoxButtons=QHBoxLayout()
#         hBoxButtons.addStretch(1)
#         hBoxButtons.addWidget(buttonApply)
#         hBoxButtons.addWidget(buttonCancel)
#         
#         outerLayout.addLayout(hBoxButtons)
#         self.setLayout(outerLayout)
#         
#     
#     def reset_poly_files(self):
#         self.setup.bgPolyFiles=[]
#         self.current_setup_label.setText(self.setup.__str__())
#         
#     def get_poly_files(self):
#         qStrList=QFileDialog.getOpenFileNames(self)
#         for s in qStrList:
#             self.setup.add_bg_poly_file(str(s))
#         self.current_setup_label.setText(self.setup.__str__())
#         
#     def write_current(self):
#         for key, cBox in self.user_io.iteritems():
#             cBox.setChecked(self.setup.settings[key])
#             
#     def update_setup(self):
#         for key, cBox in self.user_io.iteritems():
#             self.setup.settings[key]=bool(cBox.checkState())
#         self.current_setup_label.setText(self.setup.__str__())
#         
#     def show_tival_edit(self):
#         dial=BgPolyTimeWindowsEdit("test", [])
#         dial.exec_()
#         
#     def handle_apply_button(self):
#         """
#         Apply changes and check if all necessary info is available
#         """
#         msg = QMessageBox(self)
#         msg.setIcon(QMessageBox.Information)
#         msg.setText("Apply changes (please confirm)?")
#         msg.setWindowTitle("piSCOPE information")
#         msg.setDetailedText(str(self.setup))
#         msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Abort | QMessageBox.Apply)
#         msg.setDefaultButton(QMessageBox.Apply)
#         reply=msg.exec_()
#         if reply == QMessageBox.Apply:
#             self.changes_accepted=1
#         if reply == QMessageBox.Abort:
#             reply1=QMessageBox.question(self, "Confirm", "No changes will be applied",QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
#             if reply1==QMessageBox.Cancel:
#                 reply=QMessageBox.Cancel
#         if reply != QMessageBox.Cancel:        
#             self.close()
# #==============================================================================
# #             except:
# #                 val.setStyleSheet("color: rgb(255, 0, 50);")
# #                 msg="Oaans nachm annern! Geduld, Geduld mein Bester"
# #                 QMessageBox.warning(self,"Error",msg, QMessageBox.Ok)
# #==============================================================================
# class BgPolyTimeWindowsEdit(QDialog):
#     """
#     Dialog to open and edit time windows used to fit a background polynomial
#     """
#     def __init__(self, rectId, timeWindows=[], parent=None):
#         super(BgPolyTimeWindowsEdit, self).__init__(parent)     
#         self.rectId=rectId
#         self.setWindowTitle('piSCOPE: BG poly time windows (Rect: ' +\
#                                                             str(rectId) + ")")
#         self.accepted=0
#         #variables for interactive management
#         self.timeWindows=timeWindows
#         self.checkBoxes={}
#         
#         self.ivalCounter=0
#         
#         self.init_ui()
#         self.disp_current_setup()
#         
#         self.setLayout(self.layout)
#         
#     def init_ui(self):
#         self.layout=QVBoxLayout()
#         
#         self.groupIvals=QGroupBox()
#         self.groupIvals.setTitle("Edit time intervals and weighting factors"\
#             "(Rect: " + str(self.rectId) + ")")
#         
#         self.vBoxIvals=QVBoxLayout()
#         self.gridIvals = QGridLayout()
#         self.gridIvals.setHorizontalSpacing(10)
# 
#         self.gridIvals.addWidget(QLabel("Start"),0,0)
#         self.gridIvals.addWidget(QLabel("Stop"),0,1)
#         self.gridIvals.addWidget(QLabel("Weight"),0,2)
#         self.gridIvals.addWidget(QLabel("Add"),0,3)
#         
#         self.addIvalButton=QPushButton("+")
#         self.addIvalButton.setToolTip("Add one iVal")
#         self.addIvalButton.clicked.connect(self.add_ival)
#         
#         self.hBoxAddIval=QHBoxLayout()
#         self.hBoxAddIval.addWidget(self.addIvalButton)
#         self.hBoxAddIval.addStretch(1)
#         
#         self.confirmIvalButton=QPushButton("Confirm and close")
#         self.confirmIvalButton.setToolTip("Confirm the current setup")
#         self.confirmIvalButton.clicked.connect(self.confirm_ival_setup)
#         
#         self.cancelButton=QPushButton("Cancel")
#         self.cancelButton.clicked.connect(self.close)
#         
#         self.hBoxConfirm=QHBoxLayout()
#         self.hBoxConfirm.addStretch(1)
#         self.hBoxConfirm.addWidget(self.confirmIvalButton)
#         self.hBoxConfirm.addWidget(self.cancelButton)
#         
#         self.vBoxIvals.addLayout(self.gridIvals)
#         self.vBoxIvals.addLayout(self.hBoxAddIval)
#         self.vBoxIvals.addLayout(self.hBoxConfirm)
#         self.vBoxIvals.addStretch(1)
#         
#         #self.layout.addWidget(self.rectsWidget) 
#         self.groupIvals.setLayout(self.vBoxIvals)
#         self.layout.addWidget(self.groupIvals)
#     
#     def disp_current_setup(self):
#         for ival in self.timeWindows:
#             self.ivalCounter+=1
#             row=self.ivalCounter
#             self.gridIvals.addWidget(QTimeEdit(ival[0].time()),row,0)
#             self.gridIvals.addWidget(QTimeEdit(ival[1].time()),row,1)
#             self.gridIvals.addWidget(QLineEdit(ival[2]),row,2)
#             self.checkBoxes[row]=QCheckBox()
#             self.checkBoxes[row].setCheckState(1)
#             self.checkBoxes[row].setTristate(False)
#             self.gridIvals.addWidget(self.checkBoxes[row],row,3)
#                 
#     def remove_all(self):
#         for k in range(self.gridIvals.rowCount()-1):
#             for i in range(self.gridIvals.columnCount()):
#                 item=self.gridIvals.itemAtPosition(k+1,i)
#                 if item is not None:
#                     widget=item.widget()
#                     self.gridIvals.removeWidget(widget)
#                     widget.deleteLater()
#                     del widget
#                 self.gridIvals.removeItem(item)
# 
#     def confirm_ival_setup(self):
#         msg="The old setup will be overwritten. Please confirm"
#         reply=QMessageBox.information(self, "piSCOPE Information", msg,\
#             QMessageBox.Cancel, QMessageBox.Ok,QMessageBox.NoButton)
#         if reply==QMessageBox.Ok:
#             self.timeWindows=[]
#             for row in range(1,self.gridIvals.rowCount()):
#                 print "Num of rows: " + str(self.gridIvals.rowCount())
#                 print "Current row: " + str(row)
#                 ok = self.check_input_row(row)
#                 if ok and self.gridIvals.itemAtPosition(row,3).widget().checkState() != 0:
#                     start=self.gridIvals.itemAtPosition(row,0).widget().dateTime().toPyDateTime()
#                     stop=self.gridIvals.itemAtPosition(row,1).widget().dateTime().toPyDateTime()
#                     weight=int(self.gridIvals.itemAtPosition(row,2).widget().text())
#                     self.timeWindows.append([start, stop, weight])
#                 else:
#                     print "Entry in row " + str(row) + " invalid or inactive"
#         self.accepted = 1 
#         self.close()
#             
#     def check_input_row(self,rowNum):
#         ok=1
#         for k in range(3):
#             item=self.gridIvals.itemAtPosition(rowNum,k)
#             print str(item)
#             if item is None:
#                 ok=0
#         return ok
#         
#     def add_ival(self):
#         row=self.gridIvals.rowCount()+1
#         for k in range(2):
#             self.gridIvals.addWidget(QTimeEdit(datetime.now().time()),row,k)
#         self.gridIvals.addWidget(QLineEdit("1"),row, 2)
#         self.checkBoxes[row]=QCheckBox()
#         self.checkBoxes[row].setCheckState(1)
#         self.checkBoxes[row].setTristate(False)
#         self.gridIvals.addWidget(self.checkBoxes[row],row,3)
#==============================================================================
        

