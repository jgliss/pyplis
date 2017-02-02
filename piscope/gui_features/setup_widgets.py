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
        self.setWindowTitle("piscope: Measurement setup dialog")
        
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
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.close)
        
        btn_apply = QPushButton('Apply and close')
        btn_apply.clicked.connect(self.handle_apply_btn)        
        btn_apply.setDefault(True)
        
        hbox_btns = QHBoxLayout()
        hbox_btns.addStretch(1)
        hbox_btns.addWidget(btn_apply)
        hbox_btns.addWidget(btn_cancel)
                
        layout.addWidget(self.tabs)
        layout.addLayout(hbox_btns)
        self.setLayout(layout)
    
    
    def init_base_tab(self):
        """Initiation of first tab for base information"""
        tab = QWidget()
        loup_icon = get_icon("myLoupe", "k")
        
        layout = QVBoxLayout()
        
        base_group = QGroupBox("Basic information")
        base_form = QFormLayout()
        #All the import IO widgets
        self.base_path_edit = QLineEdit(self.setup.base_path)
        btn1 = QPushButton("Browse")
        btn1.clicked.connect(self.browse_base_path)
        
        self.save_path_edit = QLineEdit(self.setup.save_path)
        btn2 = QPushButton("Browse")
        btn2.clicked.connect(self.browse_save_path)
        
        for btn in (btn1, btn2):
            try:
                btn.setFlat(True)
                btn.setIcon(QIcon(loup_icon))
                btn.setIconSize(self.icon_size)
                btn.setText("")
            except:
                pass
                
        self.start_edit = QDateTimeEdit(self.setup.start)
        self.start_edit.setMaximumWidth(350)
        self.start_edit.setCalendarPopup(1)
        self.stop_edit = QDateTimeEdit(self.setup.stop)
        self.stop_edit.setCalendarPopup(1)
        self.stop_edit.setMaximumWidth(350)
        
        self.option_btns = {}        
        for key, val in self.setup.options.iteritems():
            self.option_btns[key] = bt = QRadioButton()
            bt.setChecked(val)
        
        self.option_btns["USE_ALL_FILES"].toggled.connect(\
                                        self.handle_all_files_toggle)
        
        #Descriptive labels for For
        allFilesLabel = QLabel("Use all files")
        allFilesLabel.setToolTip("If active, then all files in the base folder"
            " are included, independent of datetime")
        
        hBoxSelectBasePath = QHBoxLayout()
        hBoxSelectBasePath.addWidget(self.base_path_edit)
        hBoxSelectBasePath.addWidget(btn1)
        
        hBoxSelectSavePath = QHBoxLayout()
        hBoxSelectSavePath.addWidget(self.save_path_edit)
        hBoxSelectSavePath.addWidget(btn2)
        
        base_form.addRow(QLabel("Base path"), hBoxSelectBasePath)
        base_form.addRow(QLabel("Save path"), hBoxSelectSavePath)
    
        base_form.addRow(allFilesLabel, self.option_btns["USE_ALL_FILES"])
        base_form.addRow(QLabel("Start (UTC)"), self.start_edit)
        base_form.addRow(QLabel("Stop (UTC)"), self.stop_edit)
        
        base_group.setLayout(base_form)
        
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
        
        hbox_btns = QHBoxLayout()
        hbox_btns.addStretch(1)
        confirm_btn = QPushButton("Confirm")
        confirm_btn.clicked.connect(self.update_base_setup)
        confirm_btn.setEnabled(True)
        hbox_btns.addWidget(confirm_btn)
        
        
        layout.addWidget(base_group)
        layout.addSpacing(10)
        layout.addWidget(meteoGroup)
        layout.addSpacing(10)
        layout.addLayout(hbox_btns)
        
        if self.option_btns["USE_ALL_FILES"].isChecked():
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
        self.setup.options["USE_ALL_FILES"] = self.option_btns[\
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
    
    def handle_apply_btn(self):
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
        """What to do if ``self.option_btns["USE_ALL_FILES"]`` is clicked"""
        self.enable_time_edit(self.option_btns["USE_ALL_FILES"].isChecked())

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
        self.setWindowTitle("piscope: Edit camera setup")
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
            if not k in ["default_filters", "dark_info"]:
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
        
        hbox_btns = QHBoxLayout()
        hbox_btns.addStretch(1)
        confirm_btn = QPushButton("Confirm")
        confirm_btn.clicked.connect(self.update_setup)
        confirm_btn.setEnabled(True)
        hbox_btns.addWidget(confirm_btn)
        
        baseLayout.addLayout(base_form)
        baseLayout.addSpacing(10)
        baseLayout.addLayout(hbox_btns)
        
        groupBase.setLayout(baseLayout)
        layout.addWidget(groupBase)
        
        groupGeom=QGroupBox("Geometry setup")
        geomLayout=QVBoxLayout()
        geomForm=QFormLayout()
        for key in self._cam_setup_temp.geomData.keys():
            self.user_io[key] = edit = QLineEdit()
            edit.setFixedWidth(self.width_edit_lbl)
            geomForm.addRow(QLabel(key), edit)
        hbox_btns2=QHBoxLayout()
        hbox_btns2.addStretch(1)
        confirm_btn2=QPushButton("Confirm")
        confirm_btn2.clicked.connect(self.update_setup)
        confirm_btn2.setEnabled(True)
        hbox_btns2.addWidget(confirm_btn2)
        geomLayout.addLayout(geomForm)
        geomLayout.addSpacing(15)
        geomLayout.addLayout(hbox_btns2)
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
        
        hbox_btns = QHBoxLayout()
        hbox_btns.addStretch(1)
        confirm_btn = QPushButton("Confirm")
        confirm_btn.clicked.connect(self.update_setup)
        confirm_btn.setEnabled(True)
        hbox_btns.addWidget(confirm_btn)
        
        ioLayout.addLayout(ioForm)
        ioLayout.addLayout(hbox_btns)
        
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
        
        self.search_btn = QPushButton("Search")
        self.search_btn.setFixedWidth(60)
        self.search_btn.clicked.connect(self.search_sources)
        
        self.source_combo = QComboBox()
        self.source_combo.addItem("No results available")
        self.source_combo.currentIndexChanged[str].connect(self.check_current)
        
        self.show_info_btn = QPushButton("?")
        self.show_info_btn.setMaximumWidth(25)
        self.show_info_btn.setEnabled(False)
        self.show_info_btn.setToolTip("Display current source")
        self.show_info_btn.clicked.connect(self.disp_source_details)
        self.confirm_btn = QPushButton("Confirm and close")
        self.confirm_btn.clicked.connect(self.confirm_and_close)
        
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
        layout.addWidget(self.search_btn,0,3)
        layout.addWidget(self.source_combo,1,2)
        layout.addWidget(self.show_info_btn,1,3)
        layout.addWidget(self.confirm_btn, 2,3)
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
        self.show_info_btn.setEnabled(val)
        
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
        self.check_boxes={}
        
        self.cellCounter=0
        
        self.init_ui()
        self.disp_current_setup()
        
        self.setLayout(self.layout)
        
    def init_ui(self):
        self.layout=QVBoxLayout()
        
        self.groupCells=QGroupBox()
        self.groupCells.setTitle("Edit calibration cells")
        
        self.vBoxCells=QVBoxLayout()
        self.grid_cells = QGridLayout()
        self.grid_cells.setHorizontalSpacing(10)

        self.grid_cells.addWidget(QLabel("Cell ID"),0,0)
        self.grid_cells.addWidget(QLabel("SO2-SCD [cm-2]"),0,1)
        self.grid_cells.addWidget(QLabel("SO2-SCD Err [cm-2]"),0,2)
        self.grid_cells.addWidget(QLabel("Add"),0,3)
        
        self.addCellButton=QPushButton("+")
        self.addCellButton.setToolTip("Add one cell")
        self.addCellButton.clicked.connect(self.add_cell)
        
        self.hbox_addCell=QHBoxLayout()
        self.hbox_addCell.addWidget(self.addCellButton)
        self.hbox_addCell.addStretch(1)
        
        self.confirmCellButton=QPushButton("Confirm and close")
        self.confirmCellButton.setToolTip("Confirm the current setup")
        self.confirmCellButton.clicked.connect(self.confirm_cell_setup)
        
        self.btn_cancel=QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.close)
        
        self.hbox_confirm=QHBoxLayout()
        self.hbox_confirm.addStretch(1)
        self.hbox_confirm.addWidget(self.confirmCellButton)
        self.hbox_confirm.addWidget(self.btn_cancel)
        
        self.vBoxCells.addLayout(self.grid_cells)
        self.vBoxCells.addLayout(self.hbox_addCell)
        self.vBoxCells.addLayout(self.hbox_confirm)
        self.vBoxCells.addStretch(1)
        
        #self.layout.addWidget(self.rectsWidget) 
        self.groupCells.setLayout(self.vBoxCells)
        self.layout.addWidget(self.groupCells)
    
    def disp_current_setup(self):
        f=self.stringFormat
        for key, cell in self.cell_info_dict.iteritems():
            self.cellCounter+=1
            row=self.cellCounter
            self.grid_cells.addWidget(QLineEdit(key),row,0)
            self.grid_cells.addWidget(QLineEdit(f(cell[0])),row,1)
            self.grid_cells.addWidget(QLineEdit(f(cell[1])),row,2)
            
            self.check_boxes[row]=QCheckBox()
            self.check_boxes[row].setCheckState(1)
            self.check_boxes[row].setTristate(False)
            self.grid_cells.addWidget(self.check_boxes[row],row,3)
                
    def remove_all(self):
        for k in range(self.grid_cells.rowCount()-1):
            for i in range(self.grid_cells.columnCount()):
                item=self.grid_cells.itemAtPosition(k+1,i)
                if item is not None:
                    widget=item.widget()
                    self.grid_cells.removeWidget(widget)
                    widget.deleteLater()
                    del widget
                self.grid_cells.removeItem(item)
            #self.grid.removeItem(item)
            
#==============================================================================
#     def handle_delete_row(self):
#         print "Delete row"
#         print "i: " + str(self.cells)
#         sender=self.sender()
#         for k in range(self.grid_cells.rowCount()):
#             item=self.grid_cells.itemAtPosition(k,3)
#             if item is not None:
#                 if item.widget() is sender:
#                     print item.widget()
#                     cellID=self.grid_cells.itemAtPosition(k,0).widget().text()
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
            for row in range(1,self.grid_cells.rowCount()):
                print "Num of rows: " + str(self.grid_cells.rowCount())
                print "Current row: " + str(row)
                ok=self.check_input_row(row)
                if ok and self.grid_cells.itemAtPosition(row,3).widget().checkState() != 0:
                    id=self.grid_cells.itemAtPosition(row,0).widget().text()
                    scd=self.grid_cells.itemAtPosition(row,1).widget().text()
                    scdErr=self.grid_cells.itemAtPosition(row,2).widget().text()
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
            item=self.grid_cells.itemAtPosition(rowNum,k)
            print str(item)
            if item is None:
                ok=0
        return ok
        
    def add_cell(self):
        row=self.grid_cells.rowCount()+1
        for k in range(3):
            self.grid_cells.addWidget(QLineEdit(),row,k)
        self.check_boxes[row]=QCheckBox()
        self.check_boxes[row].setCheckState(1)
        self.check_boxes[row].setTristate(False)
        self.grid_cells.addWidget(self.check_boxes[row],row,3)
                
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
        
        
        self.check_boxes = {}
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
        
        self.hbox_addFilter = QHBoxLayout()
        self.hbox_addFilter.addWidget(self.addFilterButton)
        self.hbox_addFilter.addStretch(1)
        
        self.confirmFilterButton = QPushButton("Confirm and close")
        self.confirmFilterButton.setToolTip("Confirm the current setup")
        self.confirmFilterButton.clicked.connect(self.confirm_filter_setup)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.close)
        
        self.hbox_confirm=QHBoxLayout()
        self.hbox_confirm.addStretch(1)
        self.hbox_confirm.addWidget(self.confirmFilterButton)
        self.hbox_confirm.addWidget(self.btn_cancel)
        
        self.vBoxFilters.addLayout(self.gridFilters)
        self.vBoxFilters.addLayout(self.hbox_addFilter)
        self.vBoxFilters.addLayout(self.hbox_confirm)
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
            self.check_boxes[row] = QCheckBox()
            self.check_boxes[row].setCheckState(1)
            self.check_boxes[row].setTristate(False)
            self.gridFilters.addWidget(self.check_boxes[row], row, 5)
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
        
        self.check_boxes[row] = QCheckBox()
        self.check_boxes[row].setCheckState(1)
        self.check_boxes[row].setTristate(False)
        self.gridFilters.addWidget(self.check_boxes[row], row, 5)
        self.defaultRadioButtons[row] = QRadioButton()
        self.gridFilters.addWidget(self.defaultRadioButtons[row], row, 6)

class DarkOffsetEdit(QDialog):
    """Widget used to setup the camera filters"""   
    def __init__(self, dark_info = [], parent = None):
        super(DarkOffsetEdit, self).__init__(parent)     
        self.setWindowTitle('piscope: Define dark / offset input information')
        
        self.accepted = 0
        #variables for interactive management
        if not isinstance(dark_info, list):
            dark_info = []
        
        self.dark_info = dark_info
                
        self.check_boxes = {}
        
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
        self.grid.addWidget(QLabel("Acronym (meas type)"), 0, 3)
        self.grid.addWidget(QLabel("Read gain"), 0, 4)
        self.grid.addWidget(QLabel("Add"), 0, 5)
        
        self.btn_add = QPushButton("+")
        self.btn_add.setToolTip("Add one DarkOffsetInfo object")
        self.btn_add.clicked.connect(self.add)
        
        self.hbox_add = QHBoxLayout()
        self.hbox_add.addWidget(self.btn_add)
        self.hbox_add.addStretch(1)
        
        self.confirm_btn = QPushButton("Confirm and close")
        self.confirm_btn.setToolTip("Confirm the current setup")
        self.confirm_btn.clicked.connect(self.confirm_setup)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.close)
        
        self.hbox_confirm = QHBoxLayout()
        self.hbox_confirm.addStretch(1)
        self.hbox_confirm.addWidget(self.confirm_btn)
        self.hbox_confirm.addWidget(self.btn_cancel)
        
        self.vBox.addLayout(self.grid)
        self.vBox.addLayout(self.hbox_add)
        self.vBox.addLayout(self.hbox_confirm)
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

            self.check_boxes[row] = QCheckBox()
            self.check_boxes[row].setCheckState(1)
            self.check_boxes[row].setTristate(False)
            self.grid.addWidget(self.check_boxes[row], row, 5)
        
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

        self.check_boxes[row] = QCheckBox()
        self.check_boxes[row].setCheckState(1)
        self.check_boxes[row].setTristate(False)
        self.grid.addWidget(self.check_boxes[row], row, 5)
        

