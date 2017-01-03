# -*- coding: utf-8 -*-
from PyQt4.QtGui import QWidget, QIcon, QVBoxLayout, QPushButton,\
    QHBoxLayout, QComboBox, QMessageBox, QInputDialog
from PyQt4.QtCore import pyqtSignal, Qt, QSize

from ..inout import get_icon
from ..forms import FormCollectionBase
from collections import OrderedDict as od

class AllFormsPopup(QWidget):
    """A little popup window showing all forms"""
    changed=pyqtSignal()
    def __init__(self, fields, parent):
        """
        
        :param fields: list of strings which contain the names and ids of the
        buttons provided by this popup. This object provides support for 
        displaying the buttons with icons. This only works, if for all 
        buttons in input list an default icon exists.
        
        Currently available are icons for the followind drawing forms (id):
        
            1. Lines        ("line")
            #. Rectangles   ("rect")
        
        So if :param fields: looks like this::
        
            fields=["rect","line"]
            
        then this widget will be displayed with the icons. If instead it looks
        like this::
        
            fields=["rect","line","circle"]
            
        then this widget will be displayed with standard QPushButton design.
        
        .. todo::
        
            1. Active and inactive version of icons

        """
        
        super(AllFormsPopup,self).__init__(parent)
        self.fields=fields
        self.iconNames= od([("rect"    ,   "myRect"),
                            ("line"    ,   "myLine")])
        self.iconSize=QSize(32,32)
        self.icons=od()
        self.buttons=od()
        
        self.current={"id"  :   None,
                      "icon":   None}
        
        self.create_buttons()
        self.create_layout()
        
        if self.load_icons():
            self.update_layout()
        
        self.set_current(self.fields[0])        
        
    def create_buttons(self):
        for id in self.fields:
            self.buttons[id]=QPushButton(id)
            self.buttons[id].setToolTip(id) 
            self.buttons[id].setObjectName(id)
            self.buttons[id].clicked.connect(self.set_active)               
                
    def create_layout(self):
        self.layout=QVBoxLayout()
        for id in self.fields:
            self.layout.addWidget(self.buttons[id])
        self.layout.setContentsMargins(0,0,0,0)
        self.setWindowFlags(Qt.Popup)
        self.setLayout(self.layout)
    
    def update_layout(self):
        for id, button in self.buttons.iteritems():
            button.setText("")
            button.setFlat(True)
            button.setIcon(self.icons[id])            
            button.setIconSize(self.iconSize)
            button.setMaximumWidth(self.iconSize.width())
    
    def set_current(self, id):
        self.current["id"]=id
        self.current["icon"]=self.icons[id]
        
    def set_active(self):
        """If one of the buttons is clicked, update the current id and send
        changed signal
        """
        id=str(self.sender().objectName())
        self.set_current(id)
        self.changed.emit()
        self.hide()
        print "Forms Popup was changed\n"
        
    def load_icons(self):
        for id in self.fields:
            if not id in self.iconNames.keys():
                print "ERROR IN LOAD ICONS: COULD NOT LOAD ICON"
                return 0
            else:
                self.icons[id]=QIcon(get_icon(self.iconNames[id],"k"))
        return 1

#==============================================================================
#     def mousePressEvent(self,event):
#         if event.button()==Qt.LeftButton:
#             self.drawOrigin = QPoint(self.mapToGlobal(event.pos()))
#             print "Button pressed at: " + str(self.drawOrigin)
#==============================================================================
        
class FormCollectionBaseWidget(QWidget):
    """Widget for management of draw form collections
    
    This class implements a widget to manage
    :class:`piscope.Forms.DrawObjectCollection` objects and classes inheriting 
    from those (e.g. LineCollection).
    
    """
    changed=pyqtSignal()
    def __init__(self, formCollection, parent=None):
        super(FormCollectionBaseWidget, self).__init__(parent)
        self.formColl = formCollection
        self.setWindowTitle(self.formColl.type)
                             
        self.create_layout()
        
        self.update_forms_combo()
        self.fill_set_as_combo()
        
    def create_layout(self):
        """Create the layout of this widget"""
        self.layout = QVBoxLayout()
        
        hBoxFormsInfo = QHBoxLayout()
        self.formsCombo = QComboBox()
        
#==============================================================================
#         self.addButton=QPushButton("Add")
#         self.addButton.clicked.connect(self.add_current_rect)
#==============================================================================
        self.renameButton = QPushButton("Rename")
        self.renameButton.clicked.connect(self.rename_form)          
        
        self.delButton = QPushButton("Delete")
        self.delButton.setToolTip("Delete the selected form ")
        self.delButton.clicked.connect(self.delete_form)          
        
        self.infoButton = QPushButton("?")
        self.infoButton.setFixedWidth(self.infoButton.fontMetrics().
            boundingRect("?").width() + 20)
        self.infoButton.clicked.connect(self.disp_form_info)
        
        hBoxFormsInfo.addWidget(self.formsCombo)
        hBoxFormsInfo.addWidget(self.renameButton)
        hBoxFormsInfo.addWidget(self.delButton)
        hBoxFormsInfo.addWidget(self.infoButton)
        
        hBoxSetAs = QHBoxLayout()
        
        self.setAsCombo = QComboBox()
        self.setAsCombo.addItem("No default forms")
        self.setAsCombo.currentIndexChanged.connect(self.check_default_info)
        
        self.setAsInfoButton=QPushButton("?")
        self.setAsInfoButton.setFixedWidth(self.infoButton.width())
        self.setAsInfoButton.clicked.connect(self.disp_default_form_info)
        self.setAsInfoButton.setEnabled(False)
        
        self.setAsButton=QPushButton("Set as...")
        self.setAsButton.setToolTip("Assign the current form to one of the"
            " default forms of the DataSet object (if available). Choose from "
            " the ComboBox displayed right.")
            
        self.setAsButton.clicked.connect(self.handle_set_as_button)
        self.setAsButton.setEnabled(False)
        
        hBoxSetAs.addWidget(self.setAsButton)
        hBoxSetAs.addWidget(self.setAsCombo)
        hBoxSetAs.addWidget(self.setAsInfoButton)
        
#==============================================================================
#         self.layout.addRow(self.addButton,QLabel())
#==============================================================================
        self.layout.addLayout(hBoxFormsInfo)
        self.layout.addLayout(hBoxSetAs)
        
        self.setLayout(self.layout)
    
    def update_forms_combo(self):
        """Update the current forms combo
        
        Clear current combo and write all ids of current collection        
        """
        self.formsCombo.clear()
        self.formsCombo.addItems(self.formColl.forms.keys())
        if self.formColl.totNum > 0:
            self.activate_buttons(True)
        else:
            self.activate_buttons(False)
            
    def fill_set_as_combo(self):
        """Fill the current setAs combo for default objects with corresponding 
        ids"""
        self.setAsCombo.clear()
        items = self.formColl.defaultIds
        if bool(items):        
            self.setAsCombo.addItems(items)
            self.setAsButton.setEnabled(True)
        else:
            self.setAsCombo.addItem("No default forms")
            
    def handle_set_as_button(self):
        """What to do when the setAs button is clicked
        
        Tries to set current form as form with default key
        """
        defaultFormId = str(self.setAsCombo.currentText())
        curFormId = str(self.formsCombo.currentText())
        try:
            self.formColl.rename(curFormId, defaultFormId)
            self.update_forms_combo()
        except Exception as e:
            QMessageBox.warning(self,"piSCOPE error", repr(e), QMessageBox.Ok)
    
    def check_default_info(self):
        """Check if default infor is available for current item in 
        **self.setAsCombo``
        """
        key = str(self.setAsCombo.currentText())
        if key in self.formColl.defaultIds:
            self.setAsInfoButton.setEnabled(True)
        else:
            self.setAsInfoButton.setEnabled(False)
    
    def disp_form_info(self):
        """Display coordinate information about current form in 
        ``self.formsCombo``
        """
        id = str(self.formsCombo.currentText())
        s = self.formColl.form_info(id)
        QMessageBox.information(self, "piSCOPE Information", s, QMessageBox.Ok)        

    def disp_default_form_info(self):
        """Disp information about current default form"""
        key = str(self.setAsCombo.currentText())
        s = self.formColl.form_info(key, 1)
        QMessageBox.information(self, "piSCOPE Information", s, QMessageBox.Ok)
    
    def rename_form(self):
        """Rename a form
        
        Opens input dialog for entering new name, then perform renaming attempt
        and if this fails, open a warning box, printing the exception        
        """
        key = str(self.formsCombo.currentText())
        msg = "Insert new name for form " + key + ": "
        newName, ok = QInputDialog.getText(self, "piSCOPE: Rename form", msg, 0)                                           
        if ok:
            try:
                self.formColl.rename(key,str(newName))
                self.update_forms_combo()
            except Exception as e:
                QMessageBox.warning(self,"piSCOPE error", repr(e), QMessageBox.Ok)
                
        
    def delete_form(self):
        """Delete one form from collection
        
        Deletes the form corresponding to current item in ``self.formsCombo``        
        """
        id = str(self.formsCombo.currentText())
        self.formColl.remove(id)
        self.update_forms_combo()
        self.changed.emit()

    def activate_buttons(self, bool):
        """Activate / deactivate widget buttons"""
        self.infoButton.setEnabled(bool)
        self.delButton.setEnabled(bool)
        self.renameButton.setEnabled(bool)
        
    def load_new_collection(self, formColl):
        """Load a new collection and update widget
        
        :param FormCollectionBase formColl:   the new collection (old data will
            be deleted)       
        """
        if not isinstance(formColl, FormCollectionBase):
            msg = "Form of type %s could not be loaded..." %type(formColl)
            QMessageBox.warning(self,"Error", msg, QMessageBox.Ok)
            return
        self.formColl = formColl
        self.update_forms_combo()
