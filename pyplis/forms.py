# -*- coding: utf-8 -*-
#
# Pyplis is a Python library for the analysis of UV SO2 camera data
# Copyright (C) 2017 Jonas Gliss (jonasgliss@gmail.com)
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License a
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
from collections import OrderedDict as od
from pyplis import logger

class FormCollectionBase(object):
    """Abtract base class representing collection of geometrical forms.

    Abstract class providing basic functionality for object collections.
    Note that the basic management functions for adding / deleting forms
    :func:`add` , :func:`remove` create the objects based on start (x,y)
    and stop (x,y) points, i.e.  [x0, y0, x1, y1]

    This class and classes inheriting from it show large similarities to
    dictionaries
    """

    def __init__(self, forms_dict=None):
        """Class initialisation."""
        if forms_dict is None:
            forms_dict = {}
        self._forms = {}
        self.id_count = 0

        self.type = ""

        for key, val in forms_dict.items():
            self[key] = val

    def add(self, x0, y0, x1, y1, id=None):
        """Create a new form from input coordinates.

        :param int x0: x coordinate of start point
        :param int y0: y coordinate of start point
        :param int x1: x coordinate of stop point
        :param int y1: y coordinate of stop point
        :param str id: name of form (if None, it will be set automatically
                                        based on current object counter value)

        """
        if not self.check_input(x0, y0, x1, y1):
            raise ValueError("Check coordinates...")
        if id in self.keys():
            raise AttributeError("A form with name %s already exists" % id)
        if x1 < x0:
            x0, y0, x1, y1 = x1, y1, x0, y0
        if id is None:
            id = self.type + str(self.id_count)

        self._forms[id] = [x0, y0, x1, y1]
        self.id_count += 1
        return True

    @property
    def tot_num(self):
        """Return current number of forms in collection."""
        return len(self._forms.keys())

    def update(self, x0, y0, x1, y1, id):
        """Update an existing form (or create new if it does not exist)."""
        if id in self._forms.keys():
            logger.info("Form with ID " + str(id) + " could not be updated because "
                  "it does not exist, creating new form instead")
            self.remove(id)
        if self.add(x0, y0, x1, y1, id):
            return 1
        return 0

    def check_input(self, x0, y0, x1, y1):
        """Check if input for adding a form has the right format."""
        items = [x0, y0, x1, y1]
        for item in items:
            if not any([isinstance(item, tp) for tp in [int, float]]):
                return False
        return True

    def remove(self, id):
        """Remove one form from collection.

        :param str id: string id of the form to be deleted
        """
        if id not in self._forms.keys():
            logger.info("Error: could not delete form " + id + " from "
                  "collection, no such form in collection")
            return 0
        logger.info("Delete form " + id + " from collection")
        del self._forms[id]

    def rename(self, current_id, new_id):
        """Rename one form."""
        if new_id in self._forms.keys():
            raise Exception("Error renaming form %s to %s. Such a form already"
                            " exists in collection, delete the old one first "
                            % (current_id, new_id))
        l = self._forms[current_id]
        self.add(l[0], l[1], l[2], l[3], new_id)
        del self._forms[current_id]

    def has_form(self, form_id):
        """Check if collection has a form with input ID, returns bool."""
        if form_id in self._forms.keys():
            return True
        return False

    def form_info(self, form_id):
        """Print information about one of the forms in this collection.

        :param str form_id: string ID of form
        """
        f = self[form_id]
        add_str = ""
        if f is None:
            add_str = "(Form does not exist in collection)"
            f = ["undefined", "undefined", "undefined", "undefined"]
        s = ("ID: %s %s\nStart (x,y): %s, %s\nStop (x,y): %s, %s\n"
             % (form_id, add_str, f[0], f[1], f[2], f[3]))
        return s

    def keys(self):
        """Return names of all current forms."""
        return list(self._forms.keys())

    def values(self):
        """Return all current forms."""
        return list(self._forms.values())

    def get(self, form_id):
        """Get one form.

        :param str form_id: name of the form within collection
        """
        return self[form_id]

    def __call__(self, key):
        """Make object callable.

        Returns form based on input key (see also :func:`__getitem__`)
        """
        if key in self._forms.keys():
            return self._forms[key]
        else:
            # Default behaviour
            raise AttributeError

    def __setitem__(self, key, val):
        """Set item method.

        Adds one form to this collection

        :param str key: string ID of new form
        :param list val: list with start / stop coords ``[x0,y0, x1, y1]``
        """
        self.add(val[0], val[1], val[2], val[3], key)
        

    def __getitem__(self, name):
        """Get item.

        Returns form with corresponding to input name (if it exists)
        """
        try:
            return self._forms[name]
        except KeyError:
            logger.info("No such form: " + str(name))

    def __str__(self):
        s = ("\nForm collection %s\n-----------------------\n" % self.type)
        if not bool(self.keys()):
            return s + "No forms available"

        for key, val in self._forms.items():
            s = (s + "ID: %s\nStart (x,y): %s, %s\n"
                     "Stop (x,y): %s, %s\n"
                     % (key, val[0], val[1], val[2], val[3]))
        return s


class LineCollection(FormCollectionBase):
    """Class specifying line objects on images."""

    def __init__(self, forms_dict=None):
        super().__init__(forms_dict)
        if forms_dict is None:
            forms_dict = {}
        self.type = "line"


class RectCollection(FormCollectionBase):
    """Class specifying rectangle objects on images."""

    def __init__(self, forms_dict=None):
        super().__init__(forms_dict)
        self.type = "rect"

    def add(self, x0, y0, x1, y1, id=None):
        """Create a new form from input coordinates.

        :param int x0: x coordinate of start point
        :param int y0: y coordinate of start point
        :param int x1: x coordinate of stop point
        :param int y1: y coordinate of stop point
        :param str id: identification string of object (if None, the id
            will be set automatically based on current object counter value)

        """
        # top left / bottom right x coorindates
        tl_x, br_x = min([x0, x1]), max([x0, x1])  
        # top left / bottom right y coorindates
        tl_y, br_y = min([y0, y1]), max([y0, y1])
        return super().add(tl_x, tl_y, br_x, br_y, id)
