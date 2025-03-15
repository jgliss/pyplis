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
import pathlib 
import os
from traceback import format_exc
from sys import exit
from time import time

this_dir = pathlib.Path(__file__).parent
files = list(this_dir.glob("ex0_*.py"))

# init lists that store messages that are printed after execution of all
# scripts
test_err_messages = []
passed_messages = []

t0 = time()
for path in files:
    try:
        with open(path) as f:
            code = compile(f.read(), path, 'exec')
            exec(code)

        passed_messages.append(f"All tests passed in script: {os.path.basename(path)}")
    except AssertionError as e:
        msg = (f"\n\n"
               f"--------------------------------------------------------\n"
               f"Tests in script {os.path.basename(path)} failed.\n"
               f"Error traceback:\n {format_exc(e)}\n"
               f"--------------------------------------------------------"
               f"\n\n"
            )
        test_err_messages.append(msg)

t1 = time()


# If applicable, do some tests. This is done only if TESTMODE is active:
# testmode can be activated globally (see SETTINGS.py) or can also be
# activated from the command line when executing the script using the
# option --test 1
if int(options.test):
    print("\n----------------------------\n"
          "T E S T  F A I L U R E S"
          "\n----------------------------\n")
    if test_err_messages:
        for msg in test_err_messages:
            print(msg)
    else:
        print("None")
    print("\n----------------------------\n"
          "T E S T  S U C C E S S"
          "\n----------------------------\n")
    if passed_messages:
        for msg in passed_messages:
            print(msg)
    else:
        print("None")

print("Total runtime: %.2f s" % (t1 - t0))
exit(len(test_err_messages))
