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
from __future__ import (absolute_import, division)
from os import listdir, unlink
from os.path import basename, join, isfile
from traceback import format_exc
from SETTINGS import OPTPARSE
from sys import exit
from time import time

paths = [f for f in listdir(".") if f[:4] == "ex0_" and f[4] != "5" and
         f.endswith("py")]

# init arrays, that store messages that are printed after execution of all
# scripts
test_err_messages = []
passed_messages = []

(options, args) = OPTPARSE.parse_args()

if options.clear:
    folder = "scripts_out"
    for the_file in listdir(folder):
        file_path = join(folder, the_file)
        try:
            if isfile(file_path):
                unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

t0 = time()
for path in paths:
    try:
        with open(path) as f:
            code = compile(f.read(), path, 'exec')
            exec(code)

        passed_messages.append("All tests passed in script: %s"
                               % basename(path))
    except AssertionError as e:
        msg = ("\n\n"
               "--------------------------------------------------------\n"
               "Tests in script %s failed.\n"
               "Error traceback:\n %s\n"
               "--------------------------------------------------------"
               "\n\n"
               % (basename(path), format_exc(e)))
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
