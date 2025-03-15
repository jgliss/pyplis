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
import importlib.util
import sys
from traceback import format_exc
from time import time

from SETTINGS import ARGPARSER

IGNORE_SCRIPTS = ["ex0_5_optflow_livecam.py"]

def run_script(script_path: pathlib.Path):
    module_name = script_path.stem  # Use filename as module name

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # Add module to sys.modules
    spec.loader.exec_module(module)  # Execute module

if __name__ == "__main__":

    this_dir = pathlib.Path(__file__).parent
    all_intro_scripts = [x for x in this_dir.glob("ex0_*.py") if not x.name in IGNORE_SCRIPTS]
    
    # init lists that store messages that are printed after execution of all
    # scripts
    test_err_messages = []
    passed_messages = []

    t0 = time()
    for script_path in all_intro_scripts:
        try:
            run_script(script_path=script_path)
            passed_messages.append(f"All tests passed in script: {script_path.name}")
        except AssertionError as e:
            msg = (f"\n\n"
                f"--------------------------------------------------------\n"
                f"Tests in script {script_path.name} failed.\n"
                f"Error traceback:\n {format_exc(e)}\n"
                f"--------------------------------------------------------"
                f"\n\n"
                )
            test_err_messages.append(msg)
        except Exception as e:
            test_err_messages.append(f"Unexpected error: {e}")

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

    print(f"Total runtime: {t1 - t0:.2f} s")
    exit(len(test_err_messages))
