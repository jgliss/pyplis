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
from time import time
from SETTINGS import ARGPARSER, SCRIPTS_DIR
from run_all_helpers import get_all_script_paths, run_all_scripts, print_output_runall

IGNORE_SCRIPTS = ["ex0_5_optflow_livecam.py"]
SCRIPT_PATTERN = "ex0_*.py"
if __name__ == "__main__":

    options = ARGPARSER.parse_args()

    t0 = time()
    all_intro_scripts = get_all_script_paths(SCRIPTS_DIR, SCRIPT_PATTERN, IGNORE_SCRIPTS)
    test_err_messages, passed_messages, crashed_messages = run_all_scripts(all_intro_scripts)
    t1 = time()    
    print_output_runall(options, test_err_messages, passed_messages, crashed_messages)
    print(f"Total runtime: {t1 - t0:.2f} s")
