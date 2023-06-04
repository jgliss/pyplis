# -*- coding: utf-8 -*-
"""Test environment for setupclasses.py module.

Author: Jonas Gliss
Email: jonasgliss@gmail.com
License: GPLv3+
"""
import pytest
import os, shutil
from pyplis import inout as mod

def test_create_temporary_copy(tmpdir):
    fpath = os.path.join(tmpdir, 'file.txt')
    with open(fpath, 'w') as f:
        f.write('bla')
    assert os.path.exists(fpath)
    loc = mod.create_temporary_copy(fpath)
    assert os.path.exists(loc)
    with open(loc) as f:
        assert f.readline() == 'bla'

def test_get_my_pyplis_dir():
    usr_dir = os.path.join(os.path.expanduser('~'), "my_pyplis")
    assert os.path.samefile(usr_dir, mod.get_my_pyplis_dir())

def test_get_paths_txt():
    usr_dir = os.path.join(os.path.expanduser('~'), "my_pyplis/_paths.txt")
    assert os.path.samefile(usr_dir, mod.get_paths_txt())

def test__path_registered(tmpdir):
    path = os.path.abspath('.')
    fpath = os.path.join(tmpdir, 'file.txt')
    with open(fpath, 'a') as f:
        f.write(path + '\n')
    assert mod._path_registered(path, fpath)
    assert mod._path_registered('.', fpath)
    assert mod._path_registered(tmpdir, fpath) == False

def test_download_testdata(tmpdir):

    save_dir = mod.download_test_data(tmpdir)
    assert os.path.exists(save_dir)
    dataloc = os.path.join(save_dir, 'pyplis_etna_testdata')
    assert os.path.exists(dataloc)

def test_find_test_data():
    mod.find_test_data()

