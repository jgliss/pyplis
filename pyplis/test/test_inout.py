# -*- coding: utf-8 -*-
import pytest
import os
import shutil
from pyplis import inout as mod
from pyplis import __dir__ as PYPLIS_INSTALL_ROOT

import pytest
from unittest.mock import patch, MagicMock, mock_open
from os.path import exists
import builtins

@pytest.fixture
def setup_teardown():
    """Fixture to set up and tear down test environment."""
    test_dir = "test_dir"
    paths_txt = "paths.txt"

    # Setup: ensure the test directories and files are clean
    if exists(test_dir):
        os.rmdir(test_dir)
    if exists(paths_txt):
        os.remove(paths_txt)

    yield test_dir, paths_txt

    # Teardown: clean up the test directories and files
    if exists(test_dir):
        os.rmdir(test_dir)
    if exists(paths_txt):
        os.remove(paths_txt)

@patch('pyplis.inout.urlretrieve')
@patch('pyplis.inout.ZipFile')
@patch('pyplis.inout.get_my_pyplis_dir')
@patch('pyplis.inout.exists')
@patch('pyplis.inout.get_paths_txt')
@patch('pyplis.inout._path_registered')
@patch('builtins.open', new_callable=mock_open)
def test_download_test_data(mock_open, mock_path_registered, mock_get_paths_txt, mock_exists, mock_get_my_pyplis_dir, mock_ZipFile, mock_urlretrieve, setup_teardown):
    test_dir, paths_txt = setup_teardown
    save_dir = test_dir
    mock_get_my_pyplis_dir.return_value = save_dir
    mock_exists.return_value = True
    mock_get_paths_txt.return_value = paths_txt
    mock_path_registered.return_value = False
    
    # Mocking ZipFile object and its methods
    mock_zip = MagicMock()
    mock_ZipFile.return_value = mock_zip

    # Call the function
    result = mod.download_test_data(save_dir)

    # Assertions
    mock_urlretrieve.assert_called_once()
    mock_zip.extractall.assert_called_once_with(save_dir)
    mock_zip.close.assert_called_once()
    assert result == save_dir
    mock_open.assert_called_once_with(paths_txt, "a")
    mock_open().write.assert_called_once_with(f"\n{save_dir}\n")

def test_get_data_search_dirs():
    my_pyplis_dir = os.path.expanduser(os.path.join('~', 'my_pyplis'))
    my_pyplis_dir_exists = True if os.path.exists(my_pyplis_dir) else False
    try:
        result = mod.get_data_search_dirs()
        assert len(result) == 2
        assert os.path.samefile(my_pyplis_dir, result[0])
        assert os.path.samefile(result[1], os.path.join(PYPLIS_INSTALL_ROOT, 'data'))
    finally:
        #cleanup if needed
        if not my_pyplis_dir_exists:
            shutil.rmtree(my_pyplis_dir)

def test_get_data_search_dirs_WITH_ENVVAR(tmpdir, monkeypatch):
    monkeypatch.setenv("PYPLIS_DATADIR", str(tmpdir))

    my_pyplis_dir = os.path.expanduser(os.path.join('~', 'my_pyplis'))
    my_pyplis_dir_exists = True if os.path.exists(my_pyplis_dir) else False
    try:
        result = mod.get_data_search_dirs()
        assert len(result) == 3
        assert os.path.samefile(my_pyplis_dir, result[0])
        assert os.path.samefile(result[1], os.path.join(PYPLIS_INSTALL_ROOT, 'data'))
        assert os.path.samefile(result[2], tmpdir)
    finally:
        #cleanup if needed
        if not my_pyplis_dir_exists:
            shutil.rmtree(my_pyplis_dir)

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

    save_dir = None #mod.download_test_data(tmpdir)
    assert os.path.exists(save_dir)
    dataloc = os.path.join(save_dir, 'pyplis_etna_testdata')
    assert os.path.exists(dataloc)

def test_find_test_data():
    mod.find_test_data()

@pytest.mark.parametrize("cam_id", [
    "ecII", "usgs"
])
def test_get_camera_info(cam_id):
    info = mod.get_camera_info(cam_id)