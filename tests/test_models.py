
# test_notebook.py
import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join('../src')))

logging.basicConfig(level=logging.DEBUG)

if not os.path.exists("../src/linear_regression/Auto.csv"):
    os.system("cp ../datasets/Auto.csv ../src/linear_regression/Auto.csv")

@pytest.mark.parametrize("notebook_path", ["../src/linear_regression/linear_regression.ipynb"])
def test_run_notebook(notebook_path):
    with open(notebook_path, "r") as f:
        notebook_content = f.read()

    nb = nbformat.reads(notebook_content, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")  # Set appropriate kernel name

    try:
        ep.preprocess(nb, {"metadata": {"path": ""}}) 
    except Exception as e:
        msg = f"Error executing the notebook {notebook_path}."
        msg += f"\n\nError: {str(e)}"
        raise AssertionError(msg)
    

import subprocess
def test_run_model():
    result = subprocess.run(['python', '../src/linear_regression/main.py'], capture_output=True, text=True)
    assert result.returncode == 0

import re
def test_model_loss():
    result = subprocess.run(['python', '../src/linear_regression/main.py'], capture_output=True, text=True)
    pattern = re.compile(r"The final cost value is (\d+\.\d+)")
    result = pattern.search(result.stdout)
    assert float(result.group(1)) < 25
