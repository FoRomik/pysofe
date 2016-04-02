# this file is used to make pysofe available for tests without the
# need to install it

import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, '..')))

import pytest
