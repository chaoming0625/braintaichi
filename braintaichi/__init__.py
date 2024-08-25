# Copyright 2024- BrainPy Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-


__all__ = [
  '__version__',
  '__minimal_taichi_version__',
]

__version__ = "0.0.1"
__minimal_taichi_version__ = (1, 7, 2)

import ctypes
import os
import platform
import sys

with open(os.devnull, 'w') as devnull:
  os.environ["TI_LOG_LEVEL"] = "error"
  old_stdout = sys.stdout
  sys.stdout = devnull
  try:
    import taichi as ti  # noqa
  except ModuleNotFoundError:
    raise ModuleNotFoundError(
      f'We need taichi=={__minimal_taichi_version__}. '
      f'Currently you can install taichi=={__minimal_taichi_version__} through:\n\n'
      f'> pip install taichi=={".".join(map(str, __minimal_taichi_version__))} -U'
      # '> pip install -i https://pypi.taichi.graphics/simple/ taichi-nightly'
    )
  finally:
    sys.stdout = old_stdout
del old_stdout, devnull

# check Taichi version
if ti.__version__ != __minimal_taichi_version__:
  raise RuntimeError(
    f'We need taichi=={__minimal_taichi_version__}. '
    f'Currently you can install taichi=={__minimal_taichi_version__} through:\n\n'
    f'> pip install taichi=={".".join(map(str, __minimal_taichi_version__))} -U'
    # '> pip install -i https://pypi.taichi.graphics/simple/ taichi-nightly'
  )

# update Taichi runtime and C api
taichi_path = ti.__path__[0]
taichi_c_api_install_dir = os.path.join(taichi_path, '_lib', 'c_api')
os.environ.update({'TAICHI_C_API_INSTALL_DIR': taichi_c_api_install_dir,
                   'TI_LIB_DIR': os.path.join(taichi_c_api_install_dir, 'runtime')})

# link the Taichi C api
if platform.system() == 'Windows':
  dll_path = os.path.join(os.path.join(taichi_c_api_install_dir, 'bin/'), 'taichi_c_api.dll')
  try:
    ctypes.CDLL(dll_path)
  except OSError:
    raise OSError(f'Can not find {dll_path}')
  del dll_path
elif platform.system() == 'Linux':
  so_path = os.path.join(os.path.join(taichi_c_api_install_dir, 'lib/'), 'libtaichi_c_api.so')
  try:
    ctypes.CDLL(so_path)
  except OSError:
    raise OSError(f'Can not find {so_path}')
  del so_path

del os, sys, platform, ti, ctypes, taichi_path, taichi_c_api_install_dir

from ._base import *
from ._base import __all__ as _base_all
from ._ad_support import *
from ._ad_support import __all__ as _ad_support_all
from ._batch_utils import *
from ._batch_utils import __all__ as _batch_utils_all
from ._event_csrmm import *
from ._event_csrmm import __all__ as _event_csrmm_all
from ._event_csrmv import *
from ._event_csrmv import __all__ as _event_csrmv_all
from ._jit_csrmv import *
from ._jit_csrmv import __all__ as _jit_csrmv_all
from ._jit_event_csrmv import *
from ._jit_event_csrmv import __all__ as _jit_event_csrmv_all
from ._sparse_coomv import *
from ._sparse_coomv import __all__ as _sparse_coomv_all
from ._sparse_csrmm import *
from ._sparse_csrmm import __all__ as _sparse_csrmm_all
from ._sparse_csrmv import *
from ._sparse_csrmv import __all__ as _sparse_csrmv_all
from ._taichi_rand import *
from ._taichi_rand import __all__ as _taichi_rand_all

__all__ = (__all__ + _base_all + _ad_support_all + _batch_utils_all + _event_csrmm_all +
           _event_csrmv_all + _jit_csrmv_all + _jit_event_csrmv_all + _sparse_coomv_all +
           _sparse_csrmm_all + _sparse_csrmv_all + _taichi_rand_all)

del (_base_all, _ad_support_all, _batch_utils_all, _event_csrmm_all, _event_csrmv_all,
     _jit_csrmv_all, _jit_event_csrmv_all, _sparse_coomv_all, _sparse_csrmm_all,
     _sparse_csrmv_all, _taichi_rand_all)
