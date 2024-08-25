# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

# This module implements how to register the primitive operators in BrainTaichi.
# It includes the following parts:
#   1. Register the primitive operators in BrainTaichi.
#   2. Define the gradient operators for the primitive operators.
#   3. Define the custom operators for the primitive operators.
#   4. Define the custom operators for the primitive operators in XLA.


from ._ad_support import *
from ._ad_support import __all__ as __ad_support_all__
from ._batch_utils import *
from ._batch_utils import __all__ as __batch_utils_all__
from ._xla_custom_op import *
from ._xla_custom_op import __all__ as __xla_custom_op_all__

__all__ = __ad_support_all__ + __batch_utils_all__ + __xla_custom_op_all__
