#!/usr/bin/env python3
# Copyright 2024 Opaque Contributors
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

"""
Grad sample computation infrastructure.

Supports two modes:
- hooks: PyTorch hooks-based (default, stable)
- functorch: functorch-based (experimental, faster in some cases)
"""

from .controller import GradSampleController
from .utils import wrap_model

__all__ = [
    "GradSampleController",
    "wrap_model",
]
