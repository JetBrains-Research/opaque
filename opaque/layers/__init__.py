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
Per-sample gradient computation for supported layer types.

Currently only supports nn.Linear, which is sufficient for LoRA fine-tuning.
"""

from .linear import compute_linear_grad_sample, compute_linear_norm_sample

__all__ = [
    "compute_linear_grad_sample",
    "compute_linear_norm_sample",
]
