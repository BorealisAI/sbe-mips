# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations
import numpy as np
import pandas as pd

METHOD_ORDER = ('dm', 'mips', 'mips_with_reward_supervision', 'sbe_mips')


def require(value, message):
    if not value:
        raise ValueError(message)


def task_methods(value):
    methods = tuple(value.split(';')) if isinstance(value, str) else tuple(value)
    require(set(methods) <= set(METHOD_ORDER) and len(set(methods)) == len(methods), 'Unknown/duplicate method')
    require({'dm', 'mips_with_reward_supervision', 'sbe_mips'} <= set(methods), 'Missing required methods')
    return tuple(m for m in METHOD_ORDER if m in methods)