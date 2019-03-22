# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
sys.path.append("../common")

from builtins import range
from future.utils import iteritems
import unittest
import numpy as np
import infer_util as iu
import test_util as tu
from tensorrtserver.api import *
import os

np_dtype_string = np.dtype(object)

class InferReshapeTest(unittest.TestCase):

    def _full_reshape(self, dtype, shapes):
        # 'shapes' is list of shapes, one for each input.

        # For validation assume any shape can be used...
        if tu.validate_for_tf_model(dtype, dtype, dtype, shapes[0], shapes[0], shapes[0]):
            # model that supports batching
            for bs in (1, 8):
                iu.infer_zero(self, 'graphdef', bs, dtype, shapes)
                iu.infer_zero(self, 'savedmodel', bs, dtype, shapes)
            # model that does not support batching
            iu.infer_zero(self, 'graphdef_nobatch', 1, dtype, shapes)
            iu.infer_zero(self, 'savedmodel_nobatch', 1, dtype, shapes)

#        if tu.validate_for_c2_model(dtype, dtype, dtype, shapes[0], shapes[0], shapes[0]):
            # model that supports batching
#            for bs in (1, 8):
#                iu.infer_zero(self, 'netdef', bs, dtype, shapes)
            # model that does not support batching
#            iu.infer_zero(self, 'netdef_nobatch', 1, dtype, shapes)

    def test_ff1(self):
        self._full_reshape(np.float32, ([1,],))
    def test_ff3(self):
        self._full_reshape(np.float32, ([1,],[1,],[1,]))

    def test_hh1(self):
        self._full_reshape(np.float16, ([8,],))
    def test_hh3(self):
        self._full_reshape(np.float16, ([8,],[8,],[8,]))

if __name__ == '__main__':
    unittest.main()
