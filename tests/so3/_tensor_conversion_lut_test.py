# Copyright 2024 The e3x Authors.
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

import pathlib
from typing import Any, Dict
import e3x
from e3x.so3._tensor_conversion_lut import _generate_tensor_conversion_lookup_table
import numpy as np
import pytest


_EXPECTED_LUT: Dict[str, Any] = dict(
    max_degree=2,
    s2t=[
        np.asarray([[1]], dtype=np.float64),
        np.asarray(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [0, 1 / np.sqrt(3), 0, 0, 0, 0],
                [0, 0, 0, 0, 1 / np.sqrt(3), 0],
                [-1 / 3, 0, 0, -1 / 3, 0, 2 / 3],
                [0, 0, 1 / np.sqrt(3), 0, 0, 0],
                [1 / np.sqrt(3), 0, 0, -1 / np.sqrt(3), 0, 0],
            ],
            dtype=np.float64,
        ),
    ],
    t2s=[
        np.asarray([[1]], dtype=np.float64),
        np.asarray(
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [0, 0, -1 / 2, 0, np.sqrt(3) / 2],
                [np.sqrt(3), 0, 0, 0, 0],
                [0, 0, 0, np.sqrt(3), 0],
                [0, 0, -1 / 2, 0, -np.sqrt(3) / 2],
                [0, np.sqrt(3), 0, 0, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=np.float64,
        ),
    ],
)


def assert_lut_is_correct(lut: Dict[str, Any], max_degree: int = 2) -> None:
  assert lut["max_degree"] == max_degree
  assert len(lut["s2t"]) == max_degree + 1
  assert len(lut["t2s"]) == max_degree + 1
  for key in ("s2t", "t2s"):
    for a, b in zip(lut[key], _EXPECTED_LUT[key]):
      assert np.allclose(a, b)


@pytest.fixture(autouse=True)
def create_dummy_lut(tmp_path: pathlib.Path):
  path = tmp_path / "lut.npz"
  with path.open("wb") as f:
    np.savez_compressed(f, max_degree=-1)
  e3x.Config.set_tensor_conversion_cache(path)
  yield  # Cleanup code comes after yield.
  e3x.Config.set_tensor_conversion_cache()  # Reset state.


def test_generate_lookup_table(max_degree: int = 2) -> None:
  lut = _generate_tensor_conversion_lookup_table(max_degree)
  assert_lut_is_correct(lut)


def test_incremental_generation(max_degree: int = 2) -> None:
  lut = None
  for l in range(max_degree + 1):
    lut = _generate_tensor_conversion_lookup_table(l)
  assert_lut_is_correct(lut)
