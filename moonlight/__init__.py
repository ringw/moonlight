# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Moonlight Optical Music Recognition (OMR)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from moonlight import engine_flags

create_engine = engine_flags.create_engine

_STATIC_ENGINE = None


def run(input_pngs, output_notesequence=False):
  """Converts input PNGs into a `Score` message.

  Args:
    input_pngs: A list of PNG filenames to process.
    output_notesequence: Whether to return a NoteSequence, as opposed to a
        Score containing Pages with Glyphs.

  Returns:
    A NoteSequence message, or a Score message holding Pages for each input
        image (with their detected Glyphs).
  """
  if not _STATIC_ENGINE:
    _STATIC_ENGINE = engine_flags.create_engine()

  return _STATIC_ENGINE.run(input_pngs, output_notesequence)
