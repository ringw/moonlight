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
"""Command-line engine configuration. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from moonlight import engine
from moonlight.glyphs import saved_classifier_fn

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'patch_saved_model_dir', None,
    'Path to the patch saved model. Defaults to the included NN model.')


def create_engine():
  return engine.OMREngine(
      saved_classifier_fn.build_classifier_fn(FLAGS.patch_saved_model_dir))
