"""Tests for rotation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
from moonlight import image
from moonlight.preprocessing import rotation


class RotationTest(tf.test.TestCase):

  def testRotation(self):
    filename = os.path.join(tf.resource_loader.get_data_files_path(),
                            '../testdata/IMSLP00747-000.png')
    img = image.decode_music_score_png(tf.read_file(filename))
    angle = 0.01
    rotated_img = rotation._rotate_white_background(img, angle)
    rot = rotation.Rotation(rotated_img)
    with self.test_session():
      detected_angle = rot._get_rotation().eval()
    self.assertAlmostEqual(detected_angle, -angle, places=2)


if __name__ == '__main__':
  tf.test.main()
