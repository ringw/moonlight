"""Corrects music score image rotation.

Uses the method described in [1] for rotation removal. The staff lines are the
predominant component of the 2D FFT of a music score image, because consecutive
staff lines repeat at a constant interval (the staffline distance). These
entries with a high value in the FFT are normally vertically centered around the
origin, but are rotated along with the image.

[1] R. Lobb, T. Bell, and D. Bainbridge. Fast capture of sheet music for an
    agile digital music library. In Proceedings of the 6th International
    Conference on Music Information Retrieval, pages 145-152. ISMIR, 2005.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# internal imports
import tensorflow as tf

from moonlight.staves import staffline_distance


class Rotation(object):

  def __init__(self, image, threshold=127):
    self.original_image = image
    self.threshold = threshold
    self.image = self.rotate()

  def rotate(self):
    if self.original_image.dtype != tf.uint8:
      raise ValueError('uint8 image required. Got: %s' % (image,))
    with tf.name_scope('Rotation'):
      self.angle = self._get_rotation()
      return _rotate_white_background(self.original_image, self.angle)

  def _get_rotation(self):
    fft = tf.abs(
        tf.fft2d(
            tf.cast(tf.less(self.original_image, self.threshold),
                    tf.complex64)))
    distance, _ = staffline_distance.estimate_staffline_distance_and_thickness(
        self.original_image, self.threshold)
    shape = tf.shape(self.original_image)
    image_x, image_y = tf.meshgrid(
        tf.to_float(tf.range(shape[1])), tf.to_float(tf.range(shape[0])))
    # TODO(ringwalt): Handle the other quadrants correctly.
    pixel_distance = tf.square(image_y) + tf.square(image_x)
    mask = tf.logical_and(
        tf.greater_equal(pixel_distance,
                         tf.to_float(distance) / 2.),
        tf.less_equal(pixel_distance,
                      tf.to_float(distance) * 2.5))
    fft = tf.where(mask, fft, tf.zeros_like(fft))
    max_component = tf.argmax(fft)
    return tf.atan2(
        tf.to_float(max_component[0]), tf.to_float(max_component[1]))

def _rotate_white_background(image, angle):
  assert image.dtype == tf.uint8, "uint8 image required, got: %s" % image
  # TODO(ringw): Add a fill_value argument to rotate(), to avoid the "255 -".
  with tf.name_scope('rotate_white_background'):
    return 255 - tf.contrib.image.rotate(255 - image, angle, interpolation='BILINEAR')
