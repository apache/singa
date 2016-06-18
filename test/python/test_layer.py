import sys
import os
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/python'))

import layer


class TestPythonLayer(unittest.TestCase):
    def setUp(self):
        self.w = {'init': 'xavier', 'regularizer': 1e-4}
        self.b = {'init': 'constant', 'value': 0}

    def test_convolution_layer(self):
        in_sample_shape = (3, 224, 224)
        conv = layer.Conv2D(64, 3, 1, 'same', W_specs=self.w, b_specs=self.b,
                           input_sample_shape=in_sample_shape)
        out_sample_shape = conv.get_output_sample_shape()
        self.assertEqual(out_sample_shape, (64, 224, 224),
                         'incorrect output sample size')


if __name__ == '__main__':
    unittest.main()
