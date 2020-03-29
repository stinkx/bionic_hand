import unittest
from Model import get_model


class MyTestCase(unittest.TestCase):
    def test_model_name(self):
        self.assertRaises(ValueError, get_model, 0, 100, 5, 128, 3, 1, 0.5, True)
        self.assertRaises(ValueError, get_model, "RNNN", 100, 5, 128, 3, 1, 0.5, True)

    def test_input_size(self):
        self.assertRaises(ValueError, get_model, "RNN", "0", 5, 128, 3, 1, 0.5, True)
        self.assertRaises(ValueError, get_model, "RNN", 2.1, 5, 128, 3, 1, 0.5, True)
        self.assertRaises(ValueError, get_model, "RNN", -1, 5, 128, 3, 1, 0.5, True)

    def test_output_size(self):
        self.assertRaises(ValueError, get_model, "RNN", 100, "0", 128, 3, 1, 0.5, True)
        self.assertRaises(ValueError, get_model, "RNN", 100, 2.1, 128, 3, 1, 0.5, True)
        self.assertRaises(ValueError, get_model, "RNN", 100, -1, 128, 3, 1, 0.5, True)

    def test_hidden_size(self):
        self.assertRaises(ValueError, get_model, "RNN", 100, 5, "0", 3, 1, 0.5, True)
        self.assertRaises(ValueError, get_model, "RNN", 100, 5, 2.1, 3, 1, 0.5, True)
        self.assertRaises(ValueError, get_model, "RNN", 100, 5, -1, 3, 1, 0.5, True)

    def test_batch_size(self):
        self.assertRaises(ValueError, get_model, "RNN", 100, 5, 128, "0", 1, 0.5, True)
        self.assertRaises(ValueError, get_model, "RNN", 100, 5, 128, 2.1, 1, 0.5, True)
        self.assertRaises(ValueError, get_model, "RNN", 100, 5, 128, -1, 1, 0.5, True)

    def test_num_layers(self):
        self.assertRaises(ValueError, get_model, "RNN", 100, 5, 128, 3, "0", 0.5, True)
        self.assertRaises(ValueError, get_model, "RNN", 100, 5, 128, 3, 2.1, 0.5, True)
        self.assertRaises(ValueError, get_model, "RNN", 100, 5, 128, 3, -1, 0.5, True)
        self.assertRaises(ValueError, get_model, "RNN", 100, 5, 128, 3, 11, 0.5, True)

    def test_dropout(self):
        self.assertRaises(ValueError, get_model, "RNN", 100, 5, 128, 3, 1, "0", True)
        self.assertRaises(ValueError, get_model, "RNN", 100, 5, 128, 3, 1, 1, True)
        self.assertRaises(ValueError, get_model, "RNN", 100, 5, 128, 3, 1, -0.1, True)

    def test_bias(self):
        self.assertRaises(ValueError, get_model, "RNN", 100, 5, 128, 3, 1, 0.5, 0)
        self.assertRaises(ValueError, get_model, "RNN", 100, 5, 128, 3, 1, 0.5, "0")


if __name__ == '__main__':
    unittest.main()
