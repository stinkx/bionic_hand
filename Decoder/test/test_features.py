import unittest
from Features import calc_features


class FeaturesTest(unittest.TestCase):
    def test_feature_set(self):
        self.assertRaises(ValueError, calc_features, 0, 0, 0, 0, 0, ['mean_value'], 2000.)
        self.assertRaises(ValueError, calc_features, 0, 0, 0, 0, ['zero_crossings', 'simple_square_integral', 0], ['mean_value'], 2000.)
        self.assertRaises(ValueError, calc_features, 0, 0, 0, 0, ['zero_crossings', 'simple_square_integral', '0'], ['mean_value'], 2000.)
        self.assertRaises(ValueError, calc_features, 0, 0, 0, 0, ['zero_crossings', 'simple_square_integral', 'zero_crossings'], ['mean_value'], 2000.)

    def test_feature_set_im(self):
        self.assertRaises(ValueError, calc_features, 0, 0, 0, 0, ['zero_crossings', 'simple_square_integral', 'integrated_emg'], 0, 2000.)
        self.assertRaises(ValueError, calc_features, 0, 0, 0, 0, ['zero_crossings', 'simple_square_integral', 'integrated_emg'], ['mean_value', 0], 2000.)
        self.assertRaises(ValueError, calc_features, 0, 0, 0, 0, ['zero_crossings', 'simple_square_integral', 'integrated_emg'], ['mean_value', '0'], 2000.)
        self.assertRaises(ValueError, calc_features, 0, 0, 0, 0, ['zero_crossings', 'simple_square_integral', 'integrated_emg'], ['mean_value', 'mean_value'], 2000.)

    def test_sample_frequency(self):
        self.assertRaises(ValueError, calc_features, 0, 0, 0, 0, ['zero_crossings', 'simple_square_integral', 'integrated_emg'], ['mean_value'], "2000")
        self.assertRaises(ValueError, calc_features, 0, 0, 0, 0, ['zero_crossings', 'simple_square_integral', 'integrated_emg'], ['mean_value'], 2000)
        self.assertRaises(ValueError, calc_features, 0, 0, 0, 0, ['zero_crossings', 'simple_square_integral', 'integrated_emg'], ['mean_value'], -2000.)


if __name__ == '__main__':
    unittest.main()
