import unittest
from Loss import get_Loss


class LossTest(unittest.TestCase):

    def test_loss_name(self):
        self.assertRaises(ValueError, get_Loss, "L1Los")
        self.assertRaises(ValueError, get_Loss, 0)


if __name__ == '__main__':
    unittest.main()
