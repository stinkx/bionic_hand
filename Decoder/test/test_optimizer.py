import unittest
from Optimizer import get_optimizer
from Model import get_model
import Parameter


class OptimizerTest(unittest.TestCase):
    def setUp(self):
        self.net = get_model(Parameter.network, 180, 22, Parameter.hidden_size, Parameter.batch_size,
                             Parameter.num_layers, Parameter.dropout, Parameter.bias)

    def test_optimizer_name(self):
        self.assertRaises(ValueError, get_optimizer, "SGDD", self.net.parameters(), 0.001, 0.9, 0.9)
        self.assertRaises(ValueError, get_optimizer, 0, self.net.parameters(), 0.001, 0.9, 0.9)

    def test_learning_rate(self):
        self.assertRaises(ValueError, get_optimizer, "SGD", self.net.parameters(), "0", 0.9, 0.9)
        self.assertRaises(ValueError, get_optimizer, "SGD", self.net.parameters(), -1.3, 0.9, 0.9)
        self.assertRaises(ValueError, get_optimizer, "SGD", self.net.parameters(), 2.1, 0.9, 0.9)

    def test_weight_decay(self):
        self.assertRaises(ValueError, get_optimizer, "SGD", self.net.parameters(), 0.001, "0", 0.9)
        self.assertRaises(ValueError, get_optimizer, "SGD", self.net.parameters(), 0.001, -1.3, 0.9)
        self.assertRaises(ValueError, get_optimizer, "SGD", self.net.parameters(), 0.001, 2.1, 0.9)

    def test_momentum(self):
        self.assertRaises(ValueError, get_optimizer, "SGD", self.net.parameters(), 0.001, 0.9, "0")
        self.assertRaises(ValueError, get_optimizer, "SGD", self.net.parameters(), 0.001, 0.9, -1.3)
        self.assertRaises(ValueError, get_optimizer, "SGD", self.net.parameters(), 0.001, 0.9, 2.1)


if __name__ == '__main__':
    unittest.main()
