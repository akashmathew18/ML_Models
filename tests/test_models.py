"""Test cases for model training and predictions."""

import unittest
import numpy as np
from utils.data_preprocessing import preprocess_data
from utils.validation import validate_input

class TestModelUtils(unittest.TestCase):
    def test_data_preprocessing(self):
        # Test basic preprocessing
        data = {"feature1": 1.0, "feature2": 2.0}
        X, error = preprocess_data(data)
        self.assertIsNone(error)
        self.assertEqual(X.shape, (1, 2))
        
        # Test with invalid data
        data = {"feature1": "invalid"}
        with self.assertRaises(ValueError):
            preprocess_data(data)

    def test_input_validation(self):
        # Test SLR validation
        valid_slr = {"training_hours": 10}
        is_valid, data = validate_input(valid_slr, "SLR")
        self.assertTrue(is_valid)
        
        invalid_slr = {"training_hours": -1}
        is_valid, error = validate_input(invalid_slr, "SLR")
        self.assertFalse(is_valid)
        
        # Test MLR validation
        valid_mlr = {
            "training_hours": 10,
            "fitness_level": 80,
            "past_performance": 75
        }
        is_valid, data = validate_input(valid_mlr, "MLR")
        self.assertTrue(is_valid)
        
        invalid_mlr = {
            "training_hours": 10,
            "fitness_level": 150,  # Invalid value
            "past_performance": 75
        }
        is_valid, error = validate_input(invalid_mlr, "MLR")
        self.assertFalse(is_valid)

if __name__ == '__main__':
    unittest.main()
