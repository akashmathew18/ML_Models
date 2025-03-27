"""Test cases for Flask routes."""

import unittest
from app import app

class TestFlaskRoutes(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_model_routes(self):
        routes = ['slr', 'mlr', 'lr', 'pr', 'knn']
        for route in routes:
            response = self.app.get(f'/{route}')
            self.assertEqual(response.status_code, 200)

    def test_prediction_routes(self):
        # Test SLR prediction
        slr_data = {'training_hours': 10}
        response = self.app.post('/predict_slr', data=slr_data)
        self.assertEqual(response.status_code, 200)

        # Test MLR prediction
        mlr_data = {
            'training_hours': 10,
            'fitness_level': 80,
            'past_performance': 75
        }
        response = self.app.post('/predict_mlr', data=mlr_data)
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
