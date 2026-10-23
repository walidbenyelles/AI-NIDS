
import unittest
from pathlib import Path

import pandas as pd
from joblib import load


class TestAINIDS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dossier = Path(__file__).resolve().parents[1]

        artifact = load(
            dossier / "models" / "ai_nids_random_forest.joblib"
        )

        cls.model = artifact["model"]
        cls.features = artifact["features"]

        # Connexions fictives pour tester le programme
        cls.X = pd.DataFrame([
            [0.12, 500, 1200, 12, 15],
            [0.001, 60, 0, 2, 0]
        ], columns=cls.features)

    def test_features(self):
        self.assertEqual(
            self.features,
            ["dur", "sbytes", "dbytes", "spkts", "dpkts"]
        )

    def test_predictions(self):
        predictions = self.model.predict(self.X)

        self.assertEqual(len(predictions), 2)

        for prediction in predictions:
            self.assertIn(prediction, [0, 1])

    def test_probabilities(self):
        scores = self.model.predict_proba(self.X)

        self.assertEqual(scores.shape, (2, 2))
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())

        for ligne in scores:
            self.assertAlmostEqual(sum(ligne), 1.0)


if __name__ == "__main__":
    unittest.main()
