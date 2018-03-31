# coding=utf-8
import unittest
from TweetClassifier import classify


class CalculatorTest(unittest.TestCase):
    def test_positive_classification(self):
        # Set Up
        expected = "positive"
        text = "Lula é muito gente boa"

        actual = classify(text)

        # Assert
        self.assertEqual(actual, expected, msg="We expected {0} but received {1}".format(expected, actual))

    def test_negative_classification(self):
        # Set Up
        expected = "negative"
        text = "Lula é ladrão fdp"

        actual = classify(text)

        # Assert
        self.assertEqual(actual, expected, msg="We expected {0} but received {1}".format(expected, actual))

if __name__ == '__main__':
    unittest.main()