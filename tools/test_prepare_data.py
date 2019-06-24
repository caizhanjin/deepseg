from .prepare_data import PrepareData
import unittest


class TestPrepare(unittest.TestCase):
    def testPrepareData(self):
        prepare_data = PrepareData()
        prepare_data.prepare_data()


if __name__ == '__main__':
    unittest.main()
