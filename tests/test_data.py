import unittest
from forge.data import CharDataset
class DataTests(unittest.TestCase):
    def test_batch_and_metadata(self):
        d=CharDataset('abcabcabcabcabcabc',2,.2); x,y=d.batch('train',3,0,42)
        self.assertEqual(x.shape,(3,2)); self.assertEqual(y.shape,(3,2)); self.assertEqual(d.metadata()['characters'],3)
    def test_bad_input(self):
        with self.assertRaises(ValueError): CharDataset('a',2)
