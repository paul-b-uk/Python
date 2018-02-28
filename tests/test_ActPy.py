import unittest
import os
import sys
import numpy as np
#import xmlrunner


# The 3 line below just add the source directory to the path so we can import
os.chdir( '..' ) 
curr_dir = os.getcwd()
sys.path.append(curr_dir) 


from source import ActPy as act


class TestActPy(unittest.TestCase):

    def test_dave(self):
        self.assertEqual(1,1) 

    def test_adder(self):
        self.assertEqual(1,1)

    def test_sub1(self):
        self.assertEqual(3,3)
        
    def test_aggregate_claims1(self):
        indiv_claims    = np.array([10.0, 9.0, 3.4])
        claims_per_year = np.array([0, 3])
        expected        = np.array([0.0, 22.4])        
        actual          = act.aggregate_claims(indiv_claims, claims_per_year)
        self.assertEqual(expected.tolist(), actual.tolist())

if __name__ == '__main__':    
    #unittest.main()
    unittest.main(exit=False)
    #unittest.main(testRunner=xmlrunner.XMLTestRunner(output="python_unittests_xml"))
    