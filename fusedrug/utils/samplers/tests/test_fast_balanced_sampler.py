import unittest
from fusedrug.utils.samplers import FastBalancedSampler
from collections import Counter

class TestFastBalancedSampler(unittest.TestCase):
    def test_fast_balanced_sampler(self):        
        sampler = FastBalancedSampler(
                datasets_lengths = [10,20,30],
                minibatch_pattern = [3,2,1],
                shuffle = False,
                #shuffle_within_minibatch:bool=None,
                yield_minibatch = True,
                epoch_minibatches_count_mode = 'see_all',
            )

        self.assertEqual(len(sampler),30)
        
        count_freq = Counter()

        for mb in sampler:
            for index in mb:
                if index<10:
                    count_freq['class_a'] += 1
                elif index<10+20:
                    count_freq['class_b'] += 1
                elif index<10+20+30:
                    count_freq['class_c'] += 1
                else:
                    assert False
        
        self.assertEqual(count_freq['class_a'], count_freq['class_c']*3)
        self.assertEqual(count_freq['class_b'], count_freq['class_c']*2)

            
if __name__ == '__main__':
    unittest.main()






