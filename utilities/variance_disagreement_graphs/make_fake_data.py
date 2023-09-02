'''
Just making small fake data for debugging / testing
'''

import numpy as np
annotations = np.array([[1, 1, -1, 0, -1],
                        [0, 1, 5, 23, 23],
                        [0, 0, -1, 1, 1]])
annotations2 = np.array([[1, 1, 1, 0, 5],
                        [0, 1, 5, 23, 23],
                        [0, 0, 0, 1, 1]])

np.save('fake_annotations.npy', annotations)
np.save('fake_annotations2.npy', annotations2)