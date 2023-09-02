import numpy as np

# sample input
annotations = np.array([[1, 1, -1, 0, -1],
                        [0, 1, 5, 23, 23],
                        [0, 0, -1, 1, 1]])

# add one to all elements so -1 becomes 0
annotations += 1

# get the bincount along axis 1 for the unique values
# equivalent to bincounts = np.array([np.bincount(row, minlength=np.max(annotations)+1) for row in annotations])
bincounts = np.apply_along_axis(np.bincount, axis=1, arr=annotations, minlength=np.max(annotations)+1)
print(bincounts)

# set the count for 0 to 0
bincounts[:, 0] = 0

# get the majority vote, and subtract 1 so 0 becomes -1 again
maj_vote = np.argmax(bincounts, axis=1) - 1

print(maj_vote)