'''Check to make sure every row and column has some data'''
import numpy as np

annotations_location = "../datasets/cleaned/2xSBIC_annotations.npy"
fixed_annotations_location = "../datasets/cleaned/Fixed2xSBIC_annotations.npy"
annotations = np.load(annotations_location, allow_pickle=True)
sum_annotations = annotations.copy()
sum_annotations[sum_annotations != -1] = 1
sum_annotations[sum_annotations == -1] = 0

print("Checking...")
good = True
if not np.all(np.sum(sum_annotations, axis=0) > 0):
    print("Some columns have no data")
    # remove those columns and save the new dataset
    print("Removing those columns...")
    annotations = annotations[:, np.sum(sum_annotations, axis=0) > 0]
    np.save(fixed_annotations_location, annotations, allow_pickle=True)
    print(f"Saved to {fixed_annotations_location}")
    good = False

if not np.all(np.sum(sum_annotations, axis=1) > 0):
    print("Some rows have no data")
    good = False
    # This is not implemented yet because it would require modifying the texts as well
    raise NotImplementedError("This is not implemented yet - you will need to fix this on your own")

if good:
    print("All good!")