# Just need to remove the "_cleaned" suffix from the file names.
import os

filepath = "datasets/cleaned"

suffix = "_cleaned.npy"
extension = ".npy"

for filename in os.listdir(filepath):
    if filename.endswith(suffix):
        os.rename(os.path.join(filepath, filename), os.path.join(filepath, filename[:-len(suffix)] + extension))