import numpy as np
from compute_disagreement import main

# Generate a small 3x4 .npy file
test_data = np.array([[1, 2, 1, -1],
                      [3, 2, 3,  3],
                      [1, 1, 2, -1],
                      [3, 3, 3, 3],
                      [2, 2, 3, 3]])

np.save("test_data.npy", test_data)

# Run the main function
main("test_data.npy", "test_data_result.npy")

# Load and print the results
results = np.load("test_data_result.npy")
print("Results:")
print(results)