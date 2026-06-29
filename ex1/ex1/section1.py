import numpy as np

matrix = np.random.randint(0,10,size=(5,6))

print("original matrix")
print(matrix)

row_5 = matrix[4, : ]

sorted_indices = np.argsort(row_5)
target_column_index = sorted_indices[2]

matrix[4, target_column_index] = 10

print("\nModified Matrix (Row 5, 3rd smallest value replaced by 10):")
print(matrix)
