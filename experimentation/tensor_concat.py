import torch

# Example tensors
tensor1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # Shape (2, 4)
tensor2 = torch.tensor([[9, 10, 11, 12, 13], [14, 15, 16, 17, 18]])  # Shape (2, 5)

# Indices indicating the positions of columns in the original tensor
ind1 = [1, 4, 6, 7]
ind2 = [0, 2, 3, 5, 8]

# Combine indices and tensors
combined_indices = ind1 + ind2
combined_tensors = torch.cat((tensor1, tensor2), dim=1)

# Use scatter to place the columns in the correct order
sorted_indices = torch.argsort(torch.tensor(combined_indices))
result_tensor = combined_tensors[:, sorted_indices]

print("Combined indices:" + str(combined_indices))
print("Sorted indices:" + str(sorted_indices))
print("Final tensor:")
print(result_tensor)
