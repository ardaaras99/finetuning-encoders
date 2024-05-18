import torch


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def move_device(device: torch.device, *args):
    return [x.to(device) for x in args]


def modify_tensor(tensor, percentage):
    # Ensure the percentage is between 0 and 100
    if not (0 <= percentage <= 100):
        raise ValueError("Percentage must be between 0 and 100")

    # Find the initial number of ones (m)
    m = torch.sum(tensor).item()

    # Calculate the number of ones to keep
    keep_count = int((percentage / 100) * m)

    # Create a new tensor to modify
    new_tensor = torch.zeros_like(tensor)

    # Set the first keep_count elements to 1
    new_tensor[:keep_count] = 1

    return new_tensor


# Example usage (commented out to prevent execution)
# tensor = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
# percentage = 50
# modified_tensor = modify_tensor(tensor, percentage)
# print(modified_tensor)
