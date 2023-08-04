import numpy as np

def colocalization_quotient(image1, image2):
    # Ensure both images have the same dimensions
    assert image1.shape == image2.shape, "Images must have the same dimensions."

    # Convert images to binary (0 or 1)
    image1_binary = (image1 > 0).astype(np.uint8)
    image2_binary = (image2 > 0).astype(np.uint8)

    # Calculate the number of colocalized pixels
    colocalized_pixels = np.sum(image1_binary & image2_binary)

    # Calculate the number of total pixels in channel 1
    total_pixels_channel1 = np.sum(image1_binary)

    # Calculate the colocalization quotient
    colocalization_quotient = colocalized_pixels / total_pixels_channel1

    return colocalization_quotient

def local_colocalization_quotient(image1, image2, window_size, stride):
    # Ensure both images have the same dimensions
    assert image1.shape == image2.shape, "Images must have the same dimensions."
    height, width = image1.shape

    # Initialize an empty array to store the local colocalization quotients
    local_quotients = np.zeros((height, width), dtype=np.float)

    # Iterate over the image with the sliding window
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            # Extract the patch from both images
            patch1 = image1[y:y+window_size, x:x+window_size]
            patch2 = image2[y:y+window_size, x:x+window_size]

            # Compute the colocalization quotient for the patch
            local_quotients[y:y+window_size, x:x+window_size] = colocalization_quotient(patch1, patch2)

    return local_quotients