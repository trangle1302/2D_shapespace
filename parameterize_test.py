import os
import numpy as np
import parameterize
from skimage import morphology as skmorpho
import matplotlib.pyplot as plt

w = 100
mem = np.zeros((w, w, w), dtype=np.uint8)
mem[20:80, 20:80, 20:80] = 1
nuc = np.zeros((w, w, w), dtype=np.uint8)
nuc[40:60, 40:60, 30:50] = 1

# Create an FP signal located in the top half of the cell and outside the
# nucleus:
gfp = np.random.rand(w ** 3).reshape(w, w, w)
gfp[mem == 0] = 0
gfp[:, w // 2 :] = 0
gfp[nuc > 0] = 0

# Vizualize a center xy cross-section of our cell:
plt.imshow((mem + nuc)[w // 2], cmap="gray")
plt.imshow(gfp[w // 2], cmap="gray", alpha=0.25)
plt.axis("off")

# Creating a small round
mem_round = skmorpho.ball(w // 3)  # radius of our round cell
nuc_round = skmorpho.ball(w // 3)  # radius of our round nucleus
# Erode the nucleus so it becomes smaller than the cell
nuc_round = skmorpho.binary_erosion(nuc_round, selem=np.ones((20, 20, 20))).astype(
    np.uint8
)

# Vizualize a center xy cross-section of our round cell:
plt.imshow((mem_round + nuc_round)[w // 3], cmap="gray")
plt.axis("off")


#
# Use xxxparam to expand both cell and nuclear shapes in terms of fft
coords, coeffs_centroid = parameterize.parameterize_image_coordinates(
    seg_mem=mem,
    seg_nuc=nuc,
    lmax=16,  # Degree of the spherical harmonics expansion
    nisos=[32, 32],  # Number of interpolation layers
)
coeffs_mem, centroid_mem, coeffs_nuc, centroid_nuc = coeffs_centroid

# Run the cellular mapping to create a parameterized intensity representation
# for the FP image:
gfp_representation = parameterize.cellular_mapping(
    coeffs_mem=coeffs_mem,
    centroid_mem=centroid_mem,
    coeffs_nuc=coeffs_nuc,
    centroid_nuc=centroid_nuc,
    nisos=[32, 32],
    images_to_probe=[("gfp", gfp)],
).data.squeeze()

# The FP image is now encoded into a representation of its shape:
print(gfp_representation.shape)


# Parameterize the coordinates of the round cells
coords_round, _ = parameterize.parameterize_image_coordinates(
    seg_mem=mem_round, seg_nuc=nuc_round, lmax=16, nisos=[32, 32]
)

# Now we are ready to morph the FP image into our round cell:
gfp_morphed = parameterize.morph_representation_on_shape(
    img=mem_round + nuc_round,
    param_img_coords=coords_round,
    representation=gfp_representation,
)
# Visualize the morphed FP image:
plt.imshow((mem_round + nuc_round)[w // 3], cmap="gray")
plt.imshow(gfp_morphed[w // 3], cmap="gray", alpha=0.25)
plt.axis("off")
