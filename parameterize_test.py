import os
import parameterize

coords_round, _ = cytoparam.parameterize_image_coordinates(
    seg_mem=mem_round,
    seg_nuc=nuc_round,
    lmax=16,
    nisos=[32, 32]
)

# Now we are ready to morph the FP image into our round cell:
gfp_morphed = cytoparam.morph_representation_on_shape(
    img=mem_round + nuc_round,
    param_img_coords=coords_round,
    representation=gfp_representation
)
# Visualize the morphed FP image:
plt.imshow((mem_round + nuc_round)[w // 3], cmap='gray')
plt.imshow(gfp_morphed[w // 3], cmap='gray', alpha=0.25)
plt.axis('off')