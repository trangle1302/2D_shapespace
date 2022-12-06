
from descartes import PolygonPatch
import utils.annotationUtils as annotationUtils
import matplotlib.pyplot as plt


def plot_complete_mask(json_path):
    mask = read_from_json(json_path)
    img_size = (
        mask["bbox"][2] - mask["bbox"][0] + 1,
        mask["bbox"][3] - mask["bbox"][1] + 1,
    )
    img = np.zeros(img_size)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(img)
    for feature in mask["features"]:
        label = int(feature["properties"]["cell_idx"]) + 1
        coords = feature["geometry"]
        ax.add_patch(PolygonPatch(coords))
    ax.axis("off")
    plt.tight_layout()
    #fig.savefig("C:/Users/trang.le/Desktop/tmp.png", bbox_inches="tight")

    # img = imageio.imread('C:/Users/trang.le/Desktop/tmp.png')
    # plt.imshow(img)


def geojson_to_masks(
    file_proc,
    mask_types=["filled", "edge", "labels"],
    img_size=None,
):

    # annot_types = list(masks_to_create.keys())

    annotationsImporter = annotationUtils.GeojsonImporter()

    # Instance to save masks
    masks = annotationUtils.MaskGenerator()

    weightedEdgeMasks = annotationUtils.WeightedEdgeMaskGenerator(sigma=8, w0=10)
    distMapMasks = annotationUtils.DistanceMapGenerator(truncate_distance=None)

    # Decompose file name
    drive, path_and_file = os.path.splitdrive(file_proc)
    path, file = os.path.split(path_and_file)
    # file_base, ext = os.path.splitext(file)

    # Read annotation:  Correct class has been selected based on annot_type
    annot_dict_all, roi_size_all, image_size = annotationsImporter.load(file_proc)
    if img_size is not None:
        image_size = img_size

    annot_types = set(
        annot_dict_all[k]["properties"]["label"] for k in annot_dict_all.keys()
    )
    masks = {}
    for annot_type in annot_types:
        # print("annot_type: ", annot_type)
        # Filter the annotations by label
        annot_dict = {
            k: annot_dict_all[k]
            for k in annot_dict_all.keys()
            if annot_dict_all[k]["properties"]["label"] == annot_type
        }
        # Create masks
        # Binary - is always necessary to creat other masks
        binaryMasks = annotationUtils.BinaryMaskGenerator(
            image_size=image_size, erose_size=5, obj_size_rem=500, save_indiv=True
        )
        mask_dict = binaryMasks.generate(annot_dict)

        # Distance map
        if "distance" in mask_types:
            mask_dict = distMapMasks.generate(annot_dict, mask_dict)

        # Weighted edge mask
        if "weigthed" in mask_types:
            mask_dict = weightedEdgeMasks.generate(annot_dict, mask_dict)

        # border_mask
        if "border_mask" in mask_types:
            border_detection_threshold = max(
                round(1.33 * image_size[0] / 512 + 0.66), 1
            )
            borderMasks = annotationUtils.BorderMaskGenerator(
                border_detection_threshold=border_detection_threshold
            )
            mask_dict = borderMasks.generate(annot_dict, mask_dict)

    return mask_dict
