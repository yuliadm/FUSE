# Ideas and To-Do's for FUSE

## Meeting 13.08.2026

### Ideas

1. Include into the FUSE workflow the possibility to generate the reference image by prompting an LLM/VLM.
2. Expand the use cases to include generation of larger missing fragments requiring more imagination/ creativity (like generating the whole head of the dragon giving the body part starting from the neck sown + the broken rim geometry).
3. Add point/mesh segmentation/annotation to explicitly tell the model which parts of the point cloud or a 2D image require attention (missing parts, seam area).
4. Obtaining a more precise point cloud with VGGT alternatives:
* COLMAP
* MAST3R.
5. Obtaining better intact prior
* Hunyuan candidates ensemble instead of a single candidate
* Completion with PCN (Point Cloud Network)
6. Targeting 2D images for both prior and broken object generation instead of meshes/ point clouds (? think about how to make the obtained fragment fit the real-world object).
7. Try replacing Kaolin alignment with RANSAC (since Kaolin does not use it under the hood).

### TO DO

Yulia:
1. Try to obtain a more precise VGGT point cloud:
* filming with turned off auto-focus
* tune params
* run the model with the COLMAP Bundle Adjustment option.
2. Expand the dataset: take a video + pictures of a headless dragon.
3. Add the VLM prompting part to generate intact images (reference).
4. Try using RANSAC instead of Kaolin. 


Guillaumme:
1. Add the SAM2 annotation code.


