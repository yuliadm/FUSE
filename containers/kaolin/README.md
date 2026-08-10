# FUSE Kaolin Container 

Consumes FUSE VGGT Container & FUSE Hunyuan Container outputs:
```
 - data/cleaned_geometry/global/broken_clean.ply
 - data/cleaned_geometry/global/broken_clean_normals.ply
 - data/vggt_outputs/global/raw_vggt_cloud.ply
 - data/scenes/global/reference/
```

Tasks:
Kaolin alignment / differentiable geometry;
provides PyTorch-based 3D representations, differentiable rendering, differentiable cameras, metrics, mesh utilities, and Jupyter visualization support.


# Outputs
```
 - data/kaolin_outputs/global/
```

# Build the container 
(from the project root):
```
cd Documents/FUSE/containers/kaolin

mkdir -p cache ../../notebooks ../../data

docker compose build
docker compose up
```


Open the Jupyter notebook in the browser:
<http://localhost:8891/lab>
and run:
```
notebooks/fuse-hunyuan.ipynb
```

 





