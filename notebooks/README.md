# Start Jupyter Notebook inside the container:


First start the container - fuse-vggt, fuse-hunyuan, fuse-kaolin (on the host) from the folder containing the Dockerfile for the desired module:
```
cd /path/to/FUSE/containers/vggt

# assuming the container has been built, otherwise run: docker compose build
docker compose up
```
* resp container folder names are vggt, hunyuan, kaolin, etc. 

Open the URL in the browser (e.g. Firefox, Chrome, ...):
VGGT: <http://127.0.0.1:8890/lab>
Hunyuan: <http://127.0.0.1:8892/lab>

NOTE: they different host ports (to enable simultaneous runs if necessary).

Open the notebooks folder, select the notebook to run (fuse-vggt.ipynb or any other desired .ipynb)
```
notebooks/fuse-vggt.ipynb
```


Connect to the kernel and run the code.
