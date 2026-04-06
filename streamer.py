import mmap
import struct
import numpy as np
import time

regular = True

if regular:
    data = np.load("regular_gaussians.npz")
else:
    data = np.load("downsampled_gaussians.npz")

positions = data["position"]   # (N, G, 3)
rotations = data["rotation"]   # (N, G, 4)
scales    = data["scale"] 

if regular:
    N_GAUSSIANS = 204061
else:
    N_GAUSSIANS = 203444

BYTES_FLOAT = 4

SIZE_XYZ = 3
SIZE_SCALE = 3
SIZE_ROT = 4

DATA_SIZE = (SIZE_XYZ + SIZE_SCALE + SIZE_ROT) * BYTES_FLOAT * N_GAUSSIANS

HEADER_SIZE = 4 

shm = mmap.mmap(-1, HEADER_SIZE + DATA_SIZE, tagname="SharedGaussians")

print("openning shared memory","SharedGaussians")

max_fps = 24
target_fps = 24
target_frame_duration = 1.0 / target_fps

skip_amount = max_fps // target_fps

while True:
    for idx in range(len(positions)):
        if idx % skip_amount != 0:
            continue
        start_time = time.perf_counter()
    
        position = positions[idx]
        rotation = rotations[idx]
        scale = scales[idx]
    
        shm.seek(0)
        shm.write(struct.pack("i", 1))
        shm.write(position.tobytes())
        shm.write(rotation.tobytes())
        shm.write(scale.tobytes())
        elapsed = time.perf_counter() - start_time
        sleep_time = target_frame_duration - elapsed
    
        if sleep_time > 0:
            time.sleep(sleep_time)
    
