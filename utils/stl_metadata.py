import numpy as np
import json
import os

# ...existing code...
import numpy as np
import json
import os

def calculate_volume_and_surface_area(vertices, faces):
    """
    Calculate the volume and surface area of a mesh defined by vertices and faces.
    Vectorized, low-overhead numpy implementation (vertices in mm => volume mm3, area mm2).
    """
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    if verts.size == 0 or faces.size == 0:
        return float('nan'), float('nan')

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Surface area: sum of triangle areas
    tri_cross = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(tri_cross, axis=1)
    surface_area = float(np.sum(tri_areas))

    # Signed volume contribution for each triangle (scalar triple product)
    signed_vols = np.einsum('ij,ij->i', v0, np.cross(v1, v2))
    volume = abs(float(np.sum(signed_vols) / 6.0))

    return volume, surface_area

def save_metadata_to_json(metadata, output_file):
    """Save metadata dict to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True)
