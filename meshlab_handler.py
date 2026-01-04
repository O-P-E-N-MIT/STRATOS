import pymeshlab

# Apply the required filters for the given MeshSet.
#
# For mesh resolution, these filters can provided as kwargs,
# - targetfacenum: Target number of faces.
# - targetperc: Target size of faces as percentage of length.
def apply_filters_to_meshset (ms, **kwargs):
    # Basic mesh refinement.
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()

    # If required number of faces or percentage is not given, the default setting is taken.
    if ("targetfacenum" not in kwargs) and ("targetperc" not in kwargs):
        # kwargs["targetfacenum"] = 5000
        kwargs["targetperc"] = 0.3

    # Quadratic Edge Collapse Decimation method
    ms.meshing_decimation_quadric_edge_collapse(
        # In one of the configurations, the quality threshold was 1.
        qualitythr=0.6, 
        preserveboundary=True,
        boundaryweight=1,
        preservenormal=True,
        preservetopology=False,
        optimalplacement=True,
        planarquadric=True,
        qualityweight=False,
        autoclean=True,
        selected=False
    )

# Applies filters to an already made STL file by loading it to meshlab.
def apply_filters (filename, **kwargs):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)

    # Apply the required filters
    apply_filters_to_meshset(ms, **kwargs)

    return ms

# Get the meshdata from the given meshset.
# Returns Nx3 arrays of (vertices, faces, normals)
def get_meshdata_from_meshset (ms):
    mesh = ms.current_mesh()
    return mesh.vertex_matrix(), mesh.face_matrix(), mesh.face_normal_matrix()

# Returns the meshdata from the filename.
def get_meshdata (filename, **kwargs):
    mesh = apply_filters(filename, **kwargs)
    return get_meshdata_from_meshset(mesh)