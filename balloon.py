import numpy as np
from meshlab_handler import save_mesh

# Shapes of balloon: SPHERE, BULGED_SPHERE, PROLATE, OBLATE_LOBED
# Gore models: NONE, SMOOTH_BUMPY, FLAT_FACET, PUMPKIN, OBLATE_LOBED

# Parameters valid for the balloon shape given:
# - SPHERE:
# - BULGED_SPHERE: BULGE_AMPLITUDE, BULGE_POWER,
# - PROLATE: ASPECT_RATIO
# - OBLATE_LOBED: ASPECT_RATIO

# Parameters valid for the gore model given:
# - NONE:
# - SMOOTH_BUMPY: GORE_AMPLITUDE
# - PUMPKIN: GORE_AMPLITUDE, GORE_FADE
# - OBLATE_LOBED: GORE_AMPLITUDE, GORE_POWER
# - FLAT_FACET: 

# A function to create geometry of superpressured balloons.
def create_balloon_geometry (
        gore_model,                 # Gore model of the balloon
        target_volume,              # Target volume of the balloon
        gores,                      # Number of gores
        params,                     # Parameters for the balloon
        theta_resolution=400,       # Mesh resolution for theta
        phi_resolution=600,         # Mesh resolution for phi
        do_volume_correction=True,  # A flag for volume correction 
        output_file="output.stl",   # Output stl file
        single_gore=False           # To export only a single gore
):
    # Formation of the base shape of the balloon.
    theta = np.linspace(0, np.pi, theta_resolution)

    # If aspect ratio is not provided, it is assumed to be 1 (default case for SPHERE)
    aspect_ratio = params.get("ASPECT_RATIO", 1)

    b = (3 * target_volume / (4 * np.pi * aspect_ratio)) ** (1 / 3)
    a = aspect_ratio * b

    r_mer = b * np.sin(theta)
    z_mer = a * np.cos(theta)

    # If a bulging amplitude is given in the parameters,
    if 'BULGE_AMPLITUDE' in params:
        r_mer *= (1 + params['BULGE_AMPLITUDE'] * np.sin(theta) ** params.get('BULGE_POWER', 1))

    # The balloon formed won't have the same volume so an additional volume correction can be applied if the user wants it.
    if do_volume_correction:
        dzdth = np.abs(np.gradient(z_mer, theta))
        V_raw = np.trapezoid(np.pi * r_mer ** 2 * dzdth, theta)
        scale = (target_volume / V_raw) ** (1 / 3)
        r_mer *= scale
        z_mer *= scale

    # Formation of the gores.
    phi = np.linspace(-(np.pi)/gores, (np.pi)/gores, phi_resolution) if single_gore else np.linspace(0, 2*np.pi/gores, phi_resolution)
    TH, PH = np.meshgrid(theta, phi)

    r0 = np.interp(TH, theta, r_mer)
    z = np.interp(TH, theta, z_mer)

    r = r0

    # In case of flat facet gore model, we make the circle into a polygon,
    if gore_model == 'FLAT_FACET':
        phi_mod = np.mod(PH, 2*np.pi / gores)
        r *= np.cos(np.pi / gores) / np.cos(phi_mod - np.pi / gores)

    # In case of other gore models,
    elif gore_model != 'NONE':
        exponent = (
            params.get("GORE_POWER")    # Oblate lobed case
            or params.get("GORE_FADE")  # Pumpkin case
            or 4                        # Smooth bumpy case
        )

        r *= 1 + params["GORE_AMPLITUDE"] * np.cos(gores * PH) * np.sin(TH) ** exponent

    x = r * np.cos(PH)
    y = r * np.sin(PH)
        
    vertices = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    faces = np.zeros(((phi_resolution - 1) * (theta_resolution - 1) * 2, 3))

    for i in range(phi_resolution - 1):
        for j in range(theta_resolution - 1):
            a = i * theta_resolution + j
            b = a + 1
            c = a + theta_resolution
            d = c + 1

            k = 2 * (a - i)

            # Triangular faces: [a, b, d], [a, d, c]
            faces[k][0] = faces[k+1][0] = a
            faces[k][1] = b
            faces[k+1][2] = c
            faces[k][2] = faces[k+1][1] = d

    save_mesh(output_file, vertices, np.array(faces))