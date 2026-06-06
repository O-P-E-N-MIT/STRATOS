# INPUT PARAMETERS START

ENVELOPE_LENGTH = 100
ENVELOPE_PARAMS = (0.419, 0.337, 0.251, 0.651, 3.266)
ENVELOPE_RESOLUTION = 100
ENVELOPE_TRUNCATION_RATIO = 0
ENVELOPE_SERIES = "DRAGON_DREAM"

# --- DRAGON DREAM DEFAULTS ---
HULL_WIDTH = 29.5
HULL_HEIGHT = 14.0
BOTTOM_FLATNESS = 0.25

LOBE_NUMBER = 1
LOBE_OFFSET_X = 13.333
LOBE_OFFSET_Y = 13.333/2
LOBE_OFFSET_Z = 7

CENTRAL_LOBE_PARAMS = (0.419, 0.337, 0.251, 0.651, 3.266)
CENTRAL_LOBE_LENGTH = 80

FIN_AXIAL_OFFSET = 80
FIN_THICKNESS = 12
FIN_RC_LENGTH = 8
FIN_SECTION_RESOLUTION = 40
FIN_TAPER_RATIO = 0.5
FIN_HEIGHT = 5
FIN_SWEEP_ANGLE = 0
FIN_TIP_ANGLE = 10
FIN_NUMBER = 4
FIN_THETA_POS = None

SHEET_LENGTH_RATIO = 0.75
INCLUDE_FINS = True

INCLUDE_WINGS = False
WING_SPAN = 20.0
WING_ROOT_CHORD = 5.0
WING_TIP_CHORD = 2.0
WING_SWEEP = 15.0
WING_DIHEDRAL = 5.0
WING_TWIST_ROOT = 2.0
WING_TWIST_TIP = -2.0
WING_THICKNESS = 12.0
WING_AXIAL_OFFSET = 40.0

DIRECTORY_PATH = "C:\\MIT\\untitled"
OUTPUT_DIRECTORY = 'C:\\MIT\\untitled\\Output_1'
OUTPUT_FORMAT = "BREP"
FINAL_OBJECT_NAME = "Airship"

# INPUT PARAMETERS END

import sys
import os
import traceback
import importlib
import math

# Ensure the paths are registered
sys.path.append(DIRECTORY_PATH)

try:
    import salome
    from salome.geom import geomBuilder
    import numpy as np

    import geometry_handler
    importlib.reload(geometry_handler)

    import airfoil
    importlib.reload(airfoil)

    salome.salome_init()
    geompy = geomBuilder.New()

    O = geompy.MakeVertex(0, 0, 0)
    OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
    OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
    OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)

    # --- TOPOLOGY SAFEGUARD ---
    def safe_polyline(pts, is_closed=True):
        # We MUST NOT delete internal points, or tapered lofting will crash due to vertex mismatch.
        # We only delete the very last point if it is an exact duplicate of the first point (Trailing Edge).
        if is_closed and len(pts) > 2:
            c1 = geompy.PointCoordinates(pts[0])
            c2 = geompy.PointCoordinates(pts[-1])
            if math.sqrt(sum((a-b)**2 for a, b in zip(c1, c2))) < 1e-6:
                pts = pts[:-1]

        return geompy.MakePolyline(pts, is_closed)

    def translate_object(object, x_offset, y_offset, z_offset):
        return geompy.MakeTranslationTwoPoints(object, O, geompy.MakeVertex(x_offset * LOBE_OFFSET_X, y_offset * LOBE_OFFSET_Y, z_offset * LOBE_OFFSET_Z))

    # --- Modelling of envelope ---
    print('[LOG] Generating hull Profile...')
    sys.stdout.flush()

    def create_envelope(params, length):
        if ENVELOPE_SERIES == "DRAGON_DREAM":
            envelope_geom = geometry_handler.DragonDreamEnvelope(
                length, HULL_WIDTH, HULL_HEIGHT, BOTTOM_FLATNESS
            )

            # 1. Base unit sphere
            base_sphere = geompy.MakeSphereR(1.0)
            origin = geompy.MakeVertex(0, 0, 0)

            # 2. Scale into Tri-axial Ellipsoid
            hull = geompy.MakeScaleAlongAxes(base_sphere, origin, length/2.0, HULL_WIDTH/2.0, HULL_HEIGHT/2.0)

            # 3. Create Flat Bottom (Super Ellipse approximation via Boolean Cut)
            cut_height = (HULL_HEIGHT / 2.0) * BOTTOM_FLATNESS
            if cut_height > 0:
                cut_box = geompy.MakeBoxDXDYDZ(length * 2, HULL_WIDTH * 2, cut_height)
                # Shift box so its top face sits exactly where we want the cut
                cut_box = geompy.MakeTranslation(cut_box, -length, -HULL_WIDTH, -(HULL_HEIGHT/2.0))
                hull = geompy.MakeCutList(hull, [cut_box], True)

            # 4. Shift hull so the Nose is at X=0, matching Gertler/NACA coordinate systems
            envelope = geompy.MakeTranslation(hull, length/2.0, 0, 0)

            return envelope_geom, envelope

        # --- Axisymmetric Revolved Profiles ---
        elif ENVELOPE_SERIES == "GERTLER":
            envelope_geom = geometry_handler.GertlerEnvelope.from_parameters(params, length, ENVELOPE_RESOLUTION)
        elif ENVELOPE_SERIES == "NACA":
            envelope_geom = geometry_handler.NACAEnvelope.from_parameters((params[4],), length, ENVELOPE_RESOLUTION)

        envelope_vertices = [geompy.MakeVertex(x, y, 0) for x, y in envelope_geom.points(ENVELOPE_TRUNCATION_RATIO)]
        envelope_edges = [geompy.MakeInterpol(envelope_vertices, False, False), geompy.MakeLineTwoPnt(geompy.MakeVertex(length * (1 - ENVELOPE_TRUNCATION_RATIO), 0, 0), O)]

        if ENVELOPE_TRUNCATION_RATIO:
            try: envelope_edges.append(geompy.MakeLineTwoPnt(geompy.MakeVertex(length * (1 - ENVELOPE_TRUNCATION_RATIO), geompy.PointCoordinates(envelope_vertices[-1])[1], 0), geompy.MakeVertex(length * (1 - ENVELOPE_TRUNCATION_RATIO), 0, 0)))
            except: pass

        envelope_wire = geompy.MakeWire(envelope_edges, 1e-7)
        envelope_face = geompy.MakeFace(envelope_wire, 1)
        envelope = geompy.MakeRevolution(envelope_face, OX, 2 * np.pi)

        return envelope_geom, envelope

    extreme_envelope_geom, extreme_lobe = create_envelope(ENVELOPE_PARAMS, ENVELOPE_LENGTH)

    lobes = [extreme_lobe] if LOBE_NUMBER == 1 else [translate_object(extreme_lobe, 0, -1, 0), translate_object(extreme_lobe, 0, 1, 0)]

    if LOBE_NUMBER == 3:
        central_lobe = extreme_lobe if not CENTRAL_LOBE_PARAMS else create_envelope(CENTRAL_LOBE_PARAMS, CENTRAL_LOBE_LENGTH)[1]
        lobes.append(translate_object(central_lobe, 1, 0, 1))

    # --- Modelling of Fins ---
    fins = []
    if INCLUDE_FINS:
        print('[LOG] Generating fins...')
        sys.stdout.flush()

        try:
            RC_RADIAL_OFFSET = extreme_envelope_geom.at(FIN_AXIAL_OFFSET)
            TC_RADIAL_OFFSET = RC_RADIAL_OFFSET + FIN_HEIGHT
            RC_AXIAL_OFFSET = FIN_AXIAL_OFFSET
            TC_AXIAL_OFFSET = RC_AXIAL_OFFSET + FIN_RC_LENGTH/2 * (1 - FIN_TAPER_RATIO) + FIN_HEIGHT * np.tan(np.radians(FIN_SWEEP_ANGLE))

            COS_TIP_ANGLE = np.cos(np.radians(FIN_TIP_ANGLE))
            SIN_TIP_ANGLE = np.sin(np.radians(FIN_TIP_ANGLE))

            rc_vertices = []
            tc_vertices = []

            x_coords, y_coords = airfoil.get_airfoil_points(thickness=FIN_THICKNESS, resolution=FIN_SECTION_RESOLUTION, scale_factor=FIN_RC_LENGTH)
            for x, y in zip(x_coords, y_coords):
                rc_vertices.append(geompy.MakeVertex(RC_AXIAL_OFFSET + x, y, RC_RADIAL_OFFSET))
                tc_vertices.append(geompy.MakeVertex(TC_AXIAL_OFFSET + x * FIN_TAPER_RATIO * COS_TIP_ANGLE, y * FIN_TAPER_RATIO, TC_RADIAL_OFFSET - x * FIN_TAPER_RATIO * SIN_TIP_ANGLE))

            rc_wire = safe_polyline(rc_vertices, True)
            tc_wire = safe_polyline(tc_vertices, True)

            fin = geompy.MakeThruSections([rc_wire, tc_wire], True, 0.0001, True)

            TRAIL_X, TRAIL_Z, INTERCEPT_OFFSET = extreme_envelope_geom.get_chord_intercept(RC_AXIAL_OFFSET, FIN_RC_LENGTH)

            dz = TRAIL_Z - RC_RADIAL_OFFSET
            dx = TRAIL_X - RC_AXIAL_OFFSET
            angle = math.atan2(dz, dx)

            axis_pt = geompy.MakeVertex(RC_AXIAL_OFFSET, 0, RC_RADIAL_OFFSET)
            rot_axis = geompy.MakeLineTwoPnt(axis_pt, geompy.MakeVertex(RC_AXIAL_OFFSET, 1, RC_RADIAL_OFFSET))

            # Negate the angle to pitch DOWN
            fin = geompy.MakeRotation(fin, rot_axis, -angle)

            # Apply penetration margin
            FIN_PENETRATION_MARGIN = (FIN_THICKNESS / 100.0) * FIN_RC_LENGTH * 0.75
            fin = geompy.MakeTranslationVectorDistance(fin, OZ, -INTERCEPT_OFFSET - FIN_PENETRATION_MARGIN)

            if LOBE_NUMBER == 1:
                if not FIN_THETA_POS:
                    FIN_THETA_POS = [i * (360 / FIN_NUMBER) for i in range(0, int(FIN_NUMBER))]
                fins = [geompy.MakeRotation(fin, OX, -np.pi/2 + np.radians(theta)) for theta in FIN_THETA_POS]
            else:
                for theta in FIN_THETA_POS:
                    fins.append(translate_object(geompy.MakeRotation(fin, OX, np.radians(theta)), 0, -1, 0))
                    fins.append(translate_object(geompy.MakeRotation(fin, OX, np.radians(-theta)), 0, 1, 0))
        except Exception as e:
            print(f"[WARNING] Fin generation failed and was skipped: {e}")
            sys.stdout.flush()
    else:
        print('[LOG] Skipping fin generation...')
        sys.stdout.flush()

    # --- Modelling of Wings ---
    wings = []
    if INCLUDE_WINGS and 'WING_STATIONS' in globals() and len(WING_STATIONS) > 1:
        print('[LOG] Starting Arc-Length Standardized Wing Generation...')
        sys.stdout.flush()

        try:
            # 1. Calculate Hull Penetration
            WING_ROOT_RADIUS = extreme_envelope_geom.at(WING_AXIAL_OFFSET)
            root_chord = WING_STATIONS[0]['chord']

            try:
                _, _, WING_INTERCEPT = extreme_envelope_geom.get_chord_intercept(WING_AXIAL_OFFSET, root_chord)
            except:
                WING_INTERCEPT = 0.0

            # Push the root into the hull slightly to ensure Boolean Fusion works later
            PENETRATION_MARGIN = (root_chord * 0.25)
            WING_START_Y = max(0, WING_ROOT_RADIUS - WING_INTERCEPT - PENETRATION_MARGIN)

            wires_right = []
            wires_left = []

            for i, station in enumerate(WING_STATIONS):
                y_val = station['y'] + WING_START_Y
                chord = station['chord']
                twist = np.radians(station['twist'])
                x_le = station['x_off']
                z_shift = station['z_off']

                x_norm = np.array(WING_AIRFOILS_X[i])
                z_norm = np.array(WING_AIRFOILS_Y[i])

                # --- THE ARC-LENGTH NORMALIZER ---
                # 1. Remove duplicate TE points explicitly
                if math.hypot(x_norm[0] - x_norm[-1], z_norm[0] - z_norm[-1]) < 1e-6:
                    x_norm = x_norm[:-1]
                    z_norm = z_norm[:-1]

                # 2. Force the Trailing Edge (Max X) to be the absolute starting point
                te_idx = np.argmax(x_norm)
                x_norm = np.roll(x_norm, -te_idx)
                z_norm = np.roll(z_norm, -te_idx)

                # 3. Standardize direction (Counter-Clockwise over the top)
                area = np.sum((np.roll(x_norm, -1) - x_norm) * (np.roll(z_norm, -1) + z_norm))
                if area < 0:
                    x_norm = np.concatenate(([x_norm[0]], x_norm[1:][::-1]))
                    z_norm = np.concatenate(([z_norm[0]], z_norm[1:][::-1]))

                # 4. Reparameterize by arc-length to guarantee 1-to-1 vertex mapping
                x_closed = np.append(x_norm, x_norm[0])
                z_closed = np.append(z_norm, z_norm[0])

                dx = np.diff(x_closed)
                dz = np.diff(z_closed)
                dist = np.sqrt(dx**2 + dz**2)
                s = np.zeros(len(x_closed))
                s[1:] = np.cumsum(dist)

                if s[-1] > 0:
                    s /= s[-1]

                # Resample to exactly 120 points evenly spaced around the perimeter
                s_uniform = np.linspace(0, 1, 120)
                x_norm = np.interp(s_uniform, s, x_closed)
                z_norm = np.interp(s_uniform, s, z_closed)

                # Remove the duplicate TE so MakePolyline can close it cleanly
                x_norm = x_norm[:-1]
                z_norm = z_norm[:-1]
                # ---------------------------------

                pts_r, pts_l = [], []
                for j in range(len(x_norm)):
                    x_af = x_norm[j] * chord
                    z_af = z_norm[j] * chord

                    # Apply twist around aerodynamic center (0.25c)
                    x_qc = 0.25 * chord
                    x_shift_val = x_af - x_qc
                    x_rot = x_shift_val * np.cos(twist) + z_af * np.sin(twist)
                    z_rot = -x_shift_val * np.sin(twist) + z_af * np.cos(twist)
                    x_rot += x_qc

                    X_f, Z_f = WING_AXIAL_OFFSET + x_rot + x_le, z_rot + z_shift
                    pts_r.append(geompy.MakeVertex(X_f, y_val, Z_f))
                    pts_l.append(geompy.MakeVertex(X_f, -y_val, Z_f))

                wires_right.append(safe_polyline(pts_r, True))
                wires_left.append(safe_polyline(pts_l, True))

            # --- RULED LOFTING ---
            # Using 'False' ensures straight structural lofts between identical topological wire grids
            wing_right = geompy.MakeThruSections(wires_right, True, 0.0001, False)
            wing_left = geompy.MakeThruSections(wires_left, True, 0.0001, False)

            wings = [wing_right, wing_left]
            print(f"[SUCCESS] Wings lofted successfully.")

        except Exception as e:
            print(f"[CRITICAL ERROR] Wing generation failed: {traceback.format_exc()}")
            sys.stdout.flush()
    else:
        print('[LOG] Skipping wing generation...')
        sys.stdout.flush()

    # --- Modelling of Thin Fairings ---
    fairings = []
    def create_fairing_quad(p1, p2, p3, p4):
        fairing = geompy.MakeQuad4Vertices(geompy.MakeVertex(*p1), geompy.MakeVertex(*p2), geompy.MakeVertex(*p3), geompy.MakeVertex(*p4))
        normal = geompy.MakeVectorDXDYDZ(*np.cross(np.array(p2) - np.array(p1), np.array(p3) - np.array(p1)))
        fairings.append(geompy.MakePrismVecH2Ways(fairing, normal, 1e-7))

    if SHEET_LENGTH_RATIO:
        print("[LOG] Generating fairings...")
        sys.stdout.flush()
        SHEET_LENGTH = ENVELOPE_LENGTH * SHEET_LENGTH_RATIO
        if LOBE_NUMBER == 2:
            create_fairing_quad((ENVELOPE_LENGTH, -LOBE_OFFSET_Y, 0), (ENVELOPE_LENGTH, LOBE_OFFSET_Y, 0), (ENVELOPE_LENGTH - SHEET_LENGTH, LOBE_OFFSET_Y, 0), (ENVELOPE_LENGTH - SHEET_LENGTH, -LOBE_OFFSET_Y, 0))
        elif LOBE_NUMBER == 3:
            create_fairing_quad((ENVELOPE_LENGTH, -LOBE_OFFSET_Y, 0), (CENTRAL_LOBE_LENGTH + LOBE_OFFSET_X, 0, LOBE_OFFSET_Z), (ENVELOPE_LENGTH - SHEET_LENGTH, -LOBE_OFFSET_Y, 0), (CENTRAL_LOBE_LENGTH + LOBE_OFFSET_X - SHEET_LENGTH, 0, LOBE_OFFSET_Z))
            create_fairing_quad((ENVELOPE_LENGTH, LOBE_OFFSET_Y, 0), (CENTRAL_LOBE_LENGTH + LOBE_OFFSET_X, 0, LOBE_OFFSET_Z), (ENVELOPE_LENGTH - SHEET_LENGTH, LOBE_OFFSET_Y, 0), (CENTRAL_LOBE_LENGTH + LOBE_OFFSET_X - SHEET_LENGTH, 0, LOBE_OFFSET_Z))

    # --- Robust Final Fusion Logic ---
    print('[LOG] Fusing geometry model incrementally...')
    sys.stdout.flush()

    Final_Lobes_Solid = lobes[0]
    if len(lobes) > 1:
        try:
            Final_Lobes_Solid = geompy.MakeFuseList(lobes)
        except Exception as e:
            print(f"[WARNING] Lobe Fusion failed: {e}")

    Final_Airship_Solid = Final_Lobes_Solid

    appendages = (fins if INCLUDE_FINS else []) + wings
    for i, appendage in enumerate(appendages):
        try:
            Final_Airship_Solid = geompy.MakeFuse(Final_Airship_Solid, appendage)
        except Exception as e:
            print(f"[WARNING] Failed to fuse appendage {i}. Bypassing geometry failure: {e}")
            sys.stdout.flush()

    if len(fairings) > 0:
        try:
            Final_Airship_Solid = geompy.MakeCompound([Final_Airship_Solid] + fairings)
        except Exception as e:
            print(f"[WARNING] Failed to attach fairings: {e}")

    Final_Airship_Solid_ID = geompy.addToStudy(Final_Airship_Solid, FINAL_OBJECT_NAME)

    if salome.sg.hasDesktop():
        gg = salome.ImportComponentGUI("GEOM")
        gg.createAndDisplayGO(Final_Airship_Solid_ID)
        gg.setDisplayMode(Final_Airship_Solid_ID, 1)
        salome.sg.updateObjBrowser()

    OUTPUT_FILE_COMPLETE = os.path.join(OUTPUT_DIRECTORY, f"{FINAL_OBJECT_NAME}.{OUTPUT_FORMAT.lower()}")
    OUTPUT_FILE_LOBES = os.path.join(OUTPUT_DIRECTORY, f"{FINAL_OBJECT_NAME}_lobes.{OUTPUT_FORMAT.lower()}")

    # ALWAYS export an STL for the PyVista UI preview to read
    OUTPUT_FILE_PREVIEW_STL = os.path.join(OUTPUT_DIRECTORY, f"{FINAL_OBJECT_NAME}.stl")
    geompy.ExportSTL(Final_Airship_Solid, OUTPUT_FILE_PREVIEW_STL, False)

    print(f'[LOG] Attempting to export to {OUTPUT_FILE_COMPLETE}...')
    sys.stdout.flush()

    if OUTPUT_FORMAT == 'STL':
        geompy.ExportSTL(Final_Lobes_Solid, OUTPUT_FILE_LOBES, False)
    elif OUTPUT_FORMAT == 'BREP':
        geompy.ExportBREP(Final_Airship_Solid, OUTPUT_FILE_COMPLETE)
        geompy.ExportBREP(Final_Lobes_Solid, OUTPUT_FILE_LOBES)
    elif OUTPUT_FORMAT == 'STEP':
        geompy.ExportSTEP(Final_Airship_Solid, OUTPUT_FILE_COMPLETE)
        geompy.ExportSTEP(Final_Lobes_Solid, OUTPUT_FILE_LOBES)

    print(f'[LOG] Exported {OUTPUT_FORMAT} successfully.')
    sys.stdout.flush()

except Exception as e:
    error_log_path = os.path.join(OUTPUT_DIRECTORY, "salome_crash_log.txt")
    with open(error_log_path, "w") as f:
        f.write("--- SALOME FATAL ERROR ---\n")
        f.write(traceback.format_exc())
    print(f"[FATAL ERROR] Salome crashed critically. Check {error_log_path} for exact trace.")
    sys.stdout.flush()
