# INPUT PARAMETERS START

ENVELOPE_LENGTH = 100
ENVELOPE_PARAMS = (0.419, 0.337, 0.251, 0.651, 3.266)
ENVELOPE_RESOLUTION = 100
ENVELOPE_TRUNCATION_RATIO = 0
ENVELOPE_SERIES = "GERTLER"

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
        clean_pts = []
        for p in pts:
            if not clean_pts:
                clean_pts.append(p)
            else:
                c1 = geompy.PointCoordinates(p)
                c2 = geompy.PointCoordinates(clean_pts[-1])
                if math.sqrt(sum((a-b)**2 for a, b in zip(c1, c2))) > 1e-6:
                    clean_pts.append(p)

        if is_closed and len(clean_pts) > 2:
            c1 = geompy.PointCoordinates(clean_pts[0])
            c2 = geompy.PointCoordinates(clean_pts[-1])
            if math.sqrt(sum((a-b)**2 for a, b in zip(c1, c2))) < 1e-6:
                clean_pts = clean_pts[:-1]

        return geompy.MakePolyline(clean_pts, is_closed)

    def translate_object(object, x_offset, y_offset, z_offset):
        return geompy.MakeTranslationTwoPoints(object, O, geompy.MakeVertex(x_offset * LOBE_OFFSET_X, y_offset * LOBE_OFFSET_Y, z_offset * LOBE_OFFSET_Z))

    # --- Modelling of envelope ---
    print('[LOG] Generating hull Profile...')
    sys.stdout.flush()

    def create_envelope(params, length):
        if ENVELOPE_SERIES == "GERTLER":
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

            # FIX 1: Negate the angle! Salome right-hand rule around +Y makes a negative angle pitch UP.
            # By negating it, we force the fin to pitch DOWN to follow the hull taper.
            fin = geompy.MakeRotation(fin, rot_axis, -angle)

            # FIX 2: Apply a penetration margin. The curved hull will fall away from the flat edges of the thick fin.
            # Sinking it guarantees a clean boolean union without floating gaps.
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
    if INCLUDE_WINGS and WING_SPAN > 0:
        print('[LOG] Generating wings...')
        sys.stdout.flush()

        try:
            WING_ROOT_RADIUS = extreme_envelope_geom.at(WING_AXIAL_OFFSET)
            try:
                _, _, WING_INTERCEPT = extreme_envelope_geom.get_chord_intercept(WING_AXIAL_OFFSET, WING_ROOT_CHORD)
            except Exception:
                WING_INTERCEPT = 0.0

            PENETRATION_MARGIN = (WING_ROOT_CHORD * 0.15) + ((WING_THICKNESS / 100.0) * WING_ROOT_CHORD)
            WING_START_Y = max(0, WING_ROOT_RADIUS - WING_INTERCEPT - PENETRATION_MARGIN)

            n_span = 10
            y_stations = np.linspace(0, WING_SPAN/2, n_span)

            taper = WING_TIP_CHORD / WING_ROOT_CHORD if WING_ROOT_CHORD > 0 else 0
            c_dist = WING_ROOT_CHORD * (1 - (1 - taper) * (2 * y_stations / WING_SPAN))
            x_le = y_stations * np.tan(np.radians(WING_SWEEP))
            z_shift = y_stations * np.tan(np.radians(WING_DIHEDRAL))

            wires_right = []
            wires_left = []

            use_custom_airfoil = 'AIRFOIL_X' in globals() and 'AIRFOIL_Y' in globals() and len(AIRFOIL_X) > 5

            for i in range(len(y_stations)):
                chord = c_dist[i]
                span_half = WING_SPAN/2 if WING_SPAN > 0 else 1
                theta_val = WING_TWIST_ROOT + (WING_TWIST_TIP - WING_TWIST_ROOT) * (y_stations[i] / span_half)
                twist = np.radians(theta_val)
                y_val = y_stations[i] + WING_START_Y

                pts_right = []
                pts_left = []

                if use_custom_airfoil:
                    scaled_x = np.array(AIRFOIL_X) * chord
                    scaled_z = np.array(AIRFOIL_Y) * chord
                    af_pts = zip(scaled_x, scaled_z)
                else:
                    # OLD:
                    # af_pts = geometry_handler.naca_airfoil_points(WING_THICKNESS, FIN_SECTION_RESOLUTION, chord)

                    # NEW:
                    x_c, z_c = airfoil.get_airfoil_points(thickness=WING_THICKNESS, resolution=FIN_SECTION_RESOLUTION, scale_factor=chord)
                    af_pts = zip(x_c, z_c)

                for x_af, z_af in af_pts:
                    x_qc = 0.25 * chord
                    x_shift_val = x_af - x_qc

                    x_rot = x_shift_val * np.cos(twist) + z_af * np.sin(twist)
                    z_rot = -x_shift_val * np.sin(twist) + z_af * np.cos(twist)
                    x_rot += x_qc

                    X_final = WING_AXIAL_OFFSET + x_rot + x_le[i]
                    Z_final = z_rot + z_shift[i]

                    pts_right.append(geompy.MakeVertex(X_final, y_val, Z_final))
                    pts_left.append(geompy.MakeVertex(X_final, -y_val, Z_final))

                wires_right.append(safe_polyline(pts_right, True))
                wires_left.append(safe_polyline(pts_left, True))

            wing_right = geompy.MakeThruSections(wires_right, True, 0.0001, True)
            wing_left = geompy.MakeThruSections(wires_left, True, 0.0001, True)
            wings = [wing_right, wing_left]

        except Exception as e:
            print(f"[WARNING] Wing generation failed and was skipped: {e}")
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

    print(f'[LOG] Attempting to export to {OUTPUT_FILE_COMPLETE}...')
    sys.stdout.flush()

    if OUTPUT_FORMAT == 'STL':
        geompy.ExportSTL(Final_Airship_Solid, OUTPUT_FILE_COMPLETE, False)
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
