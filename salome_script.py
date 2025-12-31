# INPUT PARAMETERS START 

ENVELOPE_LENGTH = 100
ENVELOPE_PARAMS = (0.419, 0.337, 0.251, 0.651, 3.266)
ENVELOPE_RESOLUTION = 100
ENVELOPE_TRUNCATION_RATIO = 0

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
INCLUDE_FINS = True  # New parameter to control fin generation

DIRECTORY_PATH = "D:\\Airships\\Salome\\output"
OUTPUT_FILE = "D:\\Airships\\Salome\\output\\test.brep"
OUTPUT_FORMAT = "BREP"
FINAL_OBJECT_NAME = "Airship"

# INPUT PARAMETERS END

# Import all the required modules
import salome
from salome.geom import geomBuilder
import numpy as np
import sys
import importlib

# Salome executes the python script files from its own directory so to import local modules, we have to
# add our own path manually.
sys.path.append(DIRECTORY_PATH)

# This is to reload the local modules once they are changed.
import geometry_handler
importlib.reload(geometry_handler)

# Inititating the Geometry Module of Salome.
salome.salome_init()
geompy = geomBuilder.New()

# ---
# Elementary objects, directions and functions.
# ---

O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)

def translate_object (object, x_offset, y_offset, z_offset):
    return geompy.MakeTranslationTwoPoints(object, O, geompy.MakeVertex(x_offset * LOBE_OFFSET_X, y_offset * LOBE_OFFSET_Y, z_offset * LOBE_OFFSET_Z))

# ---
# Modelling of envelope
# ---

print('[LOG] Generating Hull Profile...')

def create_envelope (params, length):
    print(f'[LOG] Generating envelope having Gertler parameters {params}...')
    gertler = geometry_handler.GertlerEnvelope.from_parameters(params, length, ENVELOPE_RESOLUTION)
    envelope_vertices = [geompy.MakeVertex(x, y, 0) for x, y in gertler.points(ENVELOPE_TRUNCATION_RATIO)]

    envelope_edges = [geompy.MakeInterpol(envelope_vertices, False, False), geompy.MakeLineTwoPnt(geompy.MakeVertex(length * (1 - ENVELOPE_TRUNCATION_RATIO), 0, 0), O)]

    if ENVELOPE_TRUNCATION_RATIO:
        try: envelope_edges.append(geompy.MakeLineTwoPnt(geompy.MakeVertex(length * (1 - ENVELOPE_TRUNCATION_RATIO), geompy.PointCoordinates(envelope_vertices[-1])[1], 0), geompy.MakeVertex(length * (1 - ENVELOPE_TRUNCATION_RATIO), 0, 0)))
        except: pass

    envelope_wire = geompy.MakeWire(envelope_edges, 1e-7)
    envelope_face = geompy.MakeFace(envelope_wire, 1)
    envelope = geompy.MakeRevolution(envelope_face, OX, 2 * np.pi)

    return gertler, envelope

extreme_gertler, extreme_lobe = create_envelope(ENVELOPE_PARAMS, ENVELOPE_LENGTH)

lobes = [extreme_lobe] if LOBE_NUMBER == 1 else [translate_object(extreme_lobe, 0, -1, 0), translate_object(extreme_lobe, 0, 1, 0)]

if LOBE_NUMBER == 3:
    central_lobe = extreme_lobe if not CENTRAL_LOBE_PARAMS else create_envelope(CENTRAL_LOBE_PARAMS, CENTRAL_LOBE_LENGTH)[1]
    lobes.append(translate_object(central_lobe, 1, 0, 1))

# ---
# Modelling of Fins (Conditional)
# ---

fins = []

if INCLUDE_FINS:
    print('[LOG] Generating Fins...')

    RC_RADIAL_OFFSET = extreme_gertler.at(FIN_AXIAL_OFFSET)
    TC_RADIAL_OFFSET = RC_RADIAL_OFFSET + FIN_HEIGHT
    RC_AXIAL_OFFSET = FIN_AXIAL_OFFSET
    TC_AXIAL_OFFSET = RC_AXIAL_OFFSET + FIN_RC_LENGTH/2 * (1 - FIN_TAPER_RATIO) + FIN_HEIGHT * np.tan(np.radians(FIN_SWEEP_ANGLE))

    COS_TIP_ANGLE = np.cos(np.radians(FIN_TIP_ANGLE))
    SIN_TIP_ANGLE = np.sin(np.radians(FIN_TIP_ANGLE))

    rc_vertices = []
    tc_vertices = []

    for x, y in geometry_handler.naca_airfoil_points(FIN_THICKNESS, FIN_SECTION_RESOLUTION, FIN_RC_LENGTH):
        rc_vertices.append(geompy.MakeVertex(RC_AXIAL_OFFSET + x, y, RC_RADIAL_OFFSET))
        tc_vertices.append(geompy.MakeVertex(TC_AXIAL_OFFSET + x * FIN_TAPER_RATIO * COS_TIP_ANGLE, y * FIN_TAPER_RATIO, TC_RADIAL_OFFSET - x * FIN_TAPER_RATIO * SIN_TIP_ANGLE))

    rc_wire = geompy.MakePolyline(rc_vertices, True)
    tc_wire = geompy.MakePolyline(tc_vertices, True)
    rc_face = geompy.MakeFace(rc_wire, True)
    tc_face = geompy.MakeFace(tc_wire, True)

    midchord_direction = [geompy.MakeVertex(RC_AXIAL_OFFSET, 0, 0), geompy.MakeVertex(TC_AXIAL_OFFSET, 0, FIN_HEIGHT)]
    planform_surface = geompy.MakePipeWithDifferentSectionsBySteps([rc_wire, tc_wire], midchord_direction, geompy.MakePolyline(midchord_direction, False))

    TRAIL_X, TRAIL_Z, INTERCEPT_OFFSET = extreme_gertler.get_fin_intercept(RC_AXIAL_OFFSET, FIN_RC_LENGTH)

    fin = geompy.MakeSolid(geompy.MakeShell([planform_surface, rc_face, tc_face]))
    fin = geompy.MakeRotationThreePoints(fin, geompy.MakeVertex(RC_AXIAL_OFFSET, 0, RC_RADIAL_OFFSET), geompy.MakeVertex(RC_AXIAL_OFFSET + FIN_RC_LENGTH, 0, RC_RADIAL_OFFSET), geompy.MakeVertex(TRAIL_X, 0, TRAIL_Z))
    fin = geompy.MakeTranslationVectorDistance(fin, OZ, -INTERCEPT_OFFSET)

    if LOBE_NUMBER == 1:
        if not FIN_THETA_POS:
            FIN_THETA_POS = [i * (360 / FIN_NUMBER) for i in range(0, int(FIN_NUMBER))]
        fins = [geompy.MakeRotation(fin, OX, -np.pi/2 + np.radians(theta)) for theta in FIN_THETA_POS]
    else:
        for theta in FIN_THETA_POS:
            fins.append(translate_object(geompy.MakeRotation(fin, OX, np.radians(theta)), 0, -1, 0))
            fins.append(translate_object(geompy.MakeRotation(fin, OX, np.radians(-theta)), 0, 1, 0))
else:
    print('[LOG] Skipping fin generation as per user choice...')

# ---
# Modelling of Thin Fairings (conditional)
# ---

fairings = []

def create_fairing_quad (p1, p2, p3, p4):
    fairing = geompy.MakeQuad4Vertices(geompy.MakeVertex(*p1), geompy.MakeVertex(*p2), geompy.MakeVertex(*p3), geompy.MakeVertex(*p4))
    normal = geompy.MakeVectorDXDYDZ(*np.cross(np.array(p2) - np.array(p1), np.array(p3) - np.array(p1)))
    fairings.append(geompy.MakePrismVecH2Ways(fairing, normal, 1e-7))

if SHEET_LENGTH_RATIO:
    print("[LOG] Generating fairings...")

    SHEET_LENGTH = ENVELOPE_LENGTH * SHEET_LENGTH_RATIO

    if LOBE_NUMBER == 2:
        create_fairing_quad((ENVELOPE_LENGTH, -LOBE_OFFSET_Y, 0), (ENVELOPE_LENGTH, LOBE_OFFSET_Y, 0), (ENVELOPE_LENGTH - SHEET_LENGTH, LOBE_OFFSET_Y, 0), (ENVELOPE_LENGTH - SHEET_LENGTH, -LOBE_OFFSET_Y, 0))
    elif LOBE_NUMBER == 3:
        create_fairing_quad((ENVELOPE_LENGTH, -LOBE_OFFSET_Y, 0), (CENTRAL_LOBE_LENGTH + LOBE_OFFSET_X, 0, LOBE_OFFSET_Z), (ENVELOPE_LENGTH - SHEET_LENGTH, -LOBE_OFFSET_Y, 0), (CENTRAL_LOBE_LENGTH + LOBE_OFFSET_X - SHEET_LENGTH, 0, LOBE_OFFSET_Z))
        create_fairing_quad((ENVELOPE_LENGTH, LOBE_OFFSET_Y, 0), (CENTRAL_LOBE_LENGTH + LOBE_OFFSET_X, 0, LOBE_OFFSET_Z), (ENVELOPE_LENGTH - SHEET_LENGTH, LOBE_OFFSET_Y, 0), (CENTRAL_LOBE_LENGTH + LOBE_OFFSET_X - SHEET_LENGTH, 0, LOBE_OFFSET_Z))

# Final Fusion Logic
print('[LOG] Generating final model...')

Final_Hull_Solid = geompy.MakeFuseList(lobes + (fins if INCLUDE_FINS else []))
Final_Hull_Solid = geompy.MakeCompound([Final_Hull_Solid] + fairings)
Final_Hull_Solid_ID = geompy.addToStudy(Final_Hull_Solid, FINAL_OBJECT_NAME)

if salome.sg.hasDesktop():
    gg = salome.ImportComponentGUI("GEOM")
    gg.createAndDisplayGO(Final_Hull_Solid_ID)
    gg.setDisplayMode(Final_Hull_Solid_ID, 1)
    salome.sg.updateObjBrowser()

print(f'[LOG] Attempting to export to "{OUTPUT_FILE}"...')

if OUTPUT_FORMAT == 'STL':
    # The deflection used for STL generation is a default one because we can later work with it using MeshLab.
    geompy.ExportSTL(Final_Hull_Solid, OUTPUT_FILE, False)
elif OUTPUT_FORMAT == 'BREP':
    geompy.ExportBREP(Final_Hull_Solid, OUTPUT_FILE)
elif OUTPUT_FORMAT == 'STEP':
    geompy.ExportSTEP(Final_Hull_Solid, OUTPUT_FILE)

print(f'[LOG] Exported {OUTPUT_FORMAT} successfully. Waiting for user to exit (MainLoop)...')
