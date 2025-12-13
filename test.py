import numpy as np
from geometry import AirshipGeometry, plot_petal_profile, STANDARD_ENVELOPES, GertlerEnvelope
from plotter import multi_lobe_volume

# parameters = {
#     "ENVELOPE_PARAMS": STANDARD_ENVELOPES["Wang"],
#     "ENVELOPE_LENGTH": 100,
#     "ENVELOPE_RESOLUTION": 100,

#     "LOBE_NUMBER": 3,
#     "LOBE_OFFSET_X": 13.333,
#     "LOBE_OFFSET_Y": 13.333 / 2,
#     "LOBE_OFFSET_Z": 7,

#     "CENTRAL_LOBE_LENGTH": 80,

#     "FIN_AXIAL_OFFSET": 80,
#     "FIN_THICKNESS": 12,
#     "FIN_RC_LENGTH": 8,
#     "FIN_SECTION_RESOLUTION": 40,
#     "FIN_TAPER_RATIO": 0.5,
#     "FIN_HEIGHT": 5,
#     "FIN_NUMBER": 4,
# }

# geometry = AirshipGeometry(parameters, "C:\\SALOME-9.15.0\\run_salome.bat")

# print(geometry.run_salome(open_gui=False, remove_temp_script=True, export_format='BREP'))
# plot_petal_profile(geometry.envelope, 3, 100, "envelope_profile.dat", shape_name="Envelope")

envelope = GertlerEnvelope.from_parameters(STANDARD_ENVELOPES["Wang"], 100, 100)
print(multi_lobe_volume(envelope.diameter, envelope.diameter/4, 0, 3, envelope), 3 * envelope.volume(), envelope.volume_trilobe(envelope.diameter, envelope.diameter/4, 0))