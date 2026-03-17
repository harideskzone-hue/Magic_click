import math
from collections import namedtuple
from pose_scorer.face_group.smile import score_smile  # type: ignore

Landmark = namedtuple('Landmark', ['x', 'y'])
lms = [Landmark(0.0, 0.0)] * 478
valid_lm = Landmark(0.5, 0.5)
for idx in [61, 291, 13, 14, 78, 308]:
    lms[idx] = valid_lm

lms[61] = Landmark(0.4, 0.5)
lms[291] = Landmark(0.6, 0.5)
lms[13] = Landmark(0.5, 0.48)
lms[14] = Landmark(0.5, 0.55)

try:
    res = score_smile(lms, config={}, blendshapes=None)
    print("Success:", res)
except Exception as e:
    import traceback
    traceback.print_exc()

