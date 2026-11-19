"""Light expression parser: which object classes a query can be about, plus kinematic hints.

The parser is NOT the referring model. It only does two cheap things the VLM is bad at or that
save it work:

* a class family filter (a query about cars never needs to look at pedestrians), and
* kinematic predicates (moving / stationary / turning / direction relative to the drone) that
  are measured on ego-motion-compensated tracks and fed to the provisional prior.

Everything else (colour, attributes, spatial relations, context) is left to the VLM.
Vocabulary was taken from the AerialMind and Refer-KITTI-V2 seqmaps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# VisDrone classes (AERO-MOT) -> family
FAMILY_OF = {"pedestrian": "person", "people": "person", "bicycle": "two_wheeler", "motor": "two_wheeler",
             "tricycle": "tricycle", "awning-tricycle": "tricycle", "car": "car", "van": "car",
             "truck": "truck", "bus": "bus", "person": "person", "cyclist": "two_wheeler",
             "motorcycle": "two_wheeler"}
VEHICLES = {"car", "truck", "bus", "tricycle", "two_wheeler"}

_CLASS_WORDS = [
    (r"\b(pedestrians?|persons?|people|humans?|individuals?|man|men|women|woman|walkers?|"
     r"children|child|kids?|folks?|crowd)\b", {"person"}),
    (r"\b(cyclists?|bicycles?|bikes?|motorcycles?|motorbikes?|motors?|scooters?|mopeds?|riders?|"
     r"two-wheelers?|electric bikes?|e-bikes?)\b", {"two_wheeler"}),
    (r"\b(tricycles?|three-wheelers?|rickshaws?|tuk-tuks?)\b", {"tricycle"}),
    (r"\b(trucks?|lorr(y|ies)|pickups?|pick-ups?)\b", {"truck"}),
    (r"\b(bus|buses|coach(es)?|minibus(es)?)\b", {"bus"}),
    (r"\b(vans?|minivans?)\b", {"car"}),
    (r"\b(cars?|automobiles?|autos?|sedans?|suvs?|taxis?|cabs?|jeeps?)\b", {"car"}),
    (r"\b(vehicles?|traffic|transport(ation)?)\b", set(VEHICLES)),
]
# "motor vehicles" is generic, "motor" alone is VisDrone's motorbike class
_GENERIC_MOTOR = re.compile(r"\bmotor\s+vehicles?\b")


@dataclass
class Query:
    text: str
    families: set = field(default_factory=set)      # empty = any class
    moving: bool | None = None                       # True moving, False stationary, None unknown
    turning: str | None = None                       # "left" | "right" | "any" | None
    rel_ego: str | None = None                       # "same" | "opposite" | None (w.r.t. the drone/us)
    toward: str | None = None                        # "toward" | "away" | None (w.r.t. the camera)
    lateral: str | None = None                       # "left_to_right" | "right_to_left" | None
    negated: bool = False                            # "not moving", "without lights", ...
    residual: list = field(default_factory=list)     # content words the parser cannot measure

    @property
    def measurable(self) -> bool:
        """True when class + kinematics cover the whole query (no colour, attribute, relation...)."""
        return not self.residual and not self.negated

    def class_ok(self, cls_name: str) -> bool:
        if not self.families:
            return True
        return FAMILY_OF.get(cls_name, cls_name) in self.families

    def has_kinematics(self) -> bool:
        return any(v is not None for v in (self.moving, self.turning, self.rel_ego, self.toward, self.lateral))


def parse(text: str) -> Query:
    t = text.lower().replace("_", " ").replace("-", "-")
    t_cls = _GENERIC_MOTOR.sub(" vehicles ", t)
    fam: set = set()
    for pat, f in _CLASS_WORDS:
        if re.search(pat, t_cls):
            fam |= f
    q = Query(text=text, families=fam)
    q.negated = bool(re.search(r"\b(not|without|non|no)\b", t))
    if re.search(r"\b(parked|stationary|stopped|still|static|immobile|non-moving|motionless|at rest|"
                 r"waiting|halted|standing|idle|stationed|queue|queuing)\b", t):
        q.moving = False
    elif re.search(r"\b(moving|driving|traveling|travelling|walking|running|proceeding|progressing|"
                   r"advancing|heading|crossing|journeying|riding|turning|turns|overtaking|"
                   r"approaching|leaving|going|in motion|movement)\b", t):
        q.moving = True
    if re.search(r"\bnot\s+(moving|in motion)\b|\bnon-moving\b", t):
        q.moving = False
    m = re.search(r"\b(left|right)(-hand)?\s+turns?\b|\bturn(s|ing)?\s+(to\s+the\s+)?(left|right)\b", t)
    if m:
        q.turning = m.group(1) or m.group(5)
    elif re.search(r"\b(turning|turns|making a turn|u-turn)\b", t):
        q.turning = "any"
    if re.search(r"\b(same|identical|matching|parallel)\s+(direction|way|path|course|heading)\b.*\b(us|our|ours|we|observer)\b"
                 r"|\b(us|our|ours|we)\b.*\b(same|identical)\s+(direction|way)\b", t):
        q.rel_ego = "same"
    elif re.search(r"\b(opposite|contrary|reverse|opposing|counter)\b.*\b(us|our|ours|we|observer)\b"
                   r"|\b(us|our|ours|we)\b.*\b(opposite|contrary|reverse)\b|\boncoming\b", t):
        q.rel_ego = "opposite"
    if re.search(r"\b(toward|towards|approaching|coming to)\s+(us|the camera|the observer|me)\b", t):
        q.toward = "toward"
    elif re.search(r"\baway from\s+(us|the camera|the observer|me)\b", t):
        q.toward = "away"
    if re.search(r"\bfrom (the )?left (side )?to (the )?right\b|\bleft-to-right\b", t):
        q.lateral = "left_to_right"
    elif re.search(r"\bfrom (the )?right (side )?to (the )?left\b|\bright-to-left\b", t):
        q.lateral = "right_to_left"
    q.residual = _residual(t_cls)
    return q


_STOP = set("""a an the of in on at to from by with for and or is are be being as that this these those
which who whose its their it they them there here than into onto over under along within while
image frame video scene view us our ours we me observer camera drone currently presently also
one ones some all any each both other others""".split())
_KIN_WORDS = re.compile(
    r"\b(parked|stationary|stopped|still|static|immobile|non-moving|motionless|rest|waiting|halted|"
    r"standing|idle|stationed|moving|driving|traveling|travelling|walking|running|proceeding|progressing|"
    r"advancing|heading|crossing|journeying|riding|turning|turns?|making|overtaking|approaching|leaving|"
    r"going|motion|movement|move|moves|same|identical|matching|parallel|opposite|contrary|reverse|"
    r"opposing|counter|oncoming|direction|directional|way|path|course|trajectory|route|toward|towards|"
    r"away|left|right|left-hand|right-hand|u-turn|forward|ahead)\b")


def _residual(t: str) -> list:
    for pat, _ in _CLASS_WORDS:
        t = re.sub(pat, " ", t)
    t = _KIN_WORDS.sub(" ", t)
    words = re.findall(r"[a-z][a-z'-]*", t)
    return [w for w in words if w not in _STOP and len(w) > 2]
