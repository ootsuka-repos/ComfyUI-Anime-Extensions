"""Media node registration; implementations live in feature modules."""

from .avatar import VRMDance, VRMStarter
from .comic import ComicPage
from .sol_h3 import SolH3
from .text import TextCompletion, TextModelRelease, TextRequest

nodes = [VRMStarter, VRMDance, ComicPage, TextRequest, TextCompletion, TextModelRelease, SolH3]
