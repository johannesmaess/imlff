from typing import NamedTuple

Version = NamedTuple('version', [('major', int), ('minor', int), ('patch', int)])
VERSION = Version(12, 0, 0)