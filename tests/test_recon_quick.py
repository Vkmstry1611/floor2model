import numpy as np
import sys, os
sys.path.insert(0, os.getcwd())

from src.reconstruction.extruder import Extruder, Mesh3D

def test_basic():
    extruder = Extruder(wall_height=2.8, pixels_per_metre=100.0)
    pts = [(0,0),(100,0),(100,100),(0,100)]
    mesh = extruder.extrude_polygon(pts, label="OuterWall")
    assert mesh.vertex_count > 0
    print("PASS")
