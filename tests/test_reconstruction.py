import numpy as np
import pytest
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.reconstruction.extruder import Extruder, Mesh3D
from src.reconstruction.mesh_builder import MeshBuilder, Building3D
from src.reconstruction.exporter import ModelExporter


def make_square_polygon(size=100, offset=(0, 0)):
    x, y = offset
    return [(x, y), (x+size, y), (x+size, y+size), (x, y+size)]

def make_simple_mesh():
    verts = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1],
    ], dtype=np.float32)
    faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    return Mesh3D(verts, faces, label="OuterWall", color=(0.5,0.5,0.5))

def make_simple_building():
    mesh = make_simple_mesh()
    colors = np.tile(np.array([0.5,0.5,0.5]),(8,1)).astype(np.float32)
    return Building3D(
        meshes=[mesh], merged_vertices=mesh.vertices.copy(),
        merged_faces=mesh.faces, merged_colors=colors,
        wall_height=2.8, floor_area_m2=25.0, room_count=1,
    )

class TestExtruder:
    def test_extrude_square_polygon(self):
        e = Extruder(wall_height=2.8, pixels_per_metre=100.0)
        mesh = e.extrude_polygon(make_square_polygon(100), label="OuterWall")
        assert mesh.vertex_count > 0
        assert mesh.face_count > 0

    def test_extrude_creates_correct_height(self):
        e = Extruder(wall_height=3.0, pixels_per_metre=100.0)
        mesh = e.extrude_polygon(make_square_polygon(100))
        assert abs((mesh.vertices[:,2].max() - mesh.vertices[:,2].min()) - 3.0) < 0.01

    def test_degenerate_polygon_returns_empty(self):
        e = Extruder()
        mesh = e.extrude_polygon([(0,0),(1,1)], label="Wall")
        assert mesh.vertex_count == 0

    def test_correct_label_assigned(self):
        e = Extruder()
        mesh = e.extrude_polygon(make_square_polygon(100), label="Bedroom")
        assert mesh.label == "Bedroom"

    def test_normals_computed(self):
        e = Extruder()
        mesh = e.extrude_polygon(make_square_polygon(100))
        assert mesh.normals is not None
        assert mesh.normals.shape == mesh.vertices.shape

    def test_floor_slab_below_zero(self):
        e = Extruder(floor_thickness=0.2, pixels_per_metre=100.0)
        floor = e.extrude_floor_slab(make_square_polygon(100))
        assert floor.vertices[:,2].min() < 0

class TestMeshBuilder:
    def test_merge_single_mesh(self):
        mesh = make_simple_mesh()
        b = MeshBuilder(add_floor=False).build([mesh])
        assert b.vertex_count == mesh.vertex_count

    def test_merge_multiple_meshes(self):
        m1, m2 = make_simple_mesh(), make_simple_mesh()
        b = MeshBuilder(add_floor=False).build([m1, m2])
        assert b.vertex_count == m1.vertex_count + m2.vertex_count

    def test_colours_shape_matches_vertices(self):
        b = MeshBuilder(add_floor=False).build([make_simple_mesh()])
        assert b.merged_colors.shape == b.merged_vertices.shape

    def test_raises_on_empty_meshes(self):
        with pytest.raises(ValueError):
            MeshBuilder().build([])

    def test_summary_keys(self):
        s = make_simple_building().summary
        assert all(k in s for k in ["vertices","faces","wall_height","floor_area"])

class TestExporter:
    def test_export_obj_creates_file(self, tmp_path):
        paths = ModelExporter(export_obj=True, export_gltf=False).export(
            make_simple_building(), str(tmp_path), "test")
        assert os.path.exists(paths["obj"])

    def test_export_obj_valid_content(self, tmp_path):
        paths = ModelExporter(export_obj=True, export_gltf=False).export(
            make_simple_building(), str(tmp_path), "test")
        content = open(paths["obj"]).read()
        assert "v " in content and "f " in content

    def test_export_gltf_creates_file(self, tmp_path):
        paths = ModelExporter(export_obj=False, export_gltf=True).export(
            make_simple_building(), str(tmp_path), "test")
        assert os.path.exists(paths["gltf"])

    def test_export_gltf_valid_json(self, tmp_path):
        paths = ModelExporter(export_obj=False, export_gltf=True).export(
            make_simple_building(), str(tmp_path), "test")
        gltf = json.load(open(paths["gltf"]))
        assert gltf["asset"]["version"] == "2.0"

    def test_export_stl_creates_file(self, tmp_path):
        paths = ModelExporter(export_obj=False, export_gltf=False, export_stl=True).export(
            make_simple_building(), str(tmp_path), "test")
        assert os.path.getsize(paths["stl"]) > 84

    def test_triangulate_quad(self):
        tris = ModelExporter()._triangulate([[0,1,2,3]])
        assert len(tris) == 2 and all(len(t)==3 for t in tris)

    def test_triangulate_triangle_unchanged(self):
        tris = ModelExporter()._triangulate([[0,1,2]])
        assert len(tris) == 1
