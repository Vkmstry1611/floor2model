import json, struct, base64, numpy as np
from pathlib import Path

path = Path("generated_models/floorplan1/floorplan1.gltf")
gltf = json.loads(path.read_text())

print("Nodes:", [n.get("name") for n in gltf.get("nodes", [])])
print("Meshes:", [m.get("name") for m in gltf.get("meshes", [])])
print("Materials:", [m.get("name") for m in gltf.get("materials", [])])
print("Textures:", len(gltf.get("textures", [])))
print("Images:", len(gltf.get("images", [])))

# Check mesh primitives
for mesh in gltf.get("meshes", []):
    for prim in mesh.get("primitives", []):
        attrs = list(prim.get("attributes", {}).keys())
        mat = prim.get("material")
        print(f"  Mesh '{mesh['name']}': attrs={attrs}, material={mat}")

# Decode buffer and check vertex range
buf_uri = gltf["buffers"][0]["uri"]
if buf_uri.startswith("data:"):
    b64 = buf_uri.split(",", 1)[1]
    data = base64.b64decode(b64)
    print(f"\nBuffer size: {len(data)/1024:.1f} KB")

# Find position accessor for building mesh
for mesh in gltf.get("meshes", []):
    if mesh["name"] not in ("InteriorFloor", "FloorPlan"):
        for prim in mesh.get("primitives", []):
            pos_acc_idx = prim["attributes"].get("POSITION")
            if pos_acc_idx is not None:
                acc = gltf["accessors"][pos_acc_idx]
                print(f"\nBuilding mesh vertices: {acc['count']}")
                print(f"  min: {acc.get('min')}")
                print(f"  max: {acc.get('max')}")
