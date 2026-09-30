"""
Build 1:1 Anatomical Virtual Fly Body connected to FlyWire Connectome in Blender 5.2.
"""

import bpy
import numpy as np
import math
import os

def create_emission_material(name, color, strength=5.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    emit = nodes.new(type="ShaderNodeEmission")
    emit.inputs["Color"].default_value = color
    emit.inputs["Strength"].default_value = strength
    
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(emit.outputs["Emission"], output.inputs["Surface"])
    return mat

def create_glass_chitin_material(name, color, roughness=0.2, ior=1.45, transmission=0.82):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = ior
    bsdf.inputs["Transmission Weight"].default_value = transmission
    
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat

def create_curve_tube(name, points, radius=0.015, material=None):
    curve_data = bpy.data.curves.new(name=name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 4
    
    spline = curve_data.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    
    for i, pt in enumerate(points):
        bp = spline.bezier_points[i]
        bp.co = pt
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
        
    obj = bpy.data.objects.new(name, curve_data)
    if material:
        obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)
    return obj

def main():
    print("Clearing scene...")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # 1. World background
    world = bpy.data.worlds.new("BioWorld")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.01, 0.015, 0.025, 1.0)
    bg.inputs["Strength"].default_value = 0.5
    bpy.context.scene.world = world
    
    # Collections
    col_brain = bpy.data.collections.new("01_Connectome_Brain")
    col_nerves = bpy.data.collections.new("02_Nerve_Tracts_1to1")
    col_body = bpy.data.collections.new("03_Exoskeleton_Body")
    col_rig = bpy.data.collections.new("04_Armature_Rig")
    
    bpy.context.scene.collection.children.link(col_brain)
    bpy.context.scene.collection.children.link(col_nerves)
    bpy.context.scene.collection.children.link(col_body)
    bpy.context.scene.collection.children.link(col_rig)
    
    # Materials
    mat_optic = create_emission_material("Mat_OpticLobe", (0.0, 0.85, 1.0, 1.0), strength=4.0)       # Cyan
    mat_central = create_emission_material("Mat_CentralBrain", (1.0, 0.65, 0.05, 1.0), strength=3.5)   # Amber
    mat_descending = create_emission_material("Mat_DescendingNerve", (1.0, 0.1, 0.6, 1.0), strength=6.0) # Magenta
    mat_motor = create_emission_material("Mat_MotorNeuron", (1.0, 0.05, 0.05, 1.0), strength=8.0)      # Crimson
    mat_nerve_cord = create_emission_material("Mat_VentralCord", (0.3, 1.0, 0.4, 1.0), strength=5.0)   # Neon Green
    
    mat_cuticle = create_glass_chitin_material("Mat_ChitinCuticle", (0.12, 0.08, 0.05, 1.0), roughness=0.25, transmission=0.88)
    mat_eye = create_glass_chitin_material("Mat_CompoundEye", (0.6, 0.05, 0.02, 1.0), roughness=0.35, transmission=0.45)
    mat_wing = create_glass_chitin_material("Mat_WingMembrane", (0.8, 0.85, 0.9, 1.0), roughness=0.1, transmission=0.94)
    mat_leg = create_glass_chitin_material("Mat_LegChitin", (0.2, 0.15, 0.1, 1.0), roughness=0.3, transmission=0.7)
    
    # 2. Load Real Connectome Somas
    npz_path = "data/fly_somas_blender.npz"
    if os.path.exists(npz_path):
        print(f"Loading connectome somas from {npz_path}...")
        data = np.load(npz_path)
        coords = data["coords"]
        class_ids = data["class_ids"]
        
        # Subsample for dense viewport rendering:
        # Keep 100% of motor (5) and descending (4), sample 1 in 4 of optic/central
        mask = (class_ids == 4) | (class_ids == 5) | ((np.arange(len(coords)) % 4) == 0)
        sub_coords = coords[mask]
        sub_classes = class_ids[mask]
        print(f"Rendering {len(sub_coords):,} instantiated neural somas in Blender...")
        
        # Split into groups for distinct shader glows
        groups = {
            "Optic_Lobes": (sub_coords[(sub_classes == 0) | (sub_classes == 3)], mat_optic),
            "Central_Complex": (sub_coords[(sub_classes == 1) | (sub_classes == 2)], mat_central),
            "Descending_Command": (sub_coords[sub_classes == 4], mat_descending),
            "Motor_Outputs": (sub_coords[sub_classes == 5], mat_motor),
        }
        
        for g_name, (pts, g_mat) in groups.items():
            if len(pts) == 0:
                continue
            mesh = bpy.data.meshes.new(f"Mesh_{g_name}")
            mesh.from_pydata(pts.tolist(), [], [])
            mesh.update()
            
            obj = bpy.data.objects.new(f"Brain_{g_name}", mesh)
            obj.data.materials.append(g_mat)
            col_brain.objects.link(obj)
            
            # Geometry node or small icosphere instancing via modifier
            geo_mod = obj.modifiers.new("PointVisualizer", "NODES")
            # Create node tree
            ng = bpy.data.node_groups.new(f"Geo_{g_name}", "GeometryNodeTree")
            ng.is_modifier = True
            
            in_node = ng.nodes.new("NodeGroupInput")
            out_node = ng.nodes.new("NodeGroupOutput")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            
            pts_to_mesh = ng.nodes.new("GeometryNodeMeshToPoints")
            pts_to_mesh.inputs["Radius"].default_value = 0.0035 # 3.5 microns
            
            instance_sphere = ng.nodes.new("GeometryNodeInstanceOnPoints")
            sphere = ng.nodes.new("GeometryNodeMeshIcoSphere")
            sphere.inputs["Radius"].default_value = 0.003
            sphere.inputs["Subdivisions"].default_value = 1
            
            mat_set = ng.nodes.new("GeometryNodeSetMaterial")
            mat_set.inputs["Material"].default_value = g_mat
            
            ng.links.new(in_node.outputs["Geometry"], pts_to_mesh.inputs["Mesh"])
            ng.links.new(pts_to_mesh.outputs["Points"], instance_sphere.inputs["Points"])
            ng.links.new(sphere.outputs["Mesh"], instance_sphere.inputs["Instance"])
            ng.links.new(instance_sphere.outputs["Instances"], mat_set.inputs["Geometry"])
            ng.links.new(mat_set.outputs["Geometry"], out_node.inputs["Geometry"])
            
            geo_mod.node_group = ng
            
    # 3. 1:1 Nerve Tract Cables
    print("Building 1:1 Nerve Tracts...")
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children["02_Nerve_Tracts_1to1"]
    
    # Cervical Connective: Brain (0, -0.12, -0.05) -> Neck (0, -0.3, -0.15) -> Thoracic Ganglion (0, -0.75, -0.25)
    create_curve_tube("Nerve_CervicalConnective_Neck", [
        (0.0, -0.12, -0.05),
        (0.0, -0.22, -0.12),
        (0.0, -0.40, -0.18),
        (0.0, -0.75, -0.25),
    ], radius=0.022, material=mat_descending)
    
    # Thoracic Ganglion Core
    create_curve_tube("Nerve_VentralNerveCord_Spine", [
        (0.0, -0.75, -0.25),
        (0.0, -1.10, -0.25),
        (0.0, -1.40, -0.22),
    ], radius=0.028, material=mat_nerve_cord)
    
    # Optic Nerve Bundles (L & R to Compound Eyes)
    create_curve_tube("Nerve_OpticTract_L", [(-0.15, 0.0, 0.0), (-0.35, 0.02, 0.02), (-0.45, 0.05, 0.0)], radius=0.018, material=mat_optic)
    create_curve_tube("Nerve_OpticTract_R", [(0.15, 0.0, 0.0), (0.35, 0.02, 0.02), (0.45, 0.05, 0.0)], radius=0.018, material=mat_optic)
    
    # Antennal Nerves (Front of head)
    create_curve_tube("Nerve_Antennal_L", [(-0.06, 0.08, -0.02), (-0.10, 0.18, 0.05), (-0.12, 0.28, 0.08)], radius=0.009, material=mat_central)
    create_curve_tube("Nerve_Antennal_R", [(0.06, 0.08, -0.02), (0.10, 0.18, 0.05), (0.12, 0.28, 0.08)], radius=0.009, material=mat_central)
    
    # Wing Motor Nerves (Thorax to Wing Hinges)
    create_curve_tube("Nerve_WingMotor_L", [(0.0, -0.75, -0.25), (-0.18, -0.70, 0.05), (-0.35, -0.65, 0.28)], radius=0.015, material=mat_motor)
    create_curve_tube("Nerve_WingMotor_R", [(0.0, -0.75, -0.25), (0.18, -0.70, 0.05), (0.35, -0.65, 0.28)], radius=0.015, material=mat_motor)
    
    # 6 Leg Motor Nerves (Thorax ganglion to Coxae)
    leg_nerve_targets = [
        ("Nerve_LegF_L", [(-0.1, -0.65, -0.25), (-0.28, -0.55, -0.42)]),
        ("Nerve_LegF_R", [(0.1, -0.65, -0.25), (0.28, -0.55, -0.42)]),
        ("Nerve_LegM_L", [(-0.12, -0.80, -0.25), (-0.35, -0.80, -0.44)]),
        ("Nerve_LegM_R", [(0.12, -0.80, -0.25), (0.35, -0.80, -0.44)]),
        ("Nerve_LegH_L", [(-0.1, -0.98, -0.25), (-0.32, -1.05, -0.42)]),
        ("Nerve_LegH_R", [(0.1, -0.98, -0.25), (0.32, -1.05, -0.42)]),
    ]
    for l_name, pts in leg_nerve_targets:
        create_curve_tube(l_name, [(0.0, pts[0][1], -0.25)] + pts, radius=0.012, material=mat_motor)
        
    # 4. Anatomical Exoskeleton (Body Shell)
    print("Building Exoskeleton body shell...")
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children["03_Exoskeleton_Body"]
    
    # Head Capsule
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=24, radius=0.46, location=(0.0, 0.0, -0.02))
    head_obj = bpy.context.active_object
    head_obj.name = "Body_HeadCapsule"
    head_obj.scale = (1.05, 0.72, 0.88)
    head_obj.data.materials.append(mat_cuticle)
    bpy.ops.object.shade_smooth()
    
    # Compound Eyes (Left & Right)
    for eye_name, eye_x in [("Body_CompoundEye_L", -0.43), ("Body_CompoundEye_R", 0.43)]:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=28, ring_count=20, radius=0.28, location=(eye_x, 0.02, -0.01))
        eye = bpy.context.active_object
        eye.name = eye_name
        eye.scale = (0.75, 1.15, 1.15)
        eye.data.materials.append(mat_eye)
        bpy.ops.object.shade_smooth()
        
    # Thorax
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=24, radius=0.55, location=(0.0, -0.75, -0.12))
    thorax = bpy.context.active_object
    thorax.name = "Body_Thorax"
    thorax.scale = (0.85, 1.25, 0.95)
    thorax.data.materials.append(mat_cuticle)
    bpy.ops.object.shade_smooth()
    
    # Abdomen
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=24, radius=0.62, location=(0.0, -1.85, -0.22))
    abdomen = bpy.context.active_object
    abdomen.name = "Body_Abdomen"
    abdomen.scale = (0.78, 1.55, 0.72)
    abdomen.data.materials.append(mat_cuticle)
    bpy.ops.object.shade_smooth()
    
    # Wings (Translucent dorsal flight foils)
    for w_name, w_x, rot_z in [("Body_Wing_L", -0.32, 0.22), ("Body_Wing_R", 0.32, -0.22)]:
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(w_x, -1.5, 0.35))
        wing = bpy.context.active_object
        wing.name = w_name
        wing.scale = (0.48, 1.85, 0.01)
        wing.rotation_euler = (math.radians(8), math.radians(rot_z * 25), rot_z)
        wing.data.materials.append(mat_wing)
        
    # 6 Articulated Legs
    legs_config = [
        # (name, start_x, start_y, start_z, sweep_x, sweep_y)
        ("Leg_L1_Front", -0.32, -0.55, -0.42, -0.55, 0.45),
        ("Leg_R1_Front", 0.32, -0.55, -0.42, 0.55, 0.45),
        ("Leg_L2_Mid",   -0.38, -0.80, -0.44, -0.75, -0.15),
        ("Leg_R2_Mid",   0.38, -0.80, -0.44, 0.75, -0.15),
        ("Leg_L3_Hind",  -0.34, -1.05, -0.42, -0.65, -0.85),
        ("Leg_R3_Hind",  0.34, -1.05, -0.42, 0.65, -0.85),
    ]
    for leg_name, sx, sy, sz, dx, dy in legs_config:
        # Femur + Tibia joint curve
        p0 = (sx, sy, sz)
        p1 = (sx + dx * 0.4, sy + dy * 0.3, sz - 0.35)
        p2 = (sx + dx * 0.8, sy + dy * 0.7, sz - 0.75)
        p3 = (sx + dx, sy + dy, sz - 1.1)
        create_curve_tube(f"Body_{leg_name}", [p0, p1, p2, p3], radius=0.03, material=mat_leg)
        
    # 5. Armature Rig & Drivers
    print("Building Armature rig...")
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children["04_Armature_Rig"]
    
    arm_data = bpy.data.armatures.new("Armature_VirtualFly")
    arm_obj = bpy.data.objects.new("VirtualFly_Rig", arm_data)
    col_rig.objects.link(arm_obj)
    
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    
    # Spine Bones
    b_root = arm_data.edit_bones.new("Bone_Root")
    b_root.head = (0, -0.75, -0.12)
    b_root.tail = (0, -0.75, 0.2)
    
    b_head = arm_data.edit_bones.new("Bone_Head")
    b_head.head = (0, -0.15, -0.05)
    b_head.tail = (0, 0.25, 0.0)
    b_head.parent = b_root
    
    b_wing_l = arm_data.edit_bones.new("Bone_Wing_L")
    b_wing_l.head = (-0.32, -0.65, 0.28)
    b_wing_l.tail = (-0.45, -1.8, 0.35)
    b_wing_l.parent = b_root
    
    b_wing_r = arm_data.edit_bones.new("Bone_Wing_R")
    b_wing_r.head = (0.32, -0.65, 0.28)
    b_wing_r.tail = (0.45, -1.8, 0.35)
    b_wing_r.parent = b_root
    
    bpy.ops.object.mode_set(mode="OBJECT")
    
    # Add custom biological controller properties to Armature
    arm_obj["neuron_input_visual_looming"] = 0.0
    arm_obj["neuron_p1_courtship_song"] = 0.0
    arm_obj["motor_wing_left_amplitude"] = 0.0
    arm_obj["motor_wing_right_amplitude"] = 0.0
    arm_obj["motor_giant_fiber_jump"] = 0
    arm_obj["brain_tick_rate_fps"] = 239.7
    
    # 6. Camera & Lighting
    print("Setting up cinematic camera and lighting...")
    # Key light
    bpy.ops.object.light_add(type="AREA", location=(-2.2, -1.2, 2.5))
    key_light = bpy.context.active_object
    key_light.data.energy = 85.0
    key_light.data.size = 2.0
    key_light.data.color = (0.85, 0.95, 1.0)
    
    # Rim light behind for cuticle transmission
    bpy.ops.object.light_add(type="POINT", location=(1.8, -2.5, -0.5))
    rim_light = bpy.context.active_object
    rim_light.data.energy = 60.0
    rim_light.data.color = (1.0, 0.5, 0.2)
    
    # Camera at beauty 3/4 perspective
    bpy.ops.object.camera_add(location=(-2.8, -3.2, 1.8), rotation=(math.radians(65), 0, math.radians(-40)))
    cam = bpy.context.active_object
    cam.name = "Camera_BioBeauty"
    bpy.context.scene.camera = cam
    cam.data.lens = 65
    
    # Render settings
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.filepath = os.path.abspath("fly_virtual_body.png")
    
    # Save .blend
    blend_path = os.path.abspath("fly_virtual_body.blend")
    print(f"Saving project to {blend_path}...")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    
    print(f"Rendering high-res viewport to {scene.render.filepath}...")
    bpy.ops.render.render(write_still=True)
    print("Render complete!")

if __name__ == "__main__":
    main()
