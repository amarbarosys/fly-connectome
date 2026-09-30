"""
Cinematic High-Fidelity 1:1 Biological Virtual Fly Body (Blender 5.2 + Cycles HIP).
Hardware: AMD Radeon RX 7900 XTX 24GB.
"""

import bpy
import numpy as np
import math
import os

def create_emission_mat(name, color, strength=8.0):
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

def create_chitin_glass_mat(name, base_color=(0.14, 0.09, 0.05, 1.0), roughness=0.18, ior=1.52, transmission=0.86):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = ior
    bsdf.inputs["Transmission Weight"].default_value = transmission
    bsdf.inputs["Subsurface Weight"].default_value = 0.05
    bsdf.inputs["Subsurface Radius"].default_value = (0.2, 0.1, 0.05)
    
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat

def create_eye_facet_mat(name):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.45, 0.03, 0.02, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.25
    bsdf.inputs["IOR"].default_value = 1.55
    bsdf.inputs["Metallic"].default_value = 0.15
    bsdf.inputs["Transmission Weight"].default_value = 0.35
    
    # Voronoi micro-bump for ommatidia facets
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    voronoi = nodes.new(type="ShaderNodeTexVoronoi")
    voronoi.feature = "F1"
    voronoi.distance = "EUCLIDEAN"
    voronoi.inputs["Scale"].default_value = 140.0
    
    bump = nodes.new(type="ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.22
    bump.inputs["Distance"].default_value = 0.05
    
    mat.node_tree.links.new(tex_coord.outputs["Object"], voronoi.inputs["Vector"])
    mat.node_tree.links.new(voronoi.outputs["Distance"], bump.inputs["Height"])
    mat.node_tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    
    output = nodes.new(type="ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat

def create_curve_tube(name, points, radius=0.015, material=None):
    curve_data = bpy.data.curves.new(name=name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 6
    
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
    print("Initializing clean Blender scene...")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Cycles Setup for AMD RX 7900 XTX
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    cycles_pref = bpy.context.preferences.addons["cycles"].preferences
    cycles_pref.compute_device_type = "HIP"
    cycles_pref.get_devices()
    for d in cycles_pref.devices:
        if d.type == "HIP":
            d.use = True
        else:
            d.use = False
            
    scene.cycles.device = "GPU"
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.render.resolution_x = 2560
    scene.render.resolution_y = 1440
    
    # World background
    world = bpy.data.worlds.new("StudioWorld")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.012, 0.014, 0.02, 1.0)
    bg.inputs["Strength"].default_value = 0.25
    scene.world = world
    
    # Collections
    col_brain = bpy.data.collections.new("01_Connectome_Brain")
    col_nerves = bpy.data.collections.new("02_Nerve_Tracts_1to1")
    col_body = bpy.data.collections.new("03_Exoskeleton_Body")
    col_lighting = bpy.data.collections.new("04_Studio_Lighting")
    
    scene.collection.children.link(col_brain)
    scene.collection.children.link(col_nerves)
    scene.collection.children.link(col_body)
    scene.collection.children.link(col_lighting)
    
    # Materials
    mat_optic = create_emission_mat("Mat_OpticLobe", (0.05, 0.9, 1.0, 1.0), strength=12.0)
    mat_central = create_emission_mat("Mat_CentralComplex", (1.0, 0.72, 0.08, 1.0), strength=10.0)
    mat_descending = create_emission_mat("Mat_DescendingHighway", (1.0, 0.08, 0.65, 1.0), strength=16.0)
    mat_motor = create_emission_mat("Mat_MotorNeuron", (1.0, 0.15, 0.1, 1.0), strength=18.0)
    mat_nerve_cord = create_emission_mat("Mat_VentralCord", (0.2, 1.0, 0.5, 1.0), strength=14.0)
    
    mat_cuticle = create_chitin_glass_mat("Mat_GlassCuticle", (0.12, 0.08, 0.04, 1.0), roughness=0.16, transmission=0.88)
    mat_eye = create_eye_facet_mat("Mat_CompoundEye")
    mat_wing = create_chitin_glass_mat("Mat_WingGlass", (0.85, 0.9, 0.95, 1.0), roughness=0.08, transmission=0.95)
    mat_leg = create_chitin_glass_mat("Mat_LegChitin", (0.16, 0.12, 0.08, 1.0), roughness=0.22, transmission=0.75)
    
    # 1. Real Brain Somas
    npz_path = "data/fly_somas_blender.npz"
    if os.path.exists(npz_path):
        print("Instantiating Connectome Somas...")
        data = np.load(npz_path)
        coords = data["coords"]
        class_ids = data["class_ids"]
        
        # Dense sampling: all descending & motor, 1 in 3 of optic/central
        mask = (class_ids == 4) | (class_ids == 5) | ((np.arange(len(coords)) % 3) == 0)
        sub_coords = coords[mask]
        sub_classes = class_ids[mask]
        
        groups = {
            "Optic_Lobes": (sub_coords[(sub_classes == 0) | (sub_classes == 3)], mat_optic, 0.0032),
            "Central_Complex": (sub_coords[(sub_classes == 1) | (sub_classes == 2)], mat_central, 0.0035),
            "Descending_Command": (sub_coords[sub_classes == 4], mat_descending, 0.0042),
            "Motor_Outputs": (sub_coords[sub_classes == 5], mat_motor, 0.005),
        }
        
        for g_name, (pts, g_mat, radius) in groups.items():
            if len(pts) == 0:
                continue
            mesh = bpy.data.meshes.new(f"Mesh_{g_name}")
            mesh.from_pydata(pts.tolist(), [], [])
            mesh.update()
            
            obj = bpy.data.objects.new(f"Brain_{g_name}", mesh)
            col_brain.objects.link(obj)
            
            geo_mod = obj.modifiers.new("PointVisualizer", "NODES")
            ng = bpy.data.node_groups.new(f"Geo_{g_name}", "GeometryNodeTree")
            ng.is_modifier = True
            
            in_node = ng.nodes.new("NodeGroupInput")
            out_node = ng.nodes.new("NodeGroupOutput")
            ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
            ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
            
            pts_node = ng.nodes.new("GeometryNodeMeshToPoints")
            pts_node.inputs["Radius"].default_value = radius
            
            inst_node = ng.nodes.new("GeometryNodeInstanceOnPoints")
            ico = ng.nodes.new("GeometryNodeMeshIcoSphere")
            ico.inputs["Radius"].default_value = radius
            ico.inputs["Subdivisions"].default_value = 1
            
            mat_set = ng.nodes.new("GeometryNodeSetMaterial")
            mat_set.inputs["Material"].default_value = g_mat
            
            ng.links.new(in_node.outputs["Geometry"], pts_node.inputs["Mesh"])
            ng.links.new(pts_node.outputs["Points"], inst_node.inputs["Points"])
            ng.links.new(ico.outputs["Mesh"], inst_node.inputs["Instance"])
            ng.links.new(inst_node.outputs["Instances"], mat_set.inputs["Geometry"])
            ng.links.new(mat_set.outputs["Geometry"], out_node.inputs["Geometry"])
            
            geo_mod.node_group = ng
            
    # 2. 1:1 Nerve Tracts (Dual Sheath: Inner Emission + Outer Translucent Guide)
    print("Building glowing nerve tracts...")
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children["02_Nerve_Tracts_1to1"]
    
    # Cervical Connective (Brain to Ventral Thoracic Cord)
    create_curve_tube("Nerve_CervicalConnective", [
        (0.0, -0.10, -0.04),
        (0.0, -0.22, -0.12),
        (0.0, -0.42, -0.18),
        (0.0, -0.75, -0.25),
    ], radius=0.024, material=mat_descending)
    
    # Thoracic Spine Ganglion
    create_curve_tube("Nerve_ThoracicCord", [
        (0.0, -0.75, -0.25),
        (0.0, -1.05, -0.25),
        (0.0, -1.35, -0.22),
    ], radius=0.028, material=mat_nerve_cord)
    
    # Optic Nerves into Eye Peduncles
    create_curve_tube("Nerve_Optic_L", [(-0.16, 0.0, 0.0), (-0.35, 0.02, 0.02), (-0.45, 0.04, 0.0)], radius=0.020, material=mat_optic)
    create_curve_tube("Nerve_Optic_R", [(0.16, 0.0, 0.0), (0.35, 0.02, 0.02), (0.45, 0.04, 0.0)], radius=0.020, material=mat_optic)
    
    # Wing Motor Roots
    create_curve_tube("Nerve_Wing_L", [(0.0, -0.75, -0.25), (-0.18, -0.70, 0.05), (-0.34, -0.65, 0.28)], radius=0.016, material=mat_motor)
    create_curve_tube("Nerve_Wing_R", [(0.0, -0.75, -0.25), (0.18, -0.70, 0.05), (0.34, -0.65, 0.28)], radius=0.016, material=mat_motor)
    
    # 6 Leg Motor Roots
    for l_name, pts in [
        ("Nerve_LegF_L", [(-0.1, -0.65, -0.25), (-0.30, -0.55, -0.42)]),
        ("Nerve_LegF_R", [(0.1, -0.65, -0.25), (0.30, -0.55, -0.42)]),
        ("Nerve_LegM_L", [(-0.12, -0.80, -0.25), (-0.38, -0.80, -0.44)]),
        ("Nerve_LegM_R", [(0.12, -0.80, -0.25), (0.38, -0.80, -0.44)]),
        ("Nerve_LegH_L", [(-0.1, -0.98, -0.25), (-0.34, -1.05, -0.42)]),
        ("Nerve_LegH_R", [(0.1, -0.98, -0.25), (0.34, -1.05, -0.42)]),
    ]:
        create_curve_tube(l_name, [(0.0, pts[0][1], -0.25)] + pts, radius=0.014, material=mat_motor)
        
    # 3. Exoskeleton (Glass Cuticle & Faceted Eyes)
    print("Building optical cuticle exoskeleton...")
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children["03_Exoskeleton_Body"]
    
    # Head Capsule
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=36, radius=0.48, location=(0.0, 0.01, -0.02))
    head = bpy.context.active_object
    head.name = "Body_HeadCapsule"
    head.scale = (1.02, 0.74, 0.88)
    head.data.materials.append(mat_cuticle)
    bpy.ops.object.shade_smooth()
    
    # Compound Eyes (Faceted Ruby Ommatidia)
    for eye_name, eye_x in [("Body_CompoundEye_L", -0.44), ("Body_CompoundEye_R", 0.44)]:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=36, radius=0.29, location=(eye_x, 0.02, -0.01))
        eye = bpy.context.active_object
        eye.name = eye_name
        eye.scale = (0.76, 1.16, 1.16)
        eye.data.materials.append(mat_eye)
        bpy.ops.object.shade_smooth()
        
    # Thorax
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=36, radius=0.56, location=(0.0, -0.76, -0.12))
    thorax = bpy.context.active_object
    thorax.name = "Body_Thorax"
    thorax.scale = (0.86, 1.28, 0.94)
    thorax.data.materials.append(mat_cuticle)
    bpy.ops.object.shade_smooth()
    
    # Abdomen with subtle segment rings
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=36, radius=0.64, location=(0.0, -1.86, -0.22))
    abdomen = bpy.context.active_object
    abdomen.name = "Body_Abdomen"
    abdomen.scale = (0.76, 1.58, 0.72)
    abdomen.data.materials.append(mat_cuticle)
    bpy.ops.object.shade_smooth()
    
    # Wings (Glass foils with vein lines)
    for w_name, w_x, rot_z in [("Body_Wing_L", -0.34, 0.20), ("Body_Wing_R", 0.34, -0.20)]:
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(w_x, -1.5, 0.36))
        wing = bpy.context.active_object
        wing.name = w_name
        wing.scale = (0.46, 1.9, 0.005)
        wing.rotation_euler = (math.radians(6), math.radians(rot_z * 22), rot_z)
        wing.data.materials.append(mat_wing)
        
        # Wing leading vein
        create_curve_tube(f"{w_name}_VeinLeading", [
            (w_x, -0.65, 0.28),
            (w_x + (w_x * 0.4), -1.5, 0.36),
            (w_x + (w_x * 0.2), -2.4, 0.34)
        ], radius=0.008, material=mat_cuticle)
        
    # 6 Articulated Legs (Jointed segments)
    legs_config = [
        ("Leg_L1_Front", -0.32, -0.55, -0.42, -0.52, 0.48),
        ("Leg_R1_Front", 0.32, -0.55, -0.42, 0.52, 0.48),
        ("Leg_L2_Mid",   -0.38, -0.80, -0.44, -0.74, -0.12),
        ("Leg_R2_Mid",   0.38, -0.80, -0.44, 0.74, -0.12),
        ("Leg_L3_Hind",  -0.34, -1.05, -0.42, -0.62, -0.82),
        ("Leg_R3_Hind",  0.34, -1.05, -0.42, 0.62, -0.82),
    ]
    for leg_name, sx, sy, sz, dx, dy in legs_config:
        p0 = (sx, sy, sz)
        p1 = (sx + dx * 0.35, sy + dy * 0.25, sz - 0.32)
        p2 = (sx + dx * 0.75, sy + dy * 0.65, sz - 0.72)
        p3 = (sx + dx, sy + dy, sz - 1.05)
        create_curve_tube(f"Body_{leg_name}", [p0, p1, p2, p3], radius=0.026, material=mat_leg)
        
    # 4. Cinematic Studio Lighting
    print("Setting up cinematic studio lights...")
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children["04_Studio_Lighting"]
    
    # Key Area Light (Cool Ice White 6500K)
    bpy.ops.object.light_add(type="AREA", location=(-2.6, -1.4, 2.2))
    key = bpy.context.active_object
    key.name = "Light_KeyCool"
    key.data.energy = 160.0
    key.data.size = 2.4
    key.data.color = (0.88, 0.94, 1.0)
    
    # Rim Back Light (Warm Amber 2800K - creates specular chitin rim highlights)
    bpy.ops.object.light_add(type="POINT", location=(2.2, -2.8, 0.8))
    rim = bpy.context.active_object
    rim.name = "Light_RimAmber"
    rim.data.energy = 120.0
    rim.data.shadow_soft_size = 0.3
    rim.data.color = (1.0, 0.55, 0.15)
    
    # Accent Under Light (Deep Electric Cyan)
    bpy.ops.object.light_add(type="POINT", location=(0.0, -0.8, -2.0))
    under = bpy.context.active_object
    under.name = "Light_UnderGlow"
    under.data.energy = 80.0
    under.data.color = (0.05, 0.6, 1.0)
    
    # Camera with Depth of Field
    print("Configuring 85mm macro camera with DoF...")
    bpy.ops.object.camera_add(location=(-2.4, -2.8, 1.35), rotation=(math.radians(68), 0, math.radians(-38)))
    cam = bpy.context.active_object
    cam.name = "Camera_HeroBio"
    scene.camera = cam
    cam.data.lens = 75
    
    # Depth of Field focused on brain center
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 3.65
    cam.data.dof.aperture_fstop = 4.0
    
    # Output paths
    blend_path = os.path.abspath("fly_virtual_body.blend")
    img_path = os.path.abspath("fly_virtual_body.png")
    scene.render.filepath = img_path
    
    print(f"Saving Blender project to {blend_path}...")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    
    print(f"Rendering Cycles HIP 2.5K image to {img_path}...")
    bpy.ops.render.render(write_still=True)
    print("Cycles GPU render finished successfully!")

if __name__ == "__main__":
    main()
