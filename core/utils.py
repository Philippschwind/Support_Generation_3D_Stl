from pydantic import BaseModel
import open3d as o3d
from typing import List, Dict
import numpy as np
import trimesh as tm

class SupportParams(BaseModel):
    name:str
    generateSupport: bool
    supportType: float
    supportOnPart: bool
    reduceResolution: bool
    meanEdgeLength: float
    angleThresholdDownSkin: float
    angleThresholdBottomSkin: float
    angleStepsSupport: float
    targetZHeight: float
    DistanceBarSupport: float
    SupportDiameterBase: float
    SupportDiameterMiddle: float
    SupportDiameterPart: float
    SupportHeightBase: float
    SupportHeightTip: float
    SupportPenetrationDepth: float
    minGridSpacing: float
    maxGridSpacing: float
    overhangAngle: float
    minSupportDistance: float
    supportCutoffHeight: float


def create_support(support_payload:SupportPayload) -> None:
    order_name = support_payload.order_name
    support_params = support_payload.support_params

    mesh = load_stl_object(order_name)

    average_centroid, name = get_average_centroid(order_name)

    mesh = transform_mesh_vertices(mesh, True, average_centroid)
    upload_stl_object(order_name, f"05_{name}_Transormed_Part.stl", mesh)
    if support_params.reduceResolution:
        mesh = reduce_stl_resolution(mesh, support_params.meanEdgeLength)
    file_name = f"05{name}_Support.stl"

    if support_params.supportType == 0:
        generate_support(mesh, support_params, order_name, file_name)
    elif support_params.supportType == 1:
        generate_grid_support(mesh, support_params, order_name, file_name)


def rotate_mesh_180(mesh):

    vertices = mesh.vertices
    faces = mesh.faces

    R = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    transformedVertices = (R @ vertices.T).T
    transformedVertices = transformedVertices[:, :3]

    return tm.Trimesh(transformedVertices, faces)

def get_average_centroid(order_name):
    """ Calculates the average centroid for a stl file by using the construction_info file.
        This is necessary so the later transformation of the vertices fits.

    Args:
        order_name (str):
    return:
        average_centroid
        tooth_names
    """

    root = load_construction_info(order_name)

    scan_teeth = root.find("Teeth").findall("Tooth")

    result = []
    centroids = []
    name = ""
    for tooth in scan_teeth:

        reconstruction_type = tooth.find("ReconstructionType").text

        # Check the ReconstructionType condition
        if reconstruction_type in ['SecondaryTelescope', 'OffsetCoping', 'Coping', 'AnatomicWaxup']:
            center_node = tooth.find("Center")
            number = tooth.find("Number").text
            name = name + '_' + number
            if center_node:
                center = np.array([float(center_node.find("x").text),
                    float(center_node.find("y").text),
                    float(center_node.find("z").text)])
                centroids.append(center)
    
    total_centroids = np.zeros(3)
    for centroid in centroids:
        total_centroids += centroid
    
    average_centroid = total_centroids/len(centroids)

    return average_centroid, name

def transform_mesh_vertices(stl_data:tm.Trimesh, rotate_x_180, centroid):
    """Transforms coordinates of vertices for a stl mesh

    Args:
        secondary_stl (tm.Trimesh)
        teeth_data (List[ToothData])
        rotate_x_180 (bool)
    Return:
        {'vertices': np.array(), 'faces': np.array()}
        List[ToothData]
    """

    if rotate_x_180:
        R_180_X = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
    else:
        R_180_X = np.eye(4)
    
    rotation_axis = [0,1,0] # arbitrary axis since average_mean_vector == z_axis
    angle = 0
    R_align = create_rotation_matrix(rotation_axis, angle, centroid)

    R = R_align * R_180_X

    transformed_stl_vertices = transform_vertices(stl_data.vertices, R)
    new_centroid = transform_vertices(centroid, R)

    # Shift in XY 
    xy_offset = -new_centroid[0:2]

    transformed_stl_vertices[0:,0:2] = transformed_stl_vertices[0:,0:2] + xy_offset

    # Shift in Z
    z_offset = TARGET_Z_HEIGHT - min(transformed_stl_vertices[:,2])

    transformed_stl_vertices[:,2] = transformed_stl_vertices[:,2] + z_offset

    transformed_mesh = tm.Trimesh(transformed_stl_vertices, stl_data.faces)

    return transformed_mesh

def transform_vertices(vertices, R):
    """
    Transforms the vertices with the given rotation matrix.
    """
    if vertices.shape == 0:
        print("Verfickte scheiße geh hier nicht rein")
        return
    if vertices.ndim > 1:
        vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        transformedVertices = (R @ vertices.T).T
        transformedVertices = transformedVertices[:, :3]
    else:
        # If vertices is only 1 vector
        vertices = np.append(vertices, 1)
        transformedVertices = (R @ vertices.T).T
        transformedVertices = transformedVertices[:3]
    return transformedVertices


def create_rotation_matrix(axis, angle, point):
    """
    Creates a rotation matrix around an axis through a point.
    """
    u, v, w = axis
    a, b, c = point

    cosA = np.cos(np.radians(angle))
    sinA = np.sin(np.radians(angle))

    R = np.array([
        [cosA + u**2*(1 - cosA), u*v*(1 - cosA) - w*sinA, u*w*(1 - cosA) + v*sinA, (a*(v**2 + w**2) - u*(b*v + c*w))*(1 - cosA) + (b*w - c*v)*sinA],
        [v*u*(1 - cosA) + w*sinA, cosA + v**2*(1 - cosA), v*w*(1 - cosA) - u*sinA, (b*(u**2 + w**2) - v*(a*u + c*w))*(1 - cosA) + (c*u - a*w)*sinA],
        [w*u*(1 - cosA) - v*sinA, w*v*(1 - cosA) + u*sinA, cosA + w**2*(1 - cosA), (c*(u**2 + v**2) - w*(a*u + b*v))*(1 - cosA) + (a*v - b*u)*sinA],
        [0, 0, 0, 1]
    ])

    return R



def reduce_stl_resolution(mesh: tm.Trimesh, target_edge_length: float):
    """
    Reduce resolution of an STL mesh by remeshing to a target average edge length.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh loaded with trimesh.
    target_edge_length : float
        Desired average edge length in the simplified mesh.

    Returns
    -------
    trimesh.Trimesh
        Simplified mesh with reduced resolution.
    """
    # Convert Trimesh -> Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces)
    )

    # Ensure normals for stability
    o3d_mesh.compute_vertex_normals()

    # Perform simplification
    simplified_mesh = o3d_mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(mesh.edges_unique_length.mean() / target_edge_length * len(mesh.faces))
    )

    # Recompute normals after decimation
    simplified_mesh.compute_vertex_normals()

    # Convert Open3D mesh -> Trimesh
    simplified_trimesh = tm.Trimesh(
        vertices=np.asarray(simplified_mesh.vertices),
        faces=np.asarray(simplified_mesh.triangles),
        process=True
    )

    return simplified_trimesh
