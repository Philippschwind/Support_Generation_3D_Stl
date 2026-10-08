import numpy as np
import trimesh
from scipy.spatial import cKDTree

def adaptive_grid_support_points(mesh:trimesh.Trimesh, min_spacing, max_spacing):
    """
    Place support points with density variation based on overhang angle.
    """

    face_normals = mesh.face_normals
    face_centers = mesh.triangles_center
    
    z_axis = np.array([0, 0, -1])  # printing upwards, so down is "unsupported"

    # angle between normal and z-axis
    cos_angle = np.dot(face_normals, z_axis) / np.linalg.norm(face_normals, axis=1)
    angles = np.degrees(np.arccos(cos_angle))

    # overhang mask
    overhang_faces = np.where(angles < (180 - 60))[0]
    
    face_centers = face_centers#[overhang_faces]

    # Get the bounding box of the face centers
    min_coords = np.min(face_centers, axis=0)
    max_coords = np.max(face_centers, axis=0)

    # Assign each face center to a grid cell based on its angle
    cell_indices = []
    for i, center in enumerate(face_centers):
        spacing = angle_to_spacing(angles[i], min_spacing=min_spacing, max_spacing=max_spacing)
        cell_idx = np.floor((center[:2] - min_coords[:2]) / spacing).astype(int)
        cell_indices.append(cell_idx)
    cell_indices = np.array(cell_indices)

    # Find unique cells and lowest Z
    unique_cells, inverse = np.unique(cell_indices, axis=0, return_inverse=True)
    support_points = []
    support_normals = []
    cell_middle = []
    for cell in unique_cells:
        in_cell = (cell_indices == cell).all(axis=1)
        cell_centers = face_centers[in_cell]
        face_indices = np.where(in_cell)[0]
        if len(cell_centers) > 0:
            # Find the index of the element with the lowest z-value
            index_of_min_z = np.argmin(cell_centers[:, 2])
            avg_xy = np.mean(cell_centers[:, :2], axis=0)
            # Get the element with the lowest z-value
            element_with_min_z = cell_centers[index_of_min_z]
            support_points.append(element_with_min_z)
            support_normals.append(face_normals[face_indices[index_of_min_z]])
            cell_middle.append(np.append(avg_xy, 0))

    return np.array(support_points), np.array(support_normals), np.array(cell_middle)

def angle_to_spacing(angle, min_spacing=1.2, max_spacing=2.5, min_angle=30, max_angle=60):
    """
    Map overhang angle to grid spacing.
    """
    angle = np.degrees(angle)  # Convert to degrees
    if angle > max_angle:
        return min_spacing
    elif angle < min_angle:
        return max_spacing
    else:
        # Linear interpolation
        return min_spacing + (max_spacing - min_spacing) * ((max_angle - angle) / (max_angle - min_angle))

def resolve_close_supports(support_points, support_normals, cell_middle, mesh, radius=1.0):
    """
    Resolve overlapping support points by keeping the lowest Z point within radius
    and moving others to the cell middle projected onto the mesh.

    Args:
        support_points (list of np.ndarray): Support points (x,y,z).
        support_normals (list of np.ndarray): Normals for each support point.
        cell_middle (list of np.ndarray): Cell middle positions (x,y,z).
        mesh (trimesh.Trimesh): Mesh used for ray intersection.
        radius (float): Distance threshold in XY for clustering.

    Returns:
        new_points, new_normals
    """
    support_points = np.array(support_points)
    support_normals = np.array(support_normals)
    cell_middle = np.array(cell_middle)

    # Build KDTree in XY only
    tree = cKDTree(support_points[:, :2])
    n_points = len(support_points)
    visited = np.zeros(n_points, dtype=bool)

    new_points = []
    new_normals = []
    new_centers = []

    for i in range(n_points):
        if visited[i]:
            continue

        # Find all neighbors within radius in XY
        neighbors = tree.query_ball_point(support_points[i, :2], r=2*radius)
        neighbors = [n for n in neighbors if not visited[n]]

        if len(neighbors) == 1:
            # No overlap, keep original
            new_points.append(support_points[i])
            new_normals.append(support_normals[i])
            new_centers.append(cell_middle[i])
            visited[i] = True
        else:
            # Find the lowest Z among neighbors
            z_values = support_points[neighbors, 2]
            idx_lowest = neighbors[np.argmin(z_values)]

            # Keep lowest at its current position
            new_points.append(support_points[idx_lowest])
            new_normals.append(support_normals[idx_lowest])
            new_centers.append(cell_middle[idx_lowest])

            # Move others to their projected cell middle
            for n in neighbors:
                if n == idx_lowest:
                    continue
                xy = cell_middle[n][:2]
                projected_z = get_lowest_intersection(mesh, xy)
                new_points.append([xy[0], xy[1], projected_z])
                new_normals.append(support_normals[n])  # Keep normal
                new_centers.append(cell_middle[n])
            for n in neighbors:
                visited[n] = True

    return np.array(new_points), np.array(new_normals), np.array(new_centers)


def get_lowest_intersection(mesh, xy, zmax=1e6):
    """
    Casts a vertical ray downwards at (x,y) and returns lowest intersection Z.
    Requires trimesh (ray intersection).

    Args:
        mesh (trimesh.Trimesh): Mesh to intersect.
        xy (tuple): (x, y) location to project.
        zmax (float): Starting Z height.

    Returns:
        float: Intersection Z coordinate.
    """
    ray_origins = np.array([[xy[0], xy[1], zmax]])
    ray_directions = np.array([[0, 0, -1]])

    hits = mesh.ray.intersects_location(ray_origins, ray_directions, multiple_hits=True)[0]
    if len(hits) == 0:
        return 0.0  # fallback, nothing hit
    return hits[:, 2].min()  # lowest intersection

def generate_ray_bundle(origin, direction, radius, samples=8):
    """
    Create rays in a circular bundle around the main ray to approximate thickness.
    """

    rays = [direction]
    theta = np.linspace(0, 2*np.pi, samples, endpoint=False)
    for t in theta:
        offset = radius * np.array([np.cos(t), np.sin(t), 0])
        rays.append(direction)
        yield origin + offset, direction


def remove_support_on_part(mesh, support_points, support_normals, cell_centers, support_diameter):
    radius = support_diameter / 2.0

    valid_supports = []
    valid_normals = []
    valid_centers = []
    for i, sp in enumerate(support_points):
        has_collision = False
        for origin, direction in generate_ray_bundle(sp, np.array([0,0,-1]), radius):
            origin[2] = origin[2]-0.4
            locs, _, _ = mesh.ray.intersects_location([origin], [direction])
            if len(locs) > 0:
                has_collision = True
                break
        if not has_collision:
            valid_supports.append(sp)
            valid_normals.append(support_normals[i])
            valid_centers.append(cell_centers[i])

    return np.array(valid_supports), np.array(valid_normals), np.array(valid_centers)
        
def remove_wrong_direction(support_points, support_normals, cell_middle):
    mask = support_normals[:, 2] <= 0
    return support_points[mask], support_normals[mask], cell_middle[mask]

def remove_zero_height(support_points, support_normals, cell_middle):
    mask = support_points[:, 2] >= 1
    return support_points[mask], support_normals[mask], cell_middle[mask]

def remove_mismatched_supports(support_points, support_normals, cell_middle, max_distance):
    """
    Remove support points where the distance to the cell middle exceeds max_distance.

    Args:
        support_points (np.ndarray): Support points (x,y,z).
        support_normals (np.ndarray): Normals for each support point.
        cell_middle (np.ndarray): Cell middle positions (x,y,z).
        max_distance (float): Maximum allowed distance in XY.

    Returns:
        np.ndarray, np.ndarray, np.ndarray: Filtered support points, normals, and cell middles.
    """
    distances = np.linalg.norm(support_points[:, :2] - cell_middle[:, :2], axis=1)
    mask = distances <= max_distance
    return support_points[mask], support_normals[mask], cell_middle[mask]

def repair_mesh(mesh:trimesh.Trimesh):
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_inversion(mesh)
    
    return mesh

def compute_support_points(
    support_start_points,
    support_height_base,
    support_height_tip,
    support_penetration_depth,
    cell_middle,
    support_normals
):

    # support_points_base: same XY, Z=0
    support_points_base = cell_middle.copy()
    support_points_base[:, 2] = 0

    # support_points_middle_start: Z = support_height_base
    support_points_middle_start = cell_middle.copy()
    support_points_middle_start[:, 2] = support_height_base

    support_points_penetration = support_start_points.copy()
    support_points_penetration = support_start_points + support_penetration_depth*(support_normals*-1)

    # support_points_middle_end: Z - support_height_tip
    support_points_middle_end = cell_middle.copy()
    support_points_middle_end[:, 2] = support_start_points[:, 2] - support_height_tip

    # check angles between support middle and part
    support_points_middle_end = check_support_angle(support_start_points, support_points_middle_end, 30)


    # support_points_part: same as combined_points
    support_points_part = support_start_points

    return (
        support_points_base,
        support_points_middle_start,
        support_points_penetration,
        support_points_middle_end,
        support_points_part
    )

def check_support_angle(support_points_part, support_points_middle_end, alpha_deg):
    """ Vectorized function that adjusts z-coordinate(s) of multiple points A
    so each vector AB makes at least the desired angle alpha (in degrees)
    with the z-axis. If the current angle is already larger than alpha,
    the point A is left unchanged.

    Args:
        support_points_part (np.array): 
        support_points_middle_end (np.array): 
        alpha_deg (float): max angle to z-axis (degrees)

    Returns:
        (N, 3) ndarray: Adjusted A points with updated z values.
    """

    A = np.asarray(support_points_middle_end, dtype=float)
    B = np.asarray(support_points_part, dtype=float)
    alpha = np.deg2rad(alpha_deg)

    # Compute components
    d = B - A  # AB vectors
    r = np.linalg.norm(d[:, :2], axis=1)  # horizontal distance
    dz = d[:, 2]  # vertical difference
    length = np.linalg.norm(d, axis=1)

    # Current angle between AB and z-axis
    # cos(theta) = |dz| / |AB|
    cos_theta = np.abs(dz) / length
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # Identify which need adjustment (theta < alpha)
    needs_adjustment = theta > alpha

    # Avoid division by zero (alpha = 0° or 180°)
    if np.isclose(np.sin(alpha), 0):
        raise ValueError("Angle alpha cannot be 0° or 180° (undefined cotangent).")

    # Compute new zA only for those that need adjustment
    zA_new = np.where(
        needs_adjustment,
        B[:, 2] - r * np.cos(alpha) / np.sin(alpha),  # adjusted
        A[:, 2]  # unchanged
    )

    # Construct output
    A_new = A.copy()
    A_new[:, 2] = zA_new

    return A_new

def generate_support_upright(support_points_base, support_diameter_base, support_points_middle_start,
                             support_diameter_middle, support_points_middle_end, support_points_penetration,
                             support_normals_part, support_diameter_part, angle_steps_support, support_cutoff_height):
    """_summary_

    Args:
        support_points_base (_type_): _description_
        support_diameter_base (_type_): _description_
        support_points_middle_start (_type_): _description_
        support_diameter_middle (_type_): _description_
        support_points_middle_end (_type_): _description_
        support_points_penetration (_type_): _description_
        support_normals_part (_type_): _description_
        support_diameter_part (_type_): _description_
        angle_steps_support (_type_): _description_
    """
    circle_points_base = calculate_circle_points(support_points_base, support_diameter_base,
                                                 angle_steps_support)
    circle_points_middle_start = calculate_circle_points(support_points_middle_start, support_diameter_middle,
                                                 angle_steps_support)
    circle_points_middle_end = calculate_circle_points(support_points_middle_end, support_diameter_middle,
                                                 angle_steps_support)
    circle_points_penetration = calculate_tilted_circle_points(support_points_penetration, 
                                                               support_normals_part, 
                                                               support_diameter_part,
                                                               angle_steps_support, 
                                                               support_cutoff_height)
    vertices_down, faces_down = compute_down_faces(circle_points_base, support_points_base)
    vertices_up, faces_up = compute_up_faces(circle_points_penetration, support_points_penetration)

    vertices_side1, faces_side1 = compute_side_faces(circle_points_base, circle_points_middle_start)
    vertices_side2, faces_side2 = compute_side_faces(circle_points_middle_start, circle_points_middle_end)
    vertices_side3, faces_side3 = compute_side_faces(circle_points_middle_end, circle_points_penetration)

    support_vertices, support_faces = combine_and_clean_vertices(
        vertices_down, faces_down, vertices_up, faces_up, vertices_side1, faces_side1, 
        vertices_side2, faces_side2, vertices_side3, faces_side3)
    
    return support_vertices, support_faces


def calculate_circle_points(support_points_base, support_diameter_base, angle_steps_support):
    """
    For each point in support_points_base, generate points around a circle 
    of radius support_diameter_base / 2 with angle steps of angle_steps_support (in degrees).
    """
    # Number of points based on angle step
    num_points = round(360 / angle_steps_support)
    
    # Compute angles in radians
    angles = np.linspace(0, 2 * np.pi, num_points + 1)
    angles = angles[:-1]  # remove last point (duplicate of first)
    
    # Radius
    radius = support_diameter_base / 2.0

    # Compute circle offsets
    x_offsets = radius * np.cos(angles)
    y_offsets = radius * np.sin(angles)

    # Number of base points
    num_base_points = support_points_base.shape[0]

    # Initialize list to hold arrays (equivalent to cell array)
    circle_points_list = []

    for i in range(num_base_points):
        x_base, y_base, z_base = support_points_base[i, :]
        
        x_circle = x_base + x_offsets
        y_circle = y_base + y_offsets
        z_circle = np.full_like(x_circle, z_base)

        # Combine into (num_points, 3) array
        circle_coords = np.column_stack((x_circle, y_circle, z_circle))
        circle_points_list.append(circle_coords)

    return circle_points_list


def calculate_tilted_circle_points(
    support_points_penetration,
    support_normals_part,
    support_diameter_part,
    angle_steps_support,
    support_cutoff_height
):
    """
    For each point in support_points_penetration, generate a tilted circle of radius
    support_diameter_part / 2 around the normal vector support_normals_part.
    """
    # Number of points around the circle
    num_points = round(360 / angle_steps_support)
    
    # Angles in radians
    angles = np.linspace(0, 2 * np.pi, num_points + 1)
    angles = angles[:-1]  # remove duplicate last point
    
    # Radius
    radius = support_diameter_part / 2.0

    # Offsets in XY plane
    x_offsets = radius * np.cos(angles)
    y_offsets = radius * np.sin(angles)

    num_penetration_points = support_points_penetration.shape[0]

    circle_points_list = []
    z_axis = np.array([0,0,1])

    # Max angle to z-axis based on support_cutoff_height and radius
    # Angle between orthogonal vector of normal and xy-plane
    # This angle is the same as the angle between normal and z-axis
    max_angle_to_z = np.arctan2(support_cutoff_height, radius)
    min_angle = np.deg2rad(180) - max_angle_to_z        # 180 degrees needed because normal points away from part
    max_angle = np.deg2rad(180) + min_angle
    for i in range(num_penetration_points):
        x_base, y_base, z_base = support_points_penetration[i, :]
        normal = support_normals_part[i, :]
        
        # Check if normal angle to z-axis is greater than max angle
        # To prevent to higly tilted points
        dot = np.dot(normal, z_axis)
        dot = np.clip(dot, -1, 1)
        angle = np.arccos(dot)

        if angle > max_angle:
            perp = normal - dot * z_axis
            perp_norm = np.linalg.norm(perp)

            if not perp_norm < 1e-12:
                perp = perp / perp_norm

                # Build a new vector rotated to exactly max_angle from z-axis
                new_n = np.cos(max_angle) * z_axis + np.sin(max_angle) * perp

                # Scale to original vector length
                normal = new_n * np.linalg.norm(normal)
            else:
                # Vector is basically on z-axis but pointing the wrong direction
                return np.array([np.cos(max_angle), np.sin(max_angle), 0.0])
        if angle < min_angle:
            perp = normal - dot * z_axis
            perp_norm = np.linalg.norm(perp)

            if not perp_norm < 1e-12:
                perp = perp / perp_norm

                # Build a new vector rotated to exactly min_angle from z-axis
                new_n = np.cos(min_angle) * z_axis + np.sin(min_angle) * perp

                # Scale to original vector length
                normal = new_n * np.linalg.norm(normal)
            else:
                # Vector is basically on z-axis but pointing the wrong direction
                return np.array([np.cos(min_angle), np.sin(min_angle), 0.0])

        # Compute z_offsets so that points lie in the plane orthogonal to normal
        # This comes from the plane equation: n_x*(x - x0) + n_y*(y - y0) + n_z*(z - z0) = 0
        # Solve for z: z = z0 - (n_x*(x - x0) + n_y*(y - y0)) / n_z
        # Since x - x0 = x_offsets and y - y0 = y_offsets:
        z_offsets = (x_offsets * normal[0] + y_offsets * normal[1]) / normal[2]
        
        x_circle = x_base + x_offsets
        y_circle = y_base + y_offsets
        z_circle = z_base - z_offsets

        circle_coords = np.column_stack((x_circle, y_circle, z_circle))
        circle_points_list.append(circle_coords)

    return circle_points_list


def compute_down_faces(circle_points_base, support_points_base):
    """
    Build vertices and faces for downward facing triangles connecting support_points_base
    with their corresponding circle points.

    Parameters:
        circle_points_base: list of (num_circle_points, 3) arrays
        support_points_base: (num_base_points, 3) array

    Returns:
        vertices: (total_vertices, 3) array
        faces: (num_base_points * num_circle_points, 3) array of vertex indices (0-based)
    """
    num_base_points = support_points_base.shape[0]
    num_circle_points = circle_points_base[0].shape[0]

    # total number of vertices
    total_vertices = num_base_points * (1 + num_circle_points)
    vertices = np.zeros((total_vertices, 3))

    # Indices for base points in the vertices array
    base_indices = np.arange(0, total_vertices, 1 + num_circle_points)

    # circle_indices_offset: [1, 2, ..., num_circle_points]
    circle_indices_offset = np.arange(1, num_circle_points + 1)

    # Place support_points_base in vertices
    vertices[base_indices, :] = support_points_base

    # Place circle points in vertices
    all_circle_points = np.vstack(circle_points_base)
    for i, base_idx in enumerate(base_indices):
        start = base_idx + 1
        end = start + num_circle_points
        vertices[start:end, :] = circle_points_base[i]

    # Create faces
    # face1: base_indices repeated num_circle_points times
    face1 = np.repeat(base_indices, num_circle_points)

    # face2: base_indices + current circle_indices_offset
    face2 = np.hstack([
        np.full(num_circle_points, base_idx) + circle_indices_offset
        for base_idx in base_indices
    ])

    # face3: base_indices + next circle_indices_offset (wrap around)
    next_circle_indices_offset = np.roll(circle_indices_offset, -1)
    face3 = np.hstack([
        np.full(num_circle_points, base_idx) + next_circle_indices_offset
        for base_idx in base_indices
    ])

    # Compute normals
    v1 = vertices[face2, :] - vertices[face1, :]
    v2 = vertices[face3, :] - vertices[face1, :]
    normals = np.cross(v1, v2)

    # Check normals; if normals[:, 2] > 0, flip face2 and face3
    flip_faces = normals[:, 2] > 0
    face2_flipped = face2.copy()
    face3_flipped = face3.copy()
    face2_flipped[flip_faces] = face3[flip_faces]
    face3_flipped[flip_faces] = face2[flip_faces]

    # Stack faces (convert to shape (N, 3))
    faces = np.column_stack((face1, face2_flipped, face3_flipped))

    return vertices, faces

def compute_up_faces(circle_points_penetration, support_points_penetration):
    """
    Build vertices and faces for downward facing triangles connecting support_points_base
    with their corresponding circle points.

    Parameters:
        circle_points_base: list of (num_circle_points, 3) arrays
        support_points_base: (num_penetration_points, 3) array

    Returns:
        vertices: (total_vertices, 3) array
        faces: (num_penetration_points * num_circle_points, 3) array of vertex indices (0-based)
    """
    num_penetration_points = support_points_penetration.shape[0]
    num_circle_points = circle_points_penetration[0].shape[0]

    # total number of vertices
    total_vertices = num_penetration_points * (1 + num_circle_points)
    vertices = np.zeros((total_vertices, 3))

    # Indices for base points in the vertices array
    penetration_indices = np.arange(0, total_vertices, 1 + num_circle_points)

    # circle_indices_offset: [1, 2, ..., num_circle_points]
    circle_indices_offset = np.arange(1, num_circle_points + 1)

    # Place support_points_base in vertices
    vertices[penetration_indices, :] = support_points_penetration

    # Place circle points in vertices
    all_circle_points = np.vstack(circle_points_penetration)
    for i, base_idx in enumerate(penetration_indices):
        start = base_idx + 1
        end = start + num_circle_points
        vertices[start:end, :] = circle_points_penetration[i]

    # Create faces
    # face1: penetration_indices repeated num_circle_points times
    face1 = np.repeat(penetration_indices, num_circle_points)

    # face2: penetration_indices + current circle_indices_offset
    face2 = np.hstack([
        np.full(num_circle_points, base_idx) + circle_indices_offset
        for base_idx in penetration_indices
    ])

    # face3: base_indices + next circle_indices_offset (wrap around)
    next_circle_indices_offset = np.roll(circle_indices_offset, -1)
    face3 = np.hstack([
        np.full(num_circle_points, penetration_idx) + next_circle_indices_offset
        for penetration_idx in penetration_indices
    ])

    # Compute normals
    v1 = vertices[face2, :] - vertices[face1, :]
    v2 = vertices[face3, :] - vertices[face1, :]
    normals = np.cross(v1, v2)

    # Check normals; if normals[:, 2] < 0, flip face2 and face3
    flip_faces = normals[:, 2] < 0
    face2_flipped = face2.copy()
    face3_flipped = face3.copy()
    face2_flipped[flip_faces] = face3[flip_faces]
    face3_flipped[flip_faces] = face2[flip_faces]

    # Stack faces (convert to shape (N, 3))
    faces = np.column_stack((face1, face2_flipped, face3_flipped))

    return vertices, faces


def compute_side_faces(circle_points_base, circle_points_middle_start):
    """
    For each dataset (base & middle_start circle), creates side faces (two triangles per quad).
    Uses counter-clockwise winding for outward-facing normals when viewed from outside.
    
    Parameters:
        circle_points_base: list of (num_points, 3) arrays
        circle_points_middle_start: list of (num_points, 3) arrays
    
    Returns:
        vertices: (total_vertices, 3) array
        faces: (num_faces, 3) array of indices (0-based)
    """
    num_datasets = len(circle_points_base)
    num_points_per_dataset = circle_points_base[0].shape[0]

    vertices_list = []
    faces_list = []

    for i in range(num_datasets):
        base_points = circle_points_base[i]
        middle_points = circle_points_middle_start[i]

        # Add to vertices
        offset = len(vertices_list)
        vertices_list.extend(base_points)
        vertices_list.extend(middle_points)

        # Indices in the vertices array
        base_indices = np.arange(offset, offset + num_points_per_dataset)
        middle_indices = base_indices + num_points_per_dataset

        # Faces: two triangles per quad
        # For a quad with vertices [base, next_base, next_middle, middle] in CCW order (from outside):
        # Triangle 1: [base, next_base, next_middle]
        # Triangle 2: [base, next_middle, middle]
        next_base_indices = np.roll(base_indices, -1)
        next_middle_indices = np.roll(middle_indices, -1)

        face1 = np.column_stack((base_indices, next_base_indices, next_middle_indices))
        face2 = np.column_stack((base_indices, next_middle_indices, middle_indices))

        faces_list.append(face1)
        faces_list.append(face2)

    vertices = np.array(vertices_list)
    faces = np.vstack(faces_list)

    return vertices, faces

def combine_and_clean_vertices(vertices_down, faces_down,
                               vertices_up, faces_up,
                               vertices_side1, faces_side1,
                               vertices_side2, faces_side2,
                               vertices_side3, faces_side3):
    """
    Combine all vertices and faces, remove duplicate vertices,
    and remap faces to new unique vertex indices.
    
    Parameters:
        vertices_*, faces_*: numpy arrays of vertices and faces
    
    Returns:
        unique_vertices: (num_unique_vertices, 3) array
        adjusted_faces: (total_faces, 3) array of new indices
    """
    # Combine all vertices
    all_vertices = np.vstack([
        vertices_down,
        vertices_up,
        vertices_side1,
        vertices_side2,
        vertices_side3
    ])

    # Compute offsets for faces so they index into the combined vertices
    offset_up = len(vertices_down)
    offset_side1 = offset_up + len(vertices_up)
    offset_side2 = offset_side1 + len(vertices_side1)
    offset_side3 = offset_side2 + len(vertices_side2)

    all_faces = np.vstack([
        faces_down,
        faces_up + offset_up,
        faces_side1 + offset_side1,
        faces_side2 + offset_side2,
        faces_side3 + offset_side3
    ])

    # Find unique vertices (keeping first occurrence)
    unique_vertices, inverse_indices = np.unique(all_vertices, axis=0, return_inverse=True)
    # Remap faces: inverse_indices maps old vertex indices to new unique indices
    adjusted_faces = inverse_indices[all_faces]

    return unique_vertices, adjusted_faces


def generate_grid_support(mesh:trimesh.Trimesh, support_params, order_name, file_name):


    support_points, support_normals, cell_middle = adaptive_grid_support_points(mesh, support_params.minGridSpacing, support_params.maxGridSpacing)

    support_points, support_normals, cell_middle = resolve_close_supports(support_points, support_normals, cell_middle, mesh, support_params.minSupportDistance)        

    support_points, support_normals, cell_middle = remove_wrong_direction(support_points, support_normals, cell_middle)

    support_points, support_normals, cell_middle = remove_support_on_part(
        mesh, support_points, support_normals, cell_middle, support_params.SupportDiameterPart)

    support_points, support_normals, cell_middle = remove_zero_height(support_points, support_normals, cell_middle)

    support_points, support_normals, cell_middle = remove_mismatched_supports(support_points, support_normals, cell_middle, support_params.maxGridSpacing * 2)


    support_points_base, support_points_middle_start, support_points_penetration, \
        support_points_middle_end, support_points_part = compute_support_points(
            support_points, support_params.SupportHeightBase, support_params.SupportHeightTip, 
            support_params.SupportPenetrationDepth, cell_middle, support_normals)
    
    vertices, faces = generate_support_upright(
        support_points_base, support_params.SupportDiameterBase, support_points_middle_start,
        support_params.SupportDiameterMiddle, support_points_middle_end, support_points_penetration,
        support_normals, support_params.SupportDiameterPart, support_params.angleStepsSupport,
        support_params.supportCutoffHeight)
    support_mesh = trimesh.Trimesh(vertices, faces)

    return support_mesh
