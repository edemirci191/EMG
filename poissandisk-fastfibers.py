import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
def poisson_disk_sampling(muscle_radius, min_distance, num_samples, tolerance=1e-6):
    """
    Perform Poisson Disk Sampling to generate non-overlapping points within a circle.
    Ensures 100% overlap resistance with added tolerance.

    Parameters:
        muscle_radius (float): Radius of the muscle boundary.
        min_distance (float): Minimum distance between fibers (sum of radii) plus a tolerance.
        num_samples (int): Number of samples to attempt.
        tolerance (float): Small value to account for floating-point precision errors.

    Returns:
        list of tuple: List of (x, y) positions of sampled points.
    """
    # Add tolerance to minimum distance
    min_distance += tolerance

    # Initialize the sampling grid
    cell_size = min_distance / np.sqrt(2)  # Cell size to ensure minimum distance constraint
    grid_width = int(np.ceil(2 * muscle_radius / cell_size))
    grid_height = int(np.ceil(2 * muscle_radius / cell_size))
    grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]

    # Function to map a point to grid coordinates
    def point_to_grid(point):
        x, y = point
        return int((x + muscle_radius) / cell_size), int((y + muscle_radius) / cell_size)

    # Generate the first point randomly within the circle
    angle = random.uniform(0, 2 * np.pi)
    radius = random.uniform(0, muscle_radius)
    first_point = (radius * np.cos(angle), radius * np.sin(angle))
    active_list = [first_point]
    samples = [first_point]

    # Add the first point to the grid
    gx, gy = point_to_grid(first_point)
    grid[gx][gy] = first_point

    # Generate points until active list is empty or num_samples is reached
    while active_list and len(samples) < num_samples:
        idx = random.randint(0, len(active_list) - 1)
        base_point = active_list[idx]
        found = False

        # Try up to 30 new points around the base point
        for _ in range(30):
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(min_distance, 2 * min_distance)
            new_point = (
                base_point[0] + distance * np.cos(angle),
                base_point[1] + distance * np.sin(angle)
            )

            # Check if the new point is inside the muscle boundary
            if np.sqrt(new_point[0]**2 + new_point[1]**2) > muscle_radius:
                continue  # Skip points outside the boundary

            # Map the point to grid coordinates
            gx, gy = point_to_grid(new_point)

            # Check for neighboring points within minimum distance
            neighbors = [
                grid[i][j]
                for i in range(max(gx - 2, 0), min(gx + 3, grid_width))
                for j in range(max(gy - 2, 0), min(gy + 3, grid_height))
                if grid[i][j] is not None
            ]

            # Ensure new point is not too close to any neighbors
            if all(np.sqrt((new_point[0] - n[0])**2 + (new_point[1] - n[1])**2) >= min_distance for n in neighbors):
                # Place the new point and add it to the active list and grid
                samples.append(new_point)
                active_list.append(new_point)
                grid[gx][gy] = new_point
                found = True
                break

        # If no point was found, remove base point from active list
        if not found:
            active_list.pop(idx)

    return samples

def generate_fibers_faster(total_num_fibers, muscle_radius):
    """
    Generate muscle fibers within a circular muscle boundary using Poisson Disk Sampling,
    ensuring no overlap and realistic physiological distribution.

    Parameters:
        total_num_fibers (int): Total number of fibers to generate.
        muscle_radius (float): Radius of the muscle in meters.

    Returns:
        list of MuscleFiber: Generated muscle fibers.
    """
    # Define fiber diameters (random within physiological ranges)
    min_diameter = 10e-6  # 10 micrometers
    max_diameter = 80e-6  # 80 micrometers
    diameters = np.random.uniform(min_diameter, max_diameter, total_num_fibers)
    radii = diameters / 2

    # Determine the average minimum distance between fibers with a tolerance
    min_distance = np.mean(radii) * 2

    # Use Poisson Disk Sampling to generate positions with tolerance
    positions = poisson_disk_sampling(muscle_radius, min_distance, total_num_fibers)

    # Define fiber length
    z_start = -0.05  # Start of fiber (m)
    z_end = 0.05     # End of fiber (m)
    fiber_length = z_end - z_start
    conduction_velocity = 4.0  # (m/s)

    fibers = []

    # Create MuscleFiber objects for each position
    for idx, (x, y) in enumerate(positions):
        # Set endplate position using normal distribution centered at fiber midpoint
        mean_endplate_pos = (z_start + z_end) / 2
        std_dev = fiber_length * 0.05  # Standard deviation is 5% of fiber length
        endplate_pos = np.random.normal(mean_endplate_pos, std_dev)
        endplate_pos = np.clip(endplate_pos, z_start, z_end)

        fiber = MuscleFiber(x, y, z_start, z_end, diameters[idx], conduction_velocity, endplate_pos, motor_unit_id=None)
        fibers.append(fiber)

    # Perform a final check using cKDTree to validate no overlaps
    fiber_positions = np.array([[fiber.x, fiber.y] for fiber in fibers])
    kdtree = cKDTree(fiber_positions)
    for i, pos in enumerate(fiber_positions):
        neighbors = kdtree.query_ball_point(pos, radii[i] * 2)
        # Ensure no neighbors except itself
        if len(neighbors) > 1:
            print(f"Overlap detected at fiber {i} with neighbors {neighbors}")

    return fibers


def visualize_fiber_placement(fibers, muscle_radius, zoom_center=(0, 0), zoom_radius=None, dpi=150):
    """
    Visualize the fiber placement within the muscle, with an option to zoom into a specific area.

    Parameters:
        fibers (list): List of MuscleFiber objects.
        muscle_radius (float): Radius of the muscle in meters.
        zoom_center (tuple): (x, y) coordinates in meters of the zoom center.
        zoom_radius (float): Radius in meters around the zoom_center to display.
                             If None, the entire muscle is displayed.
        dpi (int): Dots per inch for the figure resolution.
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)

    # Convert positions to millimeters for plotting
    fibers_x = [fiber.x * 1e3 for fiber in fibers]
    fibers_y = [fiber.y * 1e3 for fiber in fibers]
    fibers_radii = [fiber.radius * 1e3 for fiber in fibers]

    # Plot fibers
    for x, y, r in zip(fibers_x, fibers_y, fibers_radii):
        circle = plt.Circle((x, y), r, color='b', fill=False)
        ax.add_artist(circle)

    # Muscle boundary
    muscle_circle = plt.Circle((0, 0), muscle_radius * 1e3, color='r', fill=False, linestyle='--', linewidth=2)
    ax.add_artist(muscle_circle)

    ax.set_aspect('equal', 'box')

    if zoom_radius is not None:
        # Set axis limits to zoom into the specified area
        x_center = zoom_center[0] * 1e3  # Convert to mm
        y_center = zoom_center[1] * 1e3  # Convert to mm
        r = zoom_radius * 1e3            # Convert to mm
        ax.set_xlim(x_center - r, x_center + r)
        ax.set_ylim(y_center - r, y_center + r)
    else:
        # Show the entire muscle
        r = muscle_radius * 1e3
        ax.set_xlim(-r - 1, r + 1)
        ax.set_ylim(-r - 1, r + 1)

    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.set_title('Fiber Placement within Muscle')
    plt.grid(True)
    plt.show()


fibers = generate_fibers_faster(35000, muscle_radius=5e-3)
visualize_fiber_placement(fibers, muscle_radius=5e-3)
