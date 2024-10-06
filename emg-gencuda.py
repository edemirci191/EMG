# Install CuPy (ensure the version matches the CUDA version)
# Uncomment and adjust the following line if you haven't installed CuPy yet
# !pip install cupy-cuda12x

import cupy as cp
import numpy as np  # Still used for some functions
import matplotlib.pyplot as plt
from scipy.signal import bessel, lfilter
import random
import matplotlib.patches as patches
from scipy.spatial import cKDTree
import cupyx.scipy.fft as fft  # CuPy's FFT package
import time

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Constants
SIGMA_R = 0.05  # Radial conductivity (S/m)
SIGMA_Z = 0.3   # Axial conductivity (S/m)
K = SIGMA_Z / SIGMA_R  # Anisotropy ratio
FS = 20000    # Sampling frequency (Hz)
DT = 1 / FS   # Sampling interval (s)
DURATION = 0.5  # Simulation duration (s)
T = cp.arange(0, DURATION, DT)  # Time vector using CuPy
CUTOFF_FREQ = 5000  # Anti-aliasing filter cutoff frequency (Hz)
ORDER = 2     # Order of the Bessel filter

# Electrode position (e.g., at a quarter length along z-axis)
ELECTRODE_POS = cp.array([0.00125, 0.00125, 0.025])  # Electrode position at quarter length
Z_0 = 0  # Reference point along z-axis

# Define the Intracellular Action Potential (IAP)
def iap(t, V0, duration):
    """Intracellular action potential modeled as a sine-squared pulse."""
    iap_signal = cp.zeros_like(t)
    idx = (t >= 0) & (t <= duration)
    iap_signal[idx] = V0 * cp.sin(cp.pi * t[idx] / duration) ** 2
    return iap_signal

# Muscle Fiber Class
class MuscleFiber:
    def __init__(self, x, y, z_start, z_end, diameter, conduction_velocity, endplate_pos, motor_unit_id):
        self.x = x  # Radial position x (m)
        self.y = y  # Radial position y (m)
        self.z_start = z_start  # Axial start position (m)
        self.z_end = z_end      # Axial end position (m)
        self.diameter = diameter  # Fiber diameter (m)
        self.radius = diameter / 2  # Fiber radius (m)
        self.conduction_velocity = conduction_velocity  # (m/s)
        self.endplate_pos = endplate_pos  # Endplate position along z-axis (m)
        self.motor_unit_id = motor_unit_id  # ID of the motor unit

class MotorUnit:
    def __init__(self, fibers, firing_rate, id):
        """
        Initialize a motor unit.

        Args:
            fibers (list of MuscleFiber): Fibers controlled by this motor unit.
            firing_rate (float): Firing rate in Hz.
            id (int): Motor unit identifier.
        """
        self.fibers = fibers
        self.firing_rate = firing_rate
        self.id = id
        self.ap_times = self.generate_ap_times()

    def generate_ap_times(self):
        """Generate action potential initiation times based on the firing rate."""
        inter_spike_interval = 1.0 / self.firing_rate
        # Start at a random phase between 0 and inter_spike_interval
        start_time = np.random.uniform(0, inter_spike_interval)
        # Generate spikes throughout the duration
        ap_times = cp.arange(start_time, DURATION + inter_spike_interval, inter_spike_interval)
        return ap_times

def generate_motor_units(num_motor_units, total_num_fibers):
    muscle_radius = 5e-3  # 5 mm in meters

    # Generate fibers without overlaps
    fibers = generate_fibers_faster(total_num_fibers, muscle_radius)

    # Randomly assign fibers to motor units
    motor_unit_ids = np.random.choice(num_motor_units, len(fibers))
    for idx, fiber in enumerate(fibers):
        fiber.motor_unit_id = motor_unit_ids[idx]

    # Group fibers by motor unit
    motor_units = []
    for mu_id in range(num_motor_units):
        mu_fibers = [fiber for fiber in fibers if fiber.motor_unit_id == mu_id]

        # Assign a firing rate to the motor unit
        firing_rate = np.random.uniform(8, 20)  # Firing rate between 8 and 20 Hz

        motor_unit = MotorUnit(mu_fibers, firing_rate, mu_id)
        motor_units.append(motor_unit)

    return motor_units

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

    return fibers

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

# Impedance Function Z(x, y, z)
def impedance(x, y, z, sigma_r, K):
    """Calculate the impedance Z(x, y, z) between the current source and point (x, y, z)."""
    denominator = cp.sqrt(K * (x ** 2 + y ** 2) + z ** 2)
    Z = 1 / (4 * cp.pi * sigma_r * denominator)
    return Z

# Concentric Needle (CN) Electrode Weighting Function
def cn_electrode_weighting(fiber_x, fiber_y, z, sigma_r, K, R_c, N_c, z_0, electrode_pos):
    """
    Calculate the weighting function for the CN electrode.
    """
    r_x = fiber_x - electrode_pos[0]
    r_y = fiber_y - electrode_pos[1]
    z = z - z_0

    r_u = r_x
    r_v = r_y

    weighting_sum = 0
    valid_k_count = 0  # To adjust normalization since we're skipping k=0

    for k in range(-N_c, N_c + 1):
        # Skip k = 0 to avoid division by zero
        if k == 0:
            continue

        Delta_u_k = R_c * (k / (N_c + 1))
        Delta_z_k = R_c * (k / (N_c + 1))
        B_k = r_v ** 2 + (1 / K) * (z - (Delta_z_k + z_0)) ** 2

        numerator = cp.sqrt((r_u + Delta_u_k) ** 2 + B_k) + r_u + Delta_u_k
        denominator = cp.sqrt((r_u - Delta_u_k) ** 2 + B_k) + r_u - Delta_u_k

        # Avoid division by zero or negative values inside logarithm
        ln_argument = numerator / denominator
        ln_argument = cp.where(ln_argument <= 0, 1e-12, ln_argument)

        # Use absolute value to ensure the sqrt is valid
        sqrt_term = cp.sqrt(K * cp.abs(Delta_u_k))

        # Avoid division by zero in case sqrt_term is zero
        if sqrt_term == 0:
            continue

        weighting_k = (1 / (8 * cp.pi * sigma_r * sqrt_term)) * cp.log(ln_argument)
        weighting_sum += weighting_k
        valid_k_count += 1  # Increment count of valid terms

    # Adjust normalization factor
    if valid_k_count > 0:
        weighting = weighting_sum / valid_k_count
    else:
        weighting = 0  # If no valid terms, weighting is zero

    return weighting

def compute_weighting_function_vectorized(fiber_x, fiber_y, t_w, c, endplate_pos, electrode_type, electrode_params, electrode_pos):
    """
    Compute the temporal weighting function w(t) for multiple fibers using vectorized operations.

    Args:
        fiber_x (cupy array): x positions of fibers.
        fiber_y (cupy array): y positions of fibers.
        t_w (cupy array): Time vector for the weighting function.
        c (float): Conduction velocity.
        endplate_pos (cupy array): Endplate positions of fibers.
        electrode_type (str): Type of electrode ('macro', 'sf', 'cn').
        electrode_params (dict): Parameters specific to the electrode type.
        electrode_pos (cupy array): Position of the electrode [x, y, z].

    Returns:
        cupy array: Temporal weighting function w(t) for each fiber.
    """
    # Constants
    sigma_r = SIGMA_R
    K_value = K
    z_0 = Z_0

    # Only 'cn' electrode type is considered in this version
    if electrode_type == 'cn':
        # Extract electrode parameters
        R_c = electrode_params.get('R_c', 0.3e-3)  # Radius of CN electrode
        N_c = electrode_params.get('N_c', 10)      # Number of divisions

        # Compute z positions over time for all fibers
        z = c * t_w[:, cp.newaxis] + endplate_pos[cp.newaxis, :]
        z = z - z_0

        r_x = fiber_x - electrode_pos[0]
        r_y = fiber_y - electrode_pos[1]
        r_u = r_x
        r_v = r_y

        weighting_sum = cp.zeros_like(z)
        valid_k_count = 0  # To adjust normalization since we're skipping k=0

        for k in range(-N_c, N_c + 1):
            # Skip k = 0 to avoid division by zero
            if k == 0:
                continue

            Delta_u_k = R_c * (k / (N_c + 1))
            Delta_z_k = R_c * (k / (N_c + 1))
            B_k = r_v ** 2 + (1 / K_value) * (z - (Delta_z_k + z_0)) ** 2

            numerator = cp.sqrt((r_u + Delta_u_k) ** 2 + B_k) + r_u + Delta_u_k
            denominator = cp.sqrt((r_u - Delta_u_k) ** 2 + B_k) + r_u - Delta_u_k

            # Avoid division by zero or negative values inside logarithm
            ln_argument = numerator / denominator
            ln_argument = cp.where(ln_argument <= 0, 1e-12, ln_argument)

            # Use absolute value to ensure the sqrt is valid
            sqrt_term = cp.sqrt(K_value * cp.abs(Delta_u_k))

            # Avoid division by zero in case sqrt_term is zero
            if sqrt_term == 0:
                continue

            weighting_k = (1 / (8 * cp.pi * sigma_r * sqrt_term)) * cp.log(ln_argument)
            weighting_sum += weighting_k
            valid_k_count += 1  # Increment count of valid terms

        # Adjust normalization factor
        if valid_k_count > 0:
            weighting = weighting_sum / valid_k_count
        else:
            weighting = cp.zeros_like(weighting_sum)  # If no valid terms, weighting is zero

        return weighting
    else:
        raise ValueError(f"Unknown electrode type: {electrode_type}")

def generate_action_potential_shape(t_w):
    """Generate a simple action potential shape."""
    # Parameters for IAP
    V0 = 100e-3     # Peak amplitude (V)
    duration = 2e-3  # Duration of IAP (s)
    return iap(t_w, V0, duration)

def apply_bandpass_filter(signal, fs, cutoff_freq):
    """Apply a low-pass Bessel filter to the signal."""
    b, a = bessel(ORDER, 2 * cutoff_freq / fs, btype='low', analog=False)
    filtered_signal = lfilter(b, a, cp.asnumpy(signal), axis=0)
    return cp.asarray(filtered_signal)

def compute_fiber_aps_batch(fibers, t, electrode_type, electrode_params, electrode_pos, fs, cutoff_freq, ap_times_list):
    """
    Compute the action potentials for multiple fibers using convolution via FFT.

    Args:
        fibers (list of MuscleFiber): List of muscle fibers.
        t (cupy array): Time vector.
        electrode_type (str): Type of electrode ('macro', 'sf', 'cn').
        electrode_params (dict): Parameters specific to the electrode type.
        electrode_pos (cupy array): Position of the electrode [x, y, z].
        fs (float): Sampling frequency.
        cutoff_freq (float): Cutoff frequency for the bandpass filter.
        ap_times_list (list of cupy arrays): List of AP times for each fiber.

    Returns:
        cupy array: Simulated action potentials for the fibers.
    """
    # Extract fiber properties into arrays
    fiber_x = cp.array([fiber.x for fiber in fibers])
    fiber_y = cp.array([fiber.y for fiber in fibers])
    endplate_pos = cp.array([fiber.endplate_pos for fiber in fibers])
    c = fibers[0].conduction_velocity  # Assuming same conduction velocity for all fibers

    # Compute t_w based on maximum tau_plus and tau_minus across all fibers
    tau_plus = cp.max(cp.array([fiber.z_end - fiber.endplate_pos for fiber in fibers])) / c
    tau_minus = cp.max(cp.array([fiber.endplate_pos - fiber.z_start for fiber in fibers])) / c
    t_w = cp.arange(-tau_minus, tau_plus, DT)

    # Compute weighting function for all fibers
    w_t = compute_weighting_function_vectorized(fiber_x, fiber_y, t_w, c, endplate_pos, electrode_type, electrode_params, electrode_pos)
    # w_t has shape (len(t_w), number_of_fibers)

    # Generate action potential shape
    ap_shape = generate_action_potential_shape(t_w)
    # ap_shape has shape (len(t_w),)

    # Compute transmembrane current (second derivative)
    dt = 1 / fs
    im = cp.gradient(cp.gradient(ap_shape, dt, axis=0), dt, axis=0)
    # im has shape (len(t_w),)

    # Apply bandpass filter (convert to numpy, filter, then back to cupy)
    im_filtered = apply_bandpass_filter(im, fs, cutoff_freq)
    # im_filtered has shape (len(t_w),)

    # Adjust im_filtered to match dimensions with w_t
    im_filtered = im_filtered[:, cp.newaxis]  # Shape becomes (len(t_w), 1)
    im_filtered = cp.tile(im_filtered, (1, len(fibers)))  # Shape becomes (len(t_w), number_of_fibers)

    # Perform convolution using FFT along time axis
    n_conv = im_filtered.shape[0] + w_t.shape[0] - 1
    n_fft = 2 ** int(cp.ceil(cp.log2(n_conv)))

    # Prepare arrays for FFT
    im_padded = cp.zeros((n_fft, im_filtered.shape[1]), dtype=cp.float32)
    w_padded = cp.zeros((n_fft, w_t.shape[1]), dtype=cp.float32)
    im_padded[:im_filtered.shape[0], :] = im_filtered
    w_padded[:w_t.shape[0], :] = w_t

    # Perform FFT along time axis for all fibers
    IM = fft.fft(im_padded, axis=0)
    W = fft.fft(w_padded, axis=0)
    V = IM * W
    v_t_full = fft.ifft(V, axis=0).real

    # Extract the valid part of the convolution
    v_single = v_t_full[:n_conv, :]

    # Corrected t_conv definition using linspace for consistent lengths
    t_conv = cp.linspace(0, (n_conv - 1) * DT, n_conv)

    # Initialize the fibers' EMG signals
    fiber_emg = cp.zeros((len(t), len(fibers)), dtype=cp.float32)

    # Sum contributions for each AP initiation time
    for idx in range(len(fibers)):
        ap_times = ap_times_list[idx]
        for ap_time in ap_times:
            # Shift the signal according to the AP initiation time
            t_shifted = t_conv + ap_time
            # Interpolate and add to the EMG signal
            shifted_v = cp.interp(t, t_shifted, v_single[:, idx], left=0, right=0)
            fiber_emg[:, idx] += shifted_v

    # Sum over fibers
    emg_signal = cp.sum(fiber_emg, axis=1)

    return emg_signal

def simulate_emg_signal(motor_units, t, electrode_type, electrode_params, electrode_pos, fs, cutoff_freq):
    """
    Simulate the EMG signal by summing the contributions from all motor units.

    Args:
        motor_units (list): List of MotorUnit objects.
        t (cupy array): Time vector.
        electrode_type (str): Type of electrode ('macro', 'sf', 'cn').
        electrode_params (dict): Parameters specific to the electrode type.
        electrode_pos (cupy array): Position of the electrode [x, y, z].
        fs (float): Sampling frequency.
        cutoff_freq (float): Cutoff frequency for the bandpass filter.

    Returns:
        emg_signals_per_mu (list of cupy arrays): List of EMG signals, one per motor unit.
        total_emg_signal (cupy array): The sum of all EMG signals from motor units.
    """
    emg_signals_per_mu = []
    total_emg_signal = cp.zeros_like(t)

    for mu in motor_units:
        fibers = mu.fibers
        # Convert AP times to cupy arrays
        ap_times_list = [mu.ap_times for fiber in fibers]

        # Compute EMG signals for fibers in the motor unit
        emg_signal_mu = compute_fiber_aps_batch(
            fibers, t, electrode_type, electrode_params, electrode_pos, fs, cutoff_freq, ap_times_list
        )

        emg_signals_per_mu.append(emg_signal_mu)
        total_emg_signal += emg_signal_mu

    return emg_signals_per_mu, total_emg_signal

# Visualization functions
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
    ax.set_title(f'Fiber Placement within Muscle')
    plt.grid(True)
    plt.show()

def visualize_muscle_fibers(motor_units, electrode_pos=None):
    """Visualize the arrangement of muscle fibers in 3D, optionally displaying electrode position."""
    from mpl_toolkits.mplot3d import Axes3D  # Import here to avoid issues if not used elsewhere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for mu in motor_units:
        color = colors[mu.id % len(colors)]
        for fiber in mu.fibers:
            x = [fiber.x * 1000, fiber.x * 1000]
            y = [fiber.y * 1000, fiber.y * 1000]
            z = [fiber.z_start * 1000, fiber.z_end * 1000]
            ax.plot(x, y, z, color, linewidth=1)
            # Mark the endplate position
            ax.scatter(fiber.x * 1000, fiber.y * 1000, fiber.endplate_pos * 1000, c='red', marker='o')

    # Optionally plot electrode position
    if electrode_pos is not None:
        ax.scatter(electrode_pos[0].get() * 1000, electrode_pos[1].get() * 1000, electrode_pos[2].get() * 1000,
                   c='black', marker='x', s=100, label='Electrode')
        ax.legend()

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Muscle Fiber Arrangement (3D View)')
    plt.show()

def visualize_cross_section(motor_units, z_position=0.0, electrode_pos=None):
    """
    Visualize the cross-sectional view of muscle fibers at a given z-position, with an optional electrode position marked.
    """
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Define colors for different motor units
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Calculate the maximum radius to set axis limits dynamically
    all_radii = [fiber.radius * 1000 for mu in motor_units for fiber in mu.fibers]
    max_radius = max(all_radii) if all_radii else 1  # Use max radius or 1 if empty

    # Iterate through each motor unit and its fibers
    for mu in motor_units:
        color = colors[mu.id % len(colors)]
        for fiber in mu.fibers:
            # Only include fibers that pass through the specified z-position
            if fiber.z_start <= z_position <= fiber.z_end:
                # Convert x and y positions to mm
                x_pos = fiber.x * 1000  # Convert to mm
                y_pos = fiber.y * 1000  # Convert to mm

                # Radius in mm
                radius = fiber.radius * 1000  # Convert to mm

                # Draw the circle representing the fiber cross-section
                circle = patches.Circle((x_pos, y_pos), radius, edgecolor=color, facecolor='none', linewidth=1.5)
                ax.add_patch(circle)

    # Plot electrode position if provided
    if electrode_pos is not None:
        ax.scatter(electrode_pos[0].get() * 1000, electrode_pos[1].get() * 1000,
                   color='black', marker='x', s=100, label='Electrode')
        ax.legend()

    # Set axis limits based on muscle radius and maximum fiber radius
    muscle_radius_mm = 5  # Muscle radius in mm
    ax.set_xlim(-muscle_radius_mm - max_radius, muscle_radius_mm + max_radius)
    ax.set_ylim(-muscle_radius_mm - max_radius, muscle_radius_mm + max_radius)
    ax.set_aspect('equal', 'box')  # Ensure equal aspect ratio

    # Set axis labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(f'Muscle Fiber Cross-Section at Z = {z_position * 1000:.2f} mm')
    ax.grid(True)

    # Show the plot
    plt.show()

def visualize_longitudinal_view(motor_units, electrode_pos=None):
    """Visualize the longitudinal view showing fiber lengths and endplate positions, with an optional electrode position marked."""
    fig, ax = plt.subplots()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for mu in motor_units:
        color = colors[mu.id % len(colors)]
        for fiber in mu.fibers:
            # Represent each fiber as a line from z_start to z_end at a unique y-position
            y_pos = fiber.x * 1000  # Use x-position to spread fibers along y-axis
            ax.plot([fiber.z_start * 1000, fiber.z_end * 1000], [y_pos, y_pos], color, linewidth=1)
            # Mark the endplate position
            ax.plot(fiber.endplate_pos * 1000, y_pos, 'o', color='red')  # Highlighted in red

    # Plot electrode position if provided
    if electrode_pos is not None:
        # Represent electrode position as a black 'X' at electrode_pos[2] along Z-axis
        ax.scatter(electrode_pos[2].get() * 1000, electrode_pos[0].get() * 1000,
                   color='black', marker='x', s=100, label='Electrode')
        ax.legend()

    ax.set_xlabel('Z (mm)')
    ax.set_ylabel('Fiber Position (X in mm)')
    ax.set_title('Muscle Fibers Longitudinal View with Endplates')
    ax.grid(True)
    plt.show()

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
    ax.set_title(f'Fiber Placement within Muscle')
    plt.grid(True)
    plt.show()

def visualize_muscle_fibers(motor_units, electrode_pos=None):
    """Visualize the arrangement of muscle fibers in 3D, optionally displaying electrode position."""
    from mpl_toolkits.mplot3d import Axes3D  # Import here to avoid issues if not used elsewhere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for mu in motor_units:
        color = colors[mu.id % len(colors)]
        for fiber in mu.fibers:
            x = [fiber.x * 1000, fiber.x * 1000]
            y = [fiber.y * 1000, fiber.y * 1000]
            z = [fiber.z_start * 1000, fiber.z_end * 1000]
            ax.plot(x, y, z, color, linewidth=1)
            # Mark the endplate position
            ax.scatter(fiber.x * 1000, fiber.y * 1000, fiber.endplate_pos * 1000, c='red', marker='o')

    # Optionally plot electrode position
    if electrode_pos is not None:
        ax.scatter(electrode_pos[0].get() * 1000, electrode_pos[1].get() * 1000, electrode_pos[2].get() * 1000,
                   c='black', marker='x', s=100, label='Electrode')
        ax.legend()

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Muscle Fiber Arrangement (3D View)')
    plt.show()

def visualize_cross_section(motor_units, z_position=0.0, electrode_pos=None):
    """
    Visualize the cross-sectional view of muscle fibers at a given z-position, with an optional electrode position marked.
    """
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Define colors for different motor units
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Calculate the maximum radius to set axis limits dynamically
    all_radii = [fiber.radius * 1000 for mu in motor_units for fiber in mu.fibers]
    max_radius = max(all_radii) if all_radii else 1  # Use max radius or 1 if empty

    # Iterate through each motor unit and its fibers
    for mu in motor_units:
        color = colors[mu.id % len(colors)]
        for fiber in mu.fibers:
            # Only include fibers that pass through the specified z-position
            if fiber.z_start <= z_position <= fiber.z_end:
                # Convert x and y positions to mm
                x_pos = fiber.x * 1000  # Convert to mm
                y_pos = fiber.y * 1000  # Convert to mm

                # Radius in mm
                radius = fiber.radius * 1000  # Convert to mm

                # Draw the circle representing the fiber cross-section
                circle = patches.Circle((x_pos, y_pos), radius, edgecolor=color, facecolor='none', linewidth=1.5)
                ax.add_patch(circle)

    # Plot electrode position if provided
    if electrode_pos is not None:
        ax.scatter(electrode_pos[0].get() * 1000, electrode_pos[1].get() * 1000,
                   color='black', marker='x', s=100, label='Electrode')
        ax.legend()

    # Set axis limits based on muscle radius and maximum fiber radius
    muscle_radius_mm = 5  # Muscle radius in mm
    ax.set_xlim(-muscle_radius_mm - max_radius, muscle_radius_mm + max_radius)
    ax.set_ylim(-muscle_radius_mm - max_radius, muscle_radius_mm + max_radius)
    ax.set_aspect('equal', 'box')  # Ensure equal aspect ratio

    # Set axis labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(f'Muscle Fiber Cross-Section at Z = {z_position * 1000:.2f} mm')
    ax.grid(True)

    # Show the plot
    plt.show()

def visualize_longitudinal_view(motor_units, electrode_pos=None):
    """Visualize the longitudinal view showing fiber lengths and endplate positions, with an optional electrode position marked."""
    fig, ax = plt.subplots()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for mu in motor_units:
        color = colors[mu.id % len(colors)]
        for fiber in mu.fibers:
            # Represent each fiber as a line from z_start to z_end at a unique y-position
            y_pos = fiber.x * 1000  # Use x-position to spread fibers along y-axis
            ax.plot([fiber.z_start * 1000, fiber.z_end * 1000], [y_pos, y_pos], color, linewidth=1)
            # Mark the endplate position
            ax.plot(fiber.endplate_pos * 1000, y_pos, 'o', color='red')  # Highlighted in red

    # Plot electrode position if provided
    if electrode_pos is not None:
        # Represent electrode position as a black 'X' at electrode_pos[2] along Z-axis
        ax.scatter(electrode_pos[2].get() * 1000, electrode_pos[0].get() * 1000,
                   color='black', marker='x', s=100, label='Electrode')
        ax.legend()

    ax.set_xlabel('Z (mm)')
    ax.set_ylabel('Fiber Position (X in mm)')
    ax.set_title('Muscle Fibers Longitudinal View with Endplates')
    ax.grid(True)
    plt.show()

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
    ax.set_title(f'Fiber Placement within Muscle')
    plt.grid(True)
    plt.show()

def main():
    start_time = time.time()
    # Number of motor units and total number of fibers
    num_motor_units = 10
    total_num_fibers = 3000

    # Generate motor units and their fibers
    motor_units = generate_motor_units(num_motor_units, total_num_fibers)

    # Electrode settings
    electrode_type = 'cn'  # 'macro', 'sf', or 'cn'
    electrode_params = {
        'L_M': 10e-3,    # Length for macro electrode
        'R_c': 0.3e-3,   # Radius for CN electrode
        'N_c': 10,       # Number of divisions for CN electrode
        'x_SF': 0,       # x position for SF electrode
        'R_SF': 7.5e-3   # Distance from tip for SF electrode
    }
    electrode_pos = ELECTRODE_POS  # Electrode position

    # Visualize fiber placement with zoom into the central 1 mm radius area
    fibers = [fiber for mu in motor_units for fiber in mu.fibers]
    visualize_fiber_placement(fibers, muscle_radius=5e-3)

    # Visualize muscle fibers (3D View) with electrode
    visualize_muscle_fibers(motor_units, electrode_pos=electrode_pos)

    # Visualize cross-sectional view with electrode
    visualize_cross_section(motor_units, z_position=0.0, electrode_pos=electrode_pos)

    # Visualize longitudinal view with electrode
    visualize_longitudinal_view(motor_units, electrode_pos=electrode_pos)

    # Simulate EMG signal
    emg_signals_per_mu, total_emg_signal = simulate_emg_signal(
        motor_units, T, electrode_type, electrode_params, electrode_pos, FS, CUTOFF_FREQ
    )

    # Transfer the results back to CPU for plotting
    emg_signals_per_mu_cpu = [cp.asnumpy(emg_signal) for emg_signal in emg_signals_per_mu]
    total_emg_signal_cpu = cp.asnumpy(total_emg_signal)
    T_cpu = cp.asnumpy(T)

    # Plot the individual EMG signals and the total EMG signal
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for idx, emg_signal in enumerate(emg_signals_per_mu_cpu):
        plt.plot(T_cpu * 1000, emg_signal * 1e6, color=colors[idx % len(colors)],
                 label=f'Motor Unit {idx + 1}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (μV)')
        plt.title('Simulated EMG Signal by Motor Unit and Total EMG Signal')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot the total EMG signal
    plt.figure(figsize=(12, 6))
    plt.plot(T_cpu * 1000, total_emg_signal_cpu * 1e6, 'k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (μV)')
    plt.title('Total Simulated EMG Signal')
    plt.grid(True)
    plt.show()


    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
