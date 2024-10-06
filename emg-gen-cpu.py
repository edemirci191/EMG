# EMG Simulator with Semi-Overlapping Fibers and Electrode Models, Mean Distributed End Plates
# Updated to use normal distribution for endplate positions
# Author: emre.demirci@tum.de (original code), updated by Assistant

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import bessel, lfilter
from scipy.fft import fft, ifft, fftfreq
import random
import matplotlib.patches as patches
from scipy.spatial import cKDTree

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
T = np.arange(0, DURATION, DT)  # Time vector
CUTOFF_FREQ = 5000  # Anti-aliasing filter cutoff frequency (Hz)
ORDER = 2     # Order of the Bessel filter

# Electrode position (e.g., at a quarter length along z-axis)
ELECTRODE_POS = np.array([0.00125, 0.00125, 0.025])  # Electrode position at quarter length
Z_0 = 0  # Reference point along z-axis

# Define the Intracellular Action Potential (IAP)
def iap(t, V0, duration):
    """Intracellular action potential modeled as a sine-squared pulse."""
    iap_signal = np.zeros_like(t)
    idx = (t >= 0) & (t <= duration)
    iap_signal[idx] = V0 * np.sin(np.pi * t[idx] / duration) ** 2
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
        # Generate spikes before the simulation starts to ensure coverage
        pre_spikes = np.arange(start_time - inter_spike_interval, -inter_spike_interval, -inter_spike_interval)[::-1]
        # Generate spikes throughout the duration
        main_spikes = np.arange(start_time, DURATION + inter_spike_interval, inter_spike_interval)
        # Combine pre-spikes and main spikes
        ap_times = np.concatenate((pre_spikes, main_spikes))
        # Only keep spikes within the simulation duration
        ap_times = ap_times[(ap_times >= 0) & (ap_times <= DURATION)]
        return ap_times

def generate_fibers(total_num_fibers, muscle_radius):
    """
    Generate muscle fibers within a circular muscle boundary using Variable Radius Poisson Disk Sampling,
    ensuring no overlap and realistic physiological distribution.
    """
    fibers = []
    
    # Define fiber diameters (random within physiological ranges)
    min_diameter = 10e-6  # 10 micrometers
    max_diameter = 80e-6  # 80 micrometers
    diameters = np.random.uniform(min_diameter, max_diameter, total_num_fibers)
    radii = diameters / 2
    
    # Initialize lists for positions and radii
    positions = []
    radii_list = []
    
    # Initialize active list with a random starting point within the muscle
    angle = random.uniform(0, 2 * np.pi)
    distance = random.uniform(0, muscle_radius - max(radii))
    x0 = distance * np.cos(angle)
    y0 = distance * np.sin(angle)
    
    positions.append([x0, y0])
    radii_list.append(radii[0])
    
    # Define fiber length
    z_start = -0.05  # Start of fiber (m)
    z_end = 0.05     # End of fiber (m)
    fiber_length = z_end - z_start
    
    # Create the first fiber
    conduction_velocity = 4.0  # (m/s)
    
    # **Set endplate position using normal distribution centered at fiber midpoint**
    mean_endplate_pos = (z_start + z_end) / 2
    std_dev = fiber_length * 0.05  # Standard deviation is 5% of fiber length
    endplate_pos = np.random.normal(mean_endplate_pos, std_dev)
    endplate_pos = np.clip(endplate_pos, z_start, z_end)
    
    fiber = MuscleFiber(x0, y0, z_start, z_end, diameters[0], conduction_velocity, endplate_pos, motor_unit_id=None)
    fibers.append(fiber)
    
    # Active list of points to process
    active_list = [[x0, y0, radii[0]]]
    
    # Use cKDTree for efficient neighbor searches
    kdtree = cKDTree(positions)
    
    # Index for radii
    idx = 1  # Start from the second fiber
    
    # Parameters
    k = 30  # Maximum number of attempts per active point
    
    while active_list and idx < total_num_fibers:
        current_point = random.choice(active_list)
        found = False
        for _ in range(k):
            # Generate a random point around the current point
            radius = radii[idx]
            min_distance = current_point[2] + radius  # Sum of radii
            
            # Random angle and distance
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(min_distance, 2 * min_distance)
            x_new = current_point[0] + distance * np.cos(angle)
            y_new = current_point[1] + distance * np.sin(angle)
            
            # Check if the new point is within the muscle boundary
            if x_new**2 + y_new**2 > (muscle_radius - radius)**2:
                continue  # Outside the muscle boundary
            
            # Check for overlaps using KDTree
            if positions:
                kdtree = cKDTree(positions)
                # Find all existing points within min_distance of the new point
                indices = kdtree.query_ball_point([x_new, y_new], r=min_distance)
                if len(indices) > 0:
                    continue  # Overlaps with existing fiber
            
            # Place the new fiber
            conduction_velocity = 4.0  # (m/s)
            
            # **Set endplate position using normal distribution centered at fiber midpoint**
            mean_endplate_pos = (z_start + z_end) / 2
            std_dev = fiber_length * 0.05  # Standard deviation is 5% of fiber length
            endplate_pos = np.random.normal(mean_endplate_pos, std_dev)
            endplate_pos = np.clip(endplate_pos, z_start, z_end)
            
            fiber = MuscleFiber(x_new, y_new, z_start, z_end, diameters[idx], conduction_velocity, endplate_pos, motor_unit_id=None)
            fibers.append(fiber)
            
            positions.append([x_new, y_new])
            radii_list.append(radius)
            active_list.append([x_new, y_new, radius])
            kdtree = cKDTree(positions)
            
            idx += 1
            found = True
            break  # Found a valid point, break the attempts loop
        
        if not found:
            active_list.remove(current_point)  # Remove point from active list
    
    if idx < total_num_fibers:
        print(f"Could only place {idx} fibers out of {total_num_fibers} requested.")
    
    return fibers

def generate_motor_units(num_motor_units, total_num_fibers):
    muscle_radius = 5e-3  # 5 mm in meters

    # Generate fibers without overlaps
    fibers = generate_fibers(total_num_fibers, muscle_radius)

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

# Impedance Function Z(x, y, z)
def impedance(x, y, z, sigma_r, K):
    """Calculate the impedance Z(x, y, z) between the current source and point (x, y, z)."""
    denominator = np.sqrt(K * (x ** 2 + y ** 2) + z ** 2)
    Z = 1 / (4 * np.pi * sigma_r * denominator)
    return Z

# Concentric Needle (CN) Electrode Weighting Function
def cn_electrode_weighting(fiber, z, sigma_r, K, R_c, N_c, z_0, electrode_pos):
    """
    Calculate the weighting function for the CN electrode.
    """
    r_x = fiber.x - electrode_pos[0]
    r_y = fiber.y - electrode_pos[1]
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

        numerator = np.sqrt((r_u + Delta_u_k) ** 2 + B_k) + r_u + Delta_u_k
        denominator = np.sqrt((r_u - Delta_u_k) ** 2 + B_k) + r_u - Delta_u_k

        # Avoid division by zero or negative values inside logarithm
        ln_argument = numerator / denominator
        ln_argument = np.where(ln_argument <= 0, 1e-12, ln_argument)

        # Use absolute value to ensure the sqrt is valid
        sqrt_term = np.sqrt(K * abs(Delta_u_k))

        # Avoid division by zero in case sqrt_term is zero
        if sqrt_term == 0:
            continue

        weighting_k = (1 / (8 * np.pi * sigma_r * sqrt_term)) * np.log(ln_argument)
        weighting_sum += weighting_k
        valid_k_count += 1  # Increment count of valid terms

    # Adjust normalization factor
    if valid_k_count > 0:
        weighting = weighting_sum / valid_k_count
    else:
        weighting = 0  # If no valid terms, weighting is zero

    return weighting

# Compute the Temporal Weighting Function
def compute_weighting_function(fiber, t_w, electrode_type, electrode_params, electrode_pos):
    """
    Compute the temporal weighting function w(t) for a given fiber and electrode type.

    Args:
        fiber (MuscleFiber): The muscle fiber for which the weighting function is computed.
        t_w (numpy array): Time vector for the weighting function.
        electrode_type (str): Type of electrode ('macro', 'sf', 'cn').
        electrode_params (dict): Parameters specific to the electrode type.
        electrode_pos (numpy array): Position of the electrode [x, y, z].

    Returns:
        numpy array: Temporal weighting function w(t).
    """
    # Constants
    sigma_r = SIGMA_R
    K_value = K
    z_0 = Z_0

    # Conduction velocity
    c = fiber.conduction_velocity

    # Initialize weighting function
    w_t = np.zeros_like(t_w)

    # Only 'cn' electrode type is considered in this version
    if electrode_type == 'cn':
        # Extract electrode parameters
        R_c = electrode_params.get('R_c', 0.3e-3)  # Radius of CN electrode
        N_c = electrode_params.get('N_c', 10)      # Number of divisions
        # Compute weighting function over time
        for idx, t_val in enumerate(t_w):
            z = c * t_val + fiber.endplate_pos
            w_t[idx] = cn_electrode_weighting(fiber, z, sigma_r, K_value, R_c, N_c, z_0, electrode_pos)
    else:
        raise ValueError(f"Unknown electrode type: {electrode_type}")

    return w_t

def generate_action_potential_shape(t):
    """Generate a simple action potential shape."""
    # Parameters for IAP
    V0 = 100e-3     # Peak amplitude (V)
    duration = 2e-3  # Duration of IAP (s)
    return iap(t, V0, duration)

def apply_bandpass_filter(signal, fs, cutoff_freq):
    """Apply a low-pass Bessel filter to the signal."""
    b, a = bessel(ORDER, 2 * cutoff_freq / fs, btype='low', analog=False)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def compute_fiber_ap(fiber, t, electrode_type, electrode_params, electrode_pos, fs, cutoff_freq, ap_times):
    """
    Compute the action potential for a single fiber using convolution via FFT.

    Args:
        fiber (MuscleFiber): The muscle fiber object.
        t (numpy array): Time vector.
        electrode_type (str): Type of electrode ('macro', 'sf', 'cn').
        electrode_params (dict): Parameters specific to the electrode type.
        electrode_pos (numpy array): Position of the electrode [x, y, z].
        fs (float): Sampling frequency.
        cutoff_freq (float): Cutoff frequency for the bandpass filter.
        ap_times (numpy array): Times at which action potentials occur.

    Returns:
        numpy array: Simulated action potential for the fiber.
    """
    # Compute the weighting function
    c = fiber.conduction_velocity
    tau_plus = (fiber.z_end - fiber.endplate_pos) / c
    tau_minus = (fiber.endplate_pos - fiber.z_start) / c
    t_w = np.arange(-tau_minus, tau_plus, DT)
    w_t = compute_weighting_function(fiber, t_w, electrode_type, electrode_params, electrode_pos)

    # Generate the action potential shape
    ap_shape = generate_action_potential_shape(t_w)

    # Compute the transmembrane current (second derivative)
    dt = 1 / fs
    im = np.gradient(np.gradient(ap_shape, dt), dt)

    # Apply bandpass filter
    im_filtered = apply_bandpass_filter(im, fs, cutoff_freq)

    # Perform convolution using FFT
    n_conv = len(im_filtered) + len(w_t) - 1
    n_fft = 2 ** int(np.ceil(np.log2(n_conv)))
    im_padded = np.zeros(n_fft)
    w_padded = np.zeros(n_fft)
    im_padded[:len(im_filtered)] = im_filtered
    w_padded[:len(w_t)] = w_t

    # Perform FFT
    IM = fft(im_padded)
    W = fft(w_padded)
    V = IM * W
    v_t_full = ifft(V).real

    # Extract the valid part of the convolution
    v_single = v_t_full[:n_conv]

    # Time vector for the convolved signal
    t_conv = np.arange(0, n_conv * DT, DT)

    # Initialize the fiber's EMG signal
    fiber_emg = np.zeros_like(t)

    # Sum contributions for each AP initiation time
    for ap_time in ap_times:
        # Shift the signal according to the AP initiation time
        t_shifted = t_conv + ap_time
        # Interpolate and add to the EMG signal
        shifted_v = np.interp(t, t_shifted, v_single, left=0, right=0)
        fiber_emg += shifted_v

    return fiber_emg

def simulate_emg_signal(motor_units, t, electrode_type, electrode_params, electrode_pos, fs, cutoff_freq):
    """
    Simulate the EMG signal by summing the contributions from all motor units.

    Args:
        motor_units (list): List of MotorUnit objects.
        t (numpy array): Time vector.
        electrode_type (str): Type of electrode ('macro', 'sf', 'cn').
        electrode_params (dict): Parameters specific to the electrode type.
        electrode_pos (numpy array): Position of the electrode [x, y, z].
        fs (float): Sampling frequency.
        cutoff_freq (float): Cutoff frequency for the bandpass filter.

    Returns:
        emg_signals_per_mu: List of EMG signals, one per motor unit.
        total_emg_signal: The sum of all EMG signals from motor units.
    """
    emg_signals_per_mu = []
    total_emg_signal = np.zeros_like(t)

    for mu in motor_units:
        emg_signal_mu = np.zeros_like(t)
        for fiber in mu.fibers:
            fiber_emg = compute_fiber_ap(fiber, t, electrode_type, electrode_params, electrode_pos, fs, cutoff_freq, mu.ap_times)
            emg_signal_mu += fiber_emg
        emg_signals_per_mu.append(emg_signal_mu)
        total_emg_signal += emg_signal_mu

    return emg_signals_per_mu, total_emg_signal

# Visualization functions

# Visualize the Muscle Fibers (3D View)
def visualize_muscle_fibers(motor_units):
    """Visualize the arrangement of muscle fibers in 3D."""
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
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Muscle Fiber Arrangement (3D View)')
    plt.show()

def visualize_cross_section(motor_units, z_position=0.0):
    """
    Visualize the cross-sectional view of muscle fibers at a given z-position.
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

# Visualize Longitudinal View
def visualize_longitudinal_view(motor_units):
    """Visualize the longitudinal view showing fiber lengths and endplate positions."""
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
    ax.set_title('Fiber Placement within Muscle')
    plt.grid(True)
    plt.show()

def main():
    # Number of motor units and total number of fibers
    num_motor_units = 1
    total_num_fibers = 20  # Total number of fibers across all motor units

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
    visualize_fiber_placement(fibers, muscle_radius=5e-3, zoom_center=(0, 0), zoom_radius=1e-3)

    # Visualize muscle fibers (3D View)
    visualize_muscle_fibers(motor_units)

    # Visualize cross-sectional view
    visualize_cross_section(motor_units, z_position=0.0)  # Cross-section at Z = 0 mm

    # Visualize longitudinal view
    visualize_longitudinal_view(motor_units)

    # Simulate EMG signal
    emg_signals_per_mu, total_emg_signal = simulate_emg_signal(
        motor_units, T, electrode_type, electrode_params, electrode_pos, FS, CUTOFF_FREQ
    )

    # Plot the individual EMG signals and the total EMG signal
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for idx, emg_signal in enumerate(emg_signals_per_mu):
        plt.plot(T * 1000, emg_signal * 1e6, color=colors[idx % len(colors)],
                 label=f'Motor Unit {idx + 1}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (μV)')
    plt.title('Simulated EMG Signal by Motor Unit')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the total EMG signal separately
    plt.figure(figsize=(12, 6))
    plt.plot(T * 1000, total_emg_signal * 1e6, 'k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (μV)')
    plt.title('Total Simulated EMG Signal')
    plt.grid(True)
    plt.show()

    # Compute and plot the frequency spectrum
    N = len(total_emg_signal)
    emg_fft = fft(total_emg_signal)
    freq = fftfreq(N, DT)
    idx = np.argsort(freq)

    plt.figure(figsize=(12, 6))
    plt.plot(freq[idx], np.abs(emg_fft[idx]))
    plt.xlim(0, 500)  # Limit frequency axis to 500 Hz
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Frequency Spectrum of Total Simulated EMG Signal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
