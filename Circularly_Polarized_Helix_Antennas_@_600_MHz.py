import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi

# Design parameters
freq = 600e6  # 600 MHz
wavelength = c / freq  # Wavelength in meters
Z0 = 50  # Characteristic impedance (ohms)

# Helix antenna design equations
def helix_design_normal_mode(wavelength, turns):
    """Design for Normal Mode Helix"""
    circumference = wavelength / pi  # C = λ/π
    radius = circumference / (2 * pi)
    pitch = wavelength / 4  # S = λ/4
    length = turns * pitch
    return radius, pitch, length, circumference

def helix_design_axial_mode(wavelength, turns):
    """Optimized Design for Axial Mode Helix with Circular Polarization"""
    circumference = wavelength  # C ≈ λ
    radius = circumference / (2 * pi)
    # Optimize pitch angle for circular polarization (~13 degrees)
    pitch_angle = 13 * pi / 180  # 13 degrees in radians
    pitch = np.tan(pitch_angle) * circumference  # S = tan(α) * C
    length = turns * pitch
    impedance = 140 * (circumference / wavelength)  # Z ≈ 140 * (C/λ)
    return radius, pitch, length, circumference, impedance

# Calculate for both modes
turns_normal = 2  # Short helix for normal mode
turns_axial = 12  # Increased turns for axial mode for better circular polarization

# Normal Mode Design
radius_n, pitch_n, length_n, circ_n = helix_design_normal_mode(wavelength, turns_normal)
print("Normal Mode Helix:")
print(f"Radius: {radius_n * 100:.2f} cm")
print(f"Pitch: {pitch_n * 100:.2f} cm")
print(f"Length: {length_n * 100:.2f} cm")
print(f"Circumference: {circ_n * 100:.2f} cm")
print(f"Number of Turns: {turns_normal}")

# Axial Mode Design
radius_a, pitch_a, length_a, circ_a, impedance_a = helix_design_axial_mode(wavelength, turns_axial)
print("\nAxial Mode Helix (Optimized for Circular Polarization):")
print(f"Radius: {radius_a * 100:.2f} cm")
print(f"Pitch: {pitch_a * 100:.2f} cm")
print(f"Length: {length_a * 100:.2f} cm")
print(f"Circumference: {circ_a * 100:.2f} cm")
print(f"Number of Turns: {turns_axial}")
print(f"Impedance: {impedance_a:.2f} ohms")

# Simulated Radiation Pattern (Refined Model with Back and Side Lobes)
def plot_radiation_pattern(mode, radius, pitch, length, turns, axial_ratio_db, bandwidth_mhz):
    """Plot refined 2D radiation pattern with thinner main lobe and annotations"""
    theta = np.linspace(0, 2 * pi, 1000)  # Fine resolution
    if mode == "normal":
        # Normal mode: Broadside pattern
        pattern = np.cos(theta) ** 2
    else:
        # Axial mode: End-fire pattern, thinner main lobe
        base_pattern = (0.8 * (1 + np.cos(theta)) / 2 + 0.2) ** (turns)
        side_lobes = 0.1 * np.abs(np.sin(3 * theta))
        pattern = base_pattern + side_lobes
        pattern = pattern / np.max(pattern)  # Normalize

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    # Rotate axial mode main lobe to vertical (theta = 90 degrees)
    if mode == "axial":
        ax.set_theta_zero_location('N')  # North (top) is 0°
        ax.set_theta_direction(-1)  # Clockwise
    else:
        ax.set_theta_zero_location('E')  # East (right) is 0°
        ax.set_theta_direction(-1)  # Clockwise for normal mode too, consistent

    ax.plot(theta, pattern, label=f"{mode.capitalize()} Mode")
    ax.set_title(f"Radiation Pattern ({mode.capitalize()} Mode)", va='bottom')

    # Add text box with Axial Ratio and Bandwidth
    textstr = '\n'.join((
        f"Axial Ratio: {axial_ratio_db:.2f} dB",
        f"Bandwidth: {bandwidth_mhz:.2f} MHz"))
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

    plt.legend(loc='upper right')
    plt.savefig(f"{mode}_radiation_pattern_thinner_main_lobe_annotated.png")
    plt.show()
    plt.close()


# Axial Ratio (Updated for Practical Accuracy)
def axial_ratio(mode, turns, circumference, wavelength):
    """Estimate axial ratio (updated for practical circular polarization)"""
    if mode == "axial":
        # Practical formula for axial mode helix: AR = (2N + 1) / (2N)
        ar = (2 * turns + 1) / (2 * turns)
        ar_db = 20 * np.log10(ar)  # Convert to dB
    else:
        ar = 1  # Normal mode is typically linearly polarized
        ar_db = 20 * np.log10(ar)
    return ar, ar_db



# Calculate axial ratios
ar_normal, ar_normal_db = axial_ratio("normal", turns_normal, circ_n, wavelength)
ar_axial, ar_axial_db = axial_ratio("axial", turns_axial, circ_a, wavelength)
print("\nPolarization Characteristics:")
print(f"Normal Mode Axial Ratio: {ar_normal:.2f} ({ar_normal_db:.2f} dB, Linear)")
print(f"Axial Mode Axial Ratio: {ar_axial:.2f} ({ar_axial_db:.2f} dB, Circular)")

# Bandwidth Estimation (Simplified)
def estimate_bandwidth(mode, impedance, freq):
    """Estimate input bandwidth (simplified)"""
    if mode == "axial":
        bw = 0.52 * freq / np.sqrt(impedance / Z0)  # Approx. 52% of center freq
    else:
        bw = 0.1 * freq  # Normal mode has narrower bandwidth
    return bw / 1e6  # Convert to MHz

# Plot radiation patterns
# Plot updated radiation patterns with annotations
plot_radiation_pattern("normal", radius_n, pitch_n, length_n, turns_normal, ar_normal_db, ar_normal)
plot_radiation_pattern("axial", radius_a, pitch_a, length_a, turns_axial, ar_axial_db, ar_axial)

bw_normal = estimate_bandwidth("normal", Z0, freq)
bw_axial = estimate_bandwidth("axial", impedance_a, freq)
print("\nBandwidth Estimation:")
print(f"Normal Mode Bandwidth: {bw_normal:.2f} MHz")
print(f"Axial Mode Bandwidth: {bw_axial:.2f} MHz")

# Note: NEC-Python simulation is omitted due to complexity and environment constraints.
# For accurate results, use NEC2/NEC4 with the following parameters:
# - Normal Mode: Radius, pitch, turns as calculated, ground plane optional
# - Axial Mode: Radius, pitch, turns, no ground plane
# - Frequency sweep: 550-650 MHz
# - Outputs: Gain, axial ratio, impedance
