import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Frequency and wavelength
freq = 900e6  # 900 MHz
c = 3e8       # Speed of light
wavelength = c / freq

# Yagi-Uda Element Details (optimized for back lobe suppression at 270 degrees)
element_positions = [0]  # x-positions along boom (meters)
element_lengths = [0.48 * wavelength]  # Reflector (longer for better reflection)

spacing_r_d = 0.15 * wavelength  # Reflector-to-driven spacing
spacing_d_dir = 0.18 * wavelength  # Tighter driven-to-director spacing for sharper beam
n_directors = 7  # Increased directors for enhanced directivity

element_positions.append(element_positions[-1] + spacing_r_d)  # Driven element
element_lengths.append(0.45 * wavelength)  # Driven element length

for i in range(n_directors):
    element_positions.append(element_positions[-1] + spacing_d_dir)
    element_lengths.append(0.42 * wavelength * (0.97 ** i))  # More aggressive length tapering

# Current magnitudes and phases (optimized for null at 270 degrees)
currents = [0.85] + [1.0] + [0.65 * (0.88 ** i) for i in range(n_directors)]  # Aggressive current tapering
phases = [0.0] + [0.0] + [-0.15 * (i + 1) for i in range(n_directors)]  # Progressive phase shift

# Array factor simulation with element pattern
theta = np.linspace(0, 2 * np.pi, 360)  # Radians
k = 2 * np.pi / wavelength  # Wave number

# Dipole element pattern (simplified, assuming short dipoles)
element_pattern = np.sin(theta)  # Dipole pattern (E-plane, sin(θ) for half-wave dipole)
element_pattern = np.where(element_pattern < 0, 0, element_pattern)  # Avoid negative values

AF = np.zeros_like(theta, dtype=complex)
for i, pos in enumerate(element_positions):
    AF += currents[i] * np.exp(1j * (k * pos * np.cos(theta) + phases[i]))

# Total pattern = array factor * element pattern
total_pattern = np.abs(AF) * element_pattern
total_pattern = np.where(total_pattern < 1e-6, 1e-6, total_pattern)  # Avoid log(0)

# Normalize and convert to dB
total_dB = 20 * np.log10(total_pattern / np.max(total_pattern))

# Calculate Front-to-Back Ratio (FBR) and back lobe at 270 degrees
theta_deg = np.degrees(theta)
front_idx = np.argmin(np.abs(theta_deg - 0))  # Forward direction (θ = 0°)
back_idx = np.argmin(np.abs(theta_deg - 270))  # Back lobe at 270°
FBR_dB = 20 * np.log10(total_pattern[front_idx] / total_pattern[back_idx])

# Plot 2D Radiation Pattern (polar)
plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)
ax.plot(theta, total_pattern / np.max(total_pattern))
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_title(f'Yagi-Uda 2D Radiation Pattern (E-Plane)')
ax.grid(True)
plt.savefig('yagi_2d_pattern_null_270.png')
plt.show()

# 3D Radiation Pattern
theta_grid, phi_grid = np.meshgrid(np.linspace(0, np.pi, 180), np.linspace(0, 2 * np.pi, 360))
element_pattern_3d = np.sin(theta_grid)  # Dipole pattern for 3D
element_pattern_3d = np.where(element_pattern_3d < 0, 0, element_pattern_3d)

AF_3D = np.zeros_like(theta_grid, dtype=complex)
for i, pos in enumerate(element_positions):
    AF_3D += currents[i] * np.exp(1j * (k * pos * np.cos(theta_grid) + phases[i]))

total_pattern_3d = np.abs(AF_3D) * element_pattern_3d
total_pattern_3d = np.where(total_pattern_3d < 1e-6, 1e-6, total_pattern_3d)
total_pattern_3d = total_pattern_3d / np.max(total_pattern_3d)

# Convert to Cartesian coordinates
x = total_pattern_3d * np.sin(theta_grid) * np.cos(phi_grid)
y = total_pattern_3d * np.sin(theta_grid) * np.sin(phi_grid)
z = total_pattern_3d * np.cos(theta_grid)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
ax.set_title('Yagi-Uda 3D Radiation Pattern')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.savefig('yagi_3d_pattern.png')
plt.show()

# Print antenna parameters and verify back lobe suppression
print("Yagi-Uda Antenna Parameters:")
for i, (pos, length, curr, phase) in enumerate(zip(element_positions, element_lengths, currents, phases)):
    element_name = "Reflector" if i == 0 else "Driven" if i == 1 else f"Director {i-1}"
    print(f"{element_name}: Position = {pos*100:.2f} cm, Length = {length*100:.2f} cm, Current = {curr:.2f}, Phase = {phase:.2f} rad")

print(f"Front-to-Back Ratio at 270°: {FBR_dB:.2f} dB")
print(f"Pattern magnitude at 270° (normalized): {total_pattern[back_idx] / np.max(total_pattern):.4f}")