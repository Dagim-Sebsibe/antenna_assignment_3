import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 3e8
f = 900e6  # Frequency 900 MHz (same as Yagi for consistency)
wavelength = c / f
k = 2 * np.pi / wavelength  # Wave number

# Array Parameters
N_elements_list = [4, 6, 8]  # Different number of dipoles
spacing_values = np.linspace(0.1, 2.0, 50) * wavelength  # Spacing from 0.1λ to 2.0λ

# Function to compute Array Factor (broadside array)
def array_factor(N, d, theta):
    psi = k * d * np.cos(theta)
    numerator = np.sin(N * psi / 2)
    denominator = N * np.sin(psi / 2)
    # Handle division by zero carefully
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    AF = numerator / denominator
    return np.abs(AF)

# Function to compute Directivity (approximate)
def compute_directivity(N, d):
    theta = np.linspace(0, np.pi, 500)
    AF = array_factor(N, d, theta)
    power_pattern = AF**2
    total_power = np.trapz(power_pattern * np.sin(theta), theta)
    max_power = np.max(power_pattern)
    D = 4 * np.pi * max_power / total_power
    return D

# Plot Directivity vs Spacing for different N
plt.figure(figsize=(10, 6))

for N in N_elements_list:
    directivities = []
    for d in spacing_values:
        D = compute_directivity(N, d)
        directivities.append(D)
    plt.plot(spacing_values / wavelength, directivities, label=f'N = {N} elements')

plt.title('Uniform Linear Array Directivity vs Element Spacing')
plt.xlabel('Element Spacing (λ)')
plt.ylabel('Directivity (dimensionless)')
plt.grid(True)
plt.legend()
plt.savefig('ula_directivity_vs_spacing.png')
plt.show()
