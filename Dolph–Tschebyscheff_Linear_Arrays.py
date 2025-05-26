import numpy as np
import matplotlib.pyplot as plt
from scipy.special import chebyt

# Constants
c = 3e8  # Speed of light (m/s)
f = 900e6  # Frequency (Hz)
wavelength = c / f  # Wavelength (m)
k = 2 * np.pi / wavelength  # Wavenumber

# Array parameters
N = 10  # Number of elements
d = 0.5 * wavelength  # Spacing: half wavelength

# Side-lobe suppression levels (dB)
side_lobe_levels = [20, 30, 40]

# Function to compute Chebyshev weights using chebyt
def chebyshev_weights(N, R_dB):
    R = 10 ** (R_dB / 20)  # Convert dB to linear scale
    x0 = np.cosh(np.arccosh(R) / (N - 1))  # Chebyshev parameter
    weights = np.zeros(N)
    for m in range(N):
        x = x0 * np.cos(np.pi * (2 * m + 1) / (2 * N))
        weights[m] = chebyt(N - 1)(x)
    weights /= np.max(np.abs(weights))
    return weights

# Function to compute Array Factor
def array_factor(theta, weights, d, k):
    N = len(weights)
    psi = k * d * np.cos(theta)
    AF = np.zeros_like(theta, dtype=complex)
    for n in range(N):
        AF += weights[n] * np.exp(1j * psi * (n - (N - 1) / 2))
    return np.abs(AF) / np.max(np.abs(AF))  # Normalize to 1

# Function to compute 3 dB beamwidth
def compute_beamwidth(theta_deg, AF_db):
    peak_idx = np.argmin(np.abs(theta_deg - 90))
    peak_db = AF_db[peak_idx]
    threshold = peak_db - 3
    left_idx = np.where(AF_db[:peak_idx] <= threshold)[0]
    right_idx = np.where(AF_db[peak_idx:] <= threshold)[0]
    if len(left_idx) == 0 or len(right_idx) == 0:
        return np.nan
    left_angle = theta_deg[left_idx[-1]]
    right_angle = theta_deg[peak_idx + right_idx[0]]
    return abs(right_angle - left_angle)

# Plot
theta = np.linspace(0, np.pi, 1000)
theta_deg = np.degrees(theta)

plt.figure(figsize=(10, 6))

# Store beamwidths and side-lobe levels for comparison
beamwidths = {}
sll_measured = {}

for r in side_lobe_levels:
    weights = chebyshev_weights(N, r)
    AF = array_factor(theta, weights, d, k)
    AF_db = 20 * np.log10(AF + 1e-10)
    plt.plot(theta_deg, AF_db, label=f'{r} dB SLL')
    beamwidth = compute_beamwidth(theta_deg, AF_db)
    beamwidths[r] = beamwidth
    main_lobe_region = (theta_deg > 60) & (theta_deg < 120)
    side_lobes = AF_db[~main_lobe_region]
    sll_measured[r] = np.max(side_lobes) if side_lobes.size > 0 else np.nan

# Uniform array for comparison
uniform_weights = np.ones(N)
AF_uniform = array_factor(theta, uniform_weights, d, k)
AF_uniform_db = 20 * np.log10(AF_uniform + 1e-10)
plt.plot(theta_deg, AF_uniform_db, '--', label='Uniform (no taper)')
beamwidths['Uniform'] = compute_beamwidth(theta_deg, AF_uniform_db)
side_lobes_uniform = AF_uniform_db[~main_lobe_region]
sll_measured['Uniform'] = np.max(side_lobes_uniform) if side_lobes_uniform.size > 0 else np.nan

plt.title('Dolph–Tschebyscheff Array Factor Comparison')
plt.xlabel('Angle θ (degrees)')
plt.ylabel('Array Factor (dB)')
plt.ylim([-60, 0])
plt.grid(True)
plt.legend()

# Add comparison table as text box on the plot
comparison_text = "Comparison of SLL and Beamwidths:\n"
comparison_text += f"{'Taper':<15} {'SLL (dB)':<12} {'Beamwidth (deg)':<15}\n"
comparison_text += "-" * 40 + "\n"
for r in side_lobe_levels:
    comparison_text += f"{r} dB SLL:      {sll_measured[r]:<12.2f} {beamwidths[r]:<15.2f}\n"
comparison_text += f"{'Uniform':<15} {sll_measured['Uniform']:<12.2f} {beamwidths['Uniform']:<15.2f}"

# Place text box in the bottom-left corner
plt.text(0.02, 0.02, comparison_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8))

# Save the plot
plt.savefig('chebyshev_array_factors.png')

# Print comparison to console (optional)
print(comparison_text)

plt.show()