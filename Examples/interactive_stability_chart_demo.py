"""
Interactive Stability Chart Demo
=================================

This example demonstrates the new interactive stability chart feature with spectrum overlay.

The new `interactive=True` parameter in `plot_stab()` combines:
- Interactive pole selection (like mpe_from_plot)
- Spectrum overlay (CMIF plot)
- Real-time spectrum toggling

Usage:
------
# Old way - separate methods:
ssicov.plot_stab(spectrum=True)  # Static plot only
ssicov.mpe_from_plot()            # Interactive but no spectrum

# New way - integrated:
ssicov.plot_stab(spectrum=True, interactive=True)  # Both features together!

Interactive Controls:
--------------------
- SHIFT + LEFT mouse: Select a pole
- SHIFT + RIGHT mouse: Remove last pole
- SHIFT + MIDDLE mouse: Remove closest pole
- Menu > Spectrum Overlay: Toggle spectrum visibility
- Menu > Show/Hide Unstable Poles: Toggle pole visibility
"""

import numpy as np
from pyoma2.algorithms import SSI, pLSCF
from pyoma2.setup import SingleSetup

# Generate synthetic data (5-DOF system)
def generate_test_data():
    """Generate synthetic measurement data for testing."""
    fs = 100  # Sampling frequency
    N = 10000  # Number of samples
    t = np.arange(N) / fs

    # Modal parameters (natural frequencies in Hz)
    fn = np.array([1.5, 2.0, 3.5, 4.2, 5.8])
    xi = np.array([0.02, 0.015, 0.03, 0.025, 0.02])

    # Generate multi-channel response
    n_channels = 6
    data = np.zeros((n_channels, N))

    for ch in range(n_channels):
        signal = np.random.randn(N) * 0.1  # Base noise
        for i in range(len(fn)):
            # Damped oscillation
            omega = 2 * np.pi * fn[i]
            phi = np.random.rand() * 2 * np.pi
            amp = np.random.rand() + 0.5
            signal += amp * np.exp(-xi[i] * omega * t) * np.cos(omega * t + phi)
        data[ch, :] = signal

    return data, fs


def demo_ssi_interactive():
    """Demo 1: SSI algorithm with interactive stability chart + spectrum."""
    print("\n" + "="*70)
    print("DEMO 1: SSI - Interactive Stability Chart with Spectrum Overlay")
    print("="*70)

    # Generate data
    data, fs = generate_test_data()

    # Setup
    setup = SingleSetup(data, fs=fs)

    # Initialize SSI algorithm
    ssicov = SSI(
        name="SSIcov",
        method="cov",
        br=50,
        ordmax=60,
        ordmin=2,
        step=2,
    )

    setup.add_algorithms(ssicov)
    setup.run_by_name("SSIcov")

    print("\n📊 Opening interactive plot...")
    print("\nInstructions:")
    print("  1. SHIFT + LEFT click to select poles")
    print("  2. Use 'Spectrum Overlay' menu to show/hide CMIF")
    print("  3. Use 'Show/Hide Unstable Poles' menu to toggle visibility")
    print("  4. Close window when done")

    # NEW FEATURE: Interactive + Spectrum in one call!
    ssicov.plot_stab(
        freqlim=(0, 8),
        spectrum=True,      # Enable spectrum overlay
        interactive=True,   # Enable interactive mode
        nSv=3,             # Show top 3 singular values
    )

    print("\n✅ Interactive session completed!")


def demo_plscf_interactive():
    """Demo 2: pLSCF algorithm with interactive stability chart + spectrum."""
    print("\n" + "="*70)
    print("DEMO 2: pLSCF - Interactive Stability Chart with Spectrum Overlay")
    print("="*70)

    # Generate data
    data, fs = generate_test_data()

    # Setup
    setup = SingleSetup(data, fs=fs)

    # Initialize pLSCF algorithm
    plscf = pLSCF(
        name="polymax",
        ordmax=40,
        ordmin=2,
    )

    setup.add_algorithms(plscf)
    setup.run_by_name("polymax")

    print("\n📊 Opening interactive plot...")
    print("\nInstructions:")
    print("  1. SHIFT + LEFT click to select poles")
    print("  2. Use 'Spectrum Overlay' menu to show/hide CMIF")
    print("  3. Close window when done")

    # NEW FEATURE: Interactive + Spectrum for pLSCF too!
    plscf.plot_stab(
        freqlim=(0, 8),
        spectrum=True,      # Enable spectrum overlay
        interactive=True,   # Enable interactive mode
        nSv="all",         # Show all singular values
    )

    print("\n✅ Interactive session completed!")


def demo_comparison():
    """Demo 3: Compare old vs new API."""
    print("\n" + "="*70)
    print("DEMO 3: API Comparison - Old vs New")
    print("="*70)

    # Generate data
    data, fs = generate_test_data()
    setup = SingleSetup(data, fs=fs)
    ssicov = SSI(name="SSIcov", method="cov", br=50, ordmax=60, step=2)
    setup.add_algorithms(ssicov)
    setup.run_by_name("SSIcov")

    print("\n📋 OLD WAY (2 separate methods):")
    print("   1. ssicov.plot_stab(spectrum=True)  # View only")
    print("   2. ssicov.mpe_from_plot()           # Pick modes (no spectrum)")

    print("\n📋 NEW WAY (1 integrated method):")
    print("   ssicov.plot_stab(spectrum=True, interactive=True)")
    print("   ✨ Pick modes + View spectrum simultaneously!")

    print("\n🚀 Benefits:")
    print("   ✓ Fewer method calls")
    print("   ✓ Spectrum helps identify physical modes")
    print("   ✓ Toggle spectrum on/off during picking")
    print("   ✓ Consistent API across SSI and pLSCF")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  Interactive Stability Chart with Spectrum Overlay - Demo")
    print("="*70)

    # Run demos
    print("\nSelect demo to run:")
    print("  1. SSI Interactive (with spectrum)")
    print("  2. pLSCF Interactive (with spectrum)")
    print("  3. API Comparison (information only)")
    print("  4. Run all demos")

    choice = input("\nEnter choice (1-4) or press Enter for demo 3: ").strip()

    if choice == "1":
        demo_ssi_interactive()
    elif choice == "2":
        demo_plscf_interactive()
    elif choice == "4":
        demo_ssi_interactive()
        demo_plscf_interactive()
        demo_comparison()
    else:
        demo_comparison()

    print("\n" + "="*70)
    print("Demo completed! 🎉")
    print("="*70)
