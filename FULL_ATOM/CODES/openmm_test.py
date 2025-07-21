# test_openmm.py (Corrected Version)
import openmm as mm
import sys

print("--- OpenMM Installation Test ---")

try:
    # Test 1: List available platforms correctly
    print("\n[Test 1] Listing available platforms...")
    num_platforms = mm.Platform.getNumPlatforms()
    if num_platforms == 0:
        print("Error: No OpenMM platforms found!")
        sys.exit(1)
        
    for i in range(num_platforms):
        platform = mm.Platform.getPlatform(i)
        print(f"  Platform {i}: {platform.getName()}")
    print("Platform listing successful.")

    # A minimal system to use for context creation
    system = mm.System()
    system.addParticle(1.0) # A system with one particle
    integrator = mm.VerletIntegrator(1.0 * mm.unit.femtoseconds)

    # Test 2: Attempt to create a context on the CPU
    print("\n[Test 2] Attempting to create a minimal CPU context...")
    cpu_platform = mm.Platform.getPlatformByName('CPU')
    context_cpu = mm.Context(system, integrator, cpu_platform)
    print("  SUCCESS: CPU context created successfully.")
    # Clean up immediately
    del context_cpu

    # Test 3: Attempt to create a context on the GPU (if available)
    try:
        cuda_platform = mm.Platform.getPlatformByName('CUDA')
        print("\n[Test 3] Attempting to create a minimal CUDA context...")
        properties = {'CudaPrecision': 'mixed'}
        context_cuda = mm.Context(system, integrator, cuda_platform, properties)
        print("  SUCCESS: CUDA context created successfully.")
        del context_cuda
    except mm.OpenMMException:
        print("\n[Test 3] SKIPPED: CUDA platform not found or failed to load.")

    # Clean up remaining objects
    del integrator, system

except Exception as e:
    print(f"\n--- AN ERROR OCCURRED ---")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Details: {e}")
    print("-------------------------")
    # Exit with a non-zero code to indicate failure
    sys.exit(1)

print("\n--- Test finished successfully ---")