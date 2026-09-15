import sys
import matplotlib
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print("INFO (Wrapper): Initializing...")

show_plots = "--showPlots" in sys.argv

if show_plots:
    try:
        matplotlib.use('TkAgg')
        print("INFO (Wrapper): Set interactive Matplotlib backend 'TkAgg'.")
    except ImportError:
        print("INFO (Wrapper): 'TkAgg' unavailable, attempting 'Qt5Agg'.")
        try:
            matplotlib.use('Qt5Agg')
            print("INFO (Wrapper): Set interactive backend 'Qt5Agg'.")
        except ImportError:
            matplotlib.use('Agg')
            print("\nWARNING (Wrapper): Failed to load 'TkAgg' or 'Qt5Agg' backends.")
            print("WARNING (Wrapper): Despite --showPlots, plots will not be displayed interactively.")
            print("WARNING (Wrapper): Set fallback backend 'Agg' (file export will continue to function).\n")
else:
    matplotlib.use('Agg')
    print("INFO (Wrapper): Set non-interactive Matplotlib backend 'Agg'.")

print(f"INFO (Wrapper): Selected backend: {matplotlib.get_backend()}")
sys.stdout.flush()

print("INFO (Wrapper): Importing main.py...")
sys.stdout.flush()

try:
    import main
    print("INFO (Wrapper): Successfully imported 'main.py'.")
    sys.stdout.flush()
except ImportError as e:
    print(f"\nERROR (Wrapper): Failed to import 'main.py'!")
    print(f"ERROR (Wrapper): Ensure 'main.py' is located in the same directory as 'wrapper.py'.")
    print(f"Error details: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\nERROR (Wrapper): An unexpected error occurred while importing 'main.py'.")
    print(f"Error details: {e}")
    sys.exit(1)

if hasattr(main, 'main'):
    print("\n--- Executing main.py pipeline logic ---")
    main.main()
    print("--- main.py execution completed ---")
else:
    print("ERROR (Wrapper): 'main.py' does not define a 'main()' entry point.")
    sys.exit(1)

print("INFO (Wrapper): Process completed.")