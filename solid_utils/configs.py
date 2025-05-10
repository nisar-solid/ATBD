# Author: Emre Havazli, Ekaterina Tymofyeyeva
# May, 2025
# Utilities for changing configurations


def update_reference_point(config_file, new_lat, new_lon):
    # Read the file
    with open(config_file, "r") as file:
        lines = file.readlines()

    # Update the specific key
    for i, line in enumerate(lines):
        if "mintpy.reference.lalo" in line:
            lines[i] = f"mintpy.reference.lalo = {new_lat}, {new_lon}\n"
            break  # Stop after updating the line

    # Write back the modified file
    with open(config_file, "w") as file:
        file.writelines(lines)

    print(f"Updated mintpy.reference.lalo to {new_lat}, {new_lon}")