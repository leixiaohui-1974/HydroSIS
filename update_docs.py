from hydrosis.testing.example_documenter import generate_example_documentation
from pathlib import Path

# Define the destination directory
destination_dir = Path("docs/examples")
destination_dir.mkdir(exist_ok=True)

# Generate the documentation directly in the destination directory
generate_example_documentation(destination_dir)

print("Documentation updated successfully.")