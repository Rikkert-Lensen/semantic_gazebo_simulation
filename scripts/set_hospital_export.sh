#!/bin/bash
echo "Setting IGN_GAZEBO_RESOURCE_PATH"
# Replace 'package_name' with the actual name of the package
package_name="semantic_gazebo_simulation"

# Save the output of 'rospack find' to a variable
package_path=$(rospack find $package_name)

# Create a variable for the models folder path
models_path="$package_path/models/aws-robomaker-hospital-world-ros1/models"
fuel_models_path="$package_path/models/aws-robomaker-hospital-world-ros1/fuel_models"

# Set the IGN_GAZEBO_RESOURCE_PATH environment variable
export IGN_GAZEBO_RESOURCE_PATH=$models_path:$fuel_models_path:$IGN_GAZEBO_RESOURCE_PATH
echo "Finished setting IGN_GAZEBO_RESOURCE_PATH"

