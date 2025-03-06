
from datetime import datetime
import os

def find_closest_directory(parent_dir):
    current_time = datetime.now()
    
    directories = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    
    closest_time_diff = None
    closest_directory = None
    
    # Loop through all directories
    for dir_name in directories:
        try:
            # Try to parse the directory name into a datetime object
            dir_time = datetime.strptime(dir_name, "%Y%m%d_%H%M%S")
            
            # Calculate the time difference between the current time and the directory time
            time_diff = abs((current_time - dir_time).total_seconds())
            
            # Update the closest directory if this one is closer
            if closest_time_diff is None or time_diff < closest_time_diff:
                closest_time_diff = time_diff
                closest_directory = dir_name
        except ValueError:
            # Skip directories that don't match the expected format
            continue
    
    return closest_directory