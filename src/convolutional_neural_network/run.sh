#!/bin/bash

# Check the value of the "task" environment variable
if [ "$task" = "classification" ]; then
    # If "task" is "classification", run the classification.py script
    python3 digit_classification.py
elif [ "$task" = "segmentation" ]; then
    # If "task" is "segmentation", run the segmentation.py script
    python3 semantic_seg.py
else
    # If "task" is neither "classification" nor "segmentation", print an error message
    echo "Unknown task: $task"
fi