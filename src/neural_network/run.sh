#!/bin/bash

# Check the value of the "task" environment variable
if [ "$task" = "softmax" ]; then
    # If "task" is "classification", run the classification.py script
    python3 softmax_regression.py
elif [ "$task" = "nn" ]; then
    # If "task" is "segmentation", run the segmentation.py script
    python3 nn.py
else
    # If "task" is neither "classification" nor "segmentation", print an error message
    echo "Unknown task: $task"
fi