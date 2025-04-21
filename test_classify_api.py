#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Test script for the classify API in vllm.

This script demonstrates how to use the classify API to get classification head logits
and apply softmax to get probabilities.
"""

import argparse
import json
import requests
import numpy as np


def softmax(x):
    """Apply softmax function to convert logits to probabilities."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def sigmoid(x):
    """Apply sigmoid function for binary classification."""
    return 1 / (1 + np.exp(-x))


def test_classify_api(server_url, model_name, prompts):
    """Test the classify API endpoint."""
    # Prepare the request
    request_data = {
        "model": model_name,
        "input": prompts,
        "encoding_format": "float",  # Use float format for easier processing
    }

    # Send the request to the classify API
    response = requests.post(
        f"{server_url}/v1/classify",
        headers={"Content-Type": "application/json"},
        json=request_data,
    )

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    # Parse the response
    result = response.json()
    
    # Print the raw response
    print("\nRaw API Response:")
    print(json.dumps(result, indent=2))
    
    # Process and display the results
    print("\nProcessed Results:")
    print("-" * 60)
    
    for i, (prompt, item) in enumerate(zip(prompts, result["data"])):
        logits = item["logits"]
        
        # Apply softmax to get probabilities
        probs = softmax(logits)
        
        # For binary classification, you could also use sigmoid
        binary_prob = sigmoid(logits[0]) if len(logits) == 1 else None
        
        print(f"Prompt {i+1}: {prompt}")
        print(f"Raw logits: {logits}")
        print(f"Softmax probabilities: {probs}")
        if binary_prob is not None:
            print(f"Sigmoid probability (if binary): {binary_prob}")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test the vllm classify API")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                        help="URL of the vllm server")
    parser.add_argument("--model", type=str, default="jason9693/Qwen2.5-1.5B-apeach",
                        help="Model name to use for classification")
    args = parser.parse_args()

    # Sample prompts for testing
    prompts = [
        "I love this product, it's amazing!",
        "This is the worst purchase I've ever made.",
        "The product is okay, nothing special.",
        "I'm not sure how I feel about this."
    ]

    test_classify_api(args.server, args.model, prompts)


if __name__ == "__main__":
    main()
