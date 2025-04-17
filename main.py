"""
Main entry point for running neural network examples.
"""

import argparse
import sys
import importlib


def main():

    parser = argparse.ArgumentParser(description='Run neural network examples.')
    parser.add_argument('example', type=str, choices=['mnist'], help='Example to run')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--early-stopping', action='store_true', help='Use early stopping')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--lambda-reg', type=float, default=0.0001, help='L2 regularization strength')
    
    args = parser.parse_args()
    
    try:
        example_module = importlib.import_module(f'implementations.{args.example}')
    except ImportError:
        print(f"Error: Could not import example module '{args.example}'")
        sys.exit(1)
    
    if hasattr(example_module, 'main'):
        example_module.main()
    else:
        print(f"Error: Example module '{args.example}' does not have a main function")
        sys.exit(1)


if __name__ == "__main__":
    main()