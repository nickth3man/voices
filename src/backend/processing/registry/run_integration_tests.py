#!/usr/bin/env python3
"""
Run Integration Tests for Voice Separation Components.

This script runs the integration tests for the voice separation components
and generates a report of the results.
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integration_tests.log')
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


def run_tests(output_dir=None, verbose=False):
    """
    Run the integration tests.
    
    Args:
        output_dir: Directory to save test outputs
        verbose: Whether to print verbose output
    
    Returns:
        True if all tests passed, False otherwise
    """
    import unittest
    from integration_tests import IntegrationTests
    
    # Set log level based on verbosity
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTest(IntegrationTests('test_model_registry_integration'))
    suite.addTest(IntegrationTests('test_abstraction_layer_integration'))
    suite.addTest(IntegrationTests('test_pipeline_integration'))
    suite.addTest(IntegrationTests('test_end_to_end_integration'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    # Print summary
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    
    # Return True if all tests passed
    return len(result.errors) == 0 and len(result.failures) == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run integration tests')
    parser.add_argument('--output-dir', '-o', type=str, help='Directory to save test outputs')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print verbose output')
    args = parser.parse_args()
    
    logger.info("Starting integration tests")
    start_time = time.time()
    
    success = run_tests(args.output_dir, args.verbose)
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"Integration tests completed in {duration:.2f} seconds")
    logger.info(f"Result: {'SUCCESS' if success else 'FAILURE'}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()