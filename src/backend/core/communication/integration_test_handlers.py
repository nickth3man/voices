"""
Integration Test Handlers.

This module provides handlers for integration testing commands from the frontend.
"""

import os
import logging
import tempfile
from typing import Dict, Any
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Import integration test modules
try:
    from ...processing.registry.integration_tests import IntegrationTests
    import unittest
except ImportError as e:
    logger.error(f"Error importing integration test modules: {e}")
    raise


def handle_test_model_registry(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle the 'test_model_registry' command.
    
    Args:
        params: Command parameters
    
    Returns:
        Test result
    """
    logger.info("Running model registry integration test")
    
    try:
        # Create test suite
        suite = unittest.TestSuite()
        suite.addTest(IntegrationTests('test_model_registry_integration'))
        
        # Run test
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Check result
        if result.wasSuccessful():
            return {
                'success': True,
                'message': 'Model registry integration test passed'
            }
        else:
            errors = [str(err[1]) for err in result.errors]
            failures = [str(fail[1]) for fail in result.failures]
            return {
                'success': False,
                'error': 'Test failed',
                'errors': errors,
                'failures': failures
            }
    except Exception as e:
        logger.error(f"Error running model registry test: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def handle_test_pipeline(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle the 'test_pipeline' command.
    
    Args:
        params: Command parameters
    
    Returns:
        Test result
    """
    logger.info("Running pipeline integration test")
    
    # Get parameters
    audio_path = params.get('audio_path')
    output_dir = params.get('output_dir', './test_output')
    num_speakers = params.get('num_speakers', 2)
    
    # Validate parameters
    if not audio_path:
        return {
            'success': False,
            'error': 'Missing audio_path parameter'
        }
    
    if not os.path.exists(audio_path):
        return {
            'success': False,
            'error': f'Audio file not found: {audio_path}'
        }
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create test suite
        suite = unittest.TestSuite()
        
        # Create test instance with custom parameters
        test_instance = IntegrationTests()
        test_instance.setUp()
        
        # Override test parameters
        test_instance.test_audio_path = Path(audio_path)
        test_instance.test_dir = Path(output_dir)
        test_instance.num_speakers = num_speakers
        
        # Add test
        suite.addTest(test_instance)
        
        # Run test
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Clean up
        test_instance.tearDown()
        
        # Check result
        if result.wasSuccessful():
            return {
                'success': True,
                'message': 'Pipeline integration test passed',
                'output_dir': output_dir
            }
        else:
            errors = [str(err[1]) for err in result.errors]
            failures = [str(fail[1]) for fail in result.failures]
            return {
                'success': False,
                'error': 'Test failed',
                'errors': errors,
                'failures': failures
            }
    except Exception as e:
        logger.error(f"Error running pipeline test: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def handle_test_end_to_end(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle the 'test_end_to_end' command.
    
    Args:
        params: Command parameters
    
    Returns:
        Test result
    """
    logger.info("Running end-to-end integration test")
    
    # Get parameters
    audio_path = params.get('audio_path')
    output_dir = params.get('output_dir', './test_output')
    num_speakers = params.get('num_speakers', 2)
    
    # Validate parameters
    if not audio_path:
        return {
            'success': False,
            'error': 'Missing audio_path parameter'
        }
    
    if not os.path.exists(audio_path):
        return {
            'success': False,
            'error': f'Audio file not found: {audio_path}'
        }
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create test suite
        suite = unittest.TestSuite()
        suite.addTest(IntegrationTests('test_end_to_end_integration'))
        
        # Create test instance with custom parameters
        test_instance = IntegrationTests()
        test_instance.setUp()
        
        # Override test parameters
        test_instance.test_audio_path = Path(audio_path)
        test_instance.test_dir = Path(output_dir)
        test_instance.num_speakers = num_speakers
        
        # Add test
        suite.addTest(test_instance)
        
        # Run test
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Clean up
        test_instance.tearDown()
        
        # Check result
        if result.wasSuccessful():
            return {
                'success': True,
                'message': 'End-to-end integration test passed',
                'output_dir': output_dir
            }
        else:
            errors = [str(err[1]) for err in result.errors]
            failures = [str(fail[1]) for fail in result.failures]
            return {
                'success': False,
                'error': 'Test failed',
                'errors': errors,
                'failures': failures
            }
    except Exception as e:
        logger.error(f"Error running end-to-end test: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def handle_run_all_tests(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle the 'run_all_tests' command.
    
    Args:
        params: Command parameters
    
    Returns:
        Test results
    """
    logger.info("Running all integration tests")
    
    # Get parameters
    audio_path = params.get('audio_path')
    output_dir = params.get('output_dir', './test_output')
    num_speakers = params.get('num_speakers', 2)
    
    # Validate parameters
    if not audio_path:
        return {
            'success': False,
            'error': 'Missing audio_path parameter'
        }
    
    if not os.path.exists(audio_path):
        return {
            'success': False,
            'error': f'Audio file not found: {audio_path}'
        }
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create test suite
        suite = unittest.TestSuite()
        
        # Create test instance with custom parameters
        test_instance = IntegrationTests()
        test_instance.setUp()
        
        # Override test parameters
        test_instance.test_audio_path = Path(audio_path)
        test_instance.test_dir = Path(output_dir)
        test_instance.num_speakers = num_speakers
        
        # Add all tests
        suite.addTest(IntegrationTests('test_model_registry_integration'))
        suite.addTest(IntegrationTests('test_abstraction_layer_integration'))
        suite.addTest(IntegrationTests('test_pipeline_integration'))
        suite.addTest(IntegrationTests('test_end_to_end_integration'))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Clean up
        test_instance.tearDown()
        
        # Check result
        if result.wasSuccessful():
            return {
                'success': True,
                'message': 'All integration tests passed',
                'output_dir': output_dir,
                'tests_run': result.testsRun
            }
        else:
            errors = [str(err[1]) for err in result.errors]
            failures = [str(fail[1]) for fail in result.failures]
            return {
                'success': False,
                'error': 'Some tests failed',
                'errors': errors,
                'failures': failures,
                'tests_run': result.testsRun,
                'tests_failed': len(result.errors) + len(result.failures)
            }
    except Exception as e:
        logger.error(f"Error running all tests: {e}")
        return {
            'success': False,
            'error': str(e)
        }


# Register handlers
INTEGRATION_TEST_HANDLERS = {
    'test_model_registry': handle_test_model_registry,
    'test_pipeline': handle_test_pipeline,
    'test_end_to_end': handle_test_end_to_end,
    'run_all_tests': handle_run_all_tests
}