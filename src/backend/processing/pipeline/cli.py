"""
Command-line interface for the audio processing pipeline.

This module provides a command-line interface for using the audio processing pipeline,
allowing users to process audio files and configure the pipeline from the command line.
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .components import AudioProcessingPipeline
from .utils import (
    create_pipeline_from_config,
    load_pipeline_config,
    save_pipeline_config,
    create_default_pipeline_config,
    process_directory
)
from ..models.abstraction import ModelType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audio Processing Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process a single file
    process_parser = subparsers.add_parser("process", help="Process an audio file")
    process_parser.add_argument("input", help="Input audio file path")
    process_parser.add_argument("--output-dir", "-o", help="Output directory")
    process_parser.add_argument("--config", "-c", help="Pipeline configuration file")
    process_parser.add_argument("--model-type", choices=["svoice", "demucs", "auto"], default="auto", help="Voice separation model type")
    process_parser.add_argument("--num-speakers", "-n", type=int, help="Number of speakers to separate")
    process_parser.add_argument("--apply-denoising", action="store_true", help="Apply denoising to separated sources")
    process_parser.add_argument("--no-normalization", action="store_true", help="Disable normalization of separated sources")
    process_parser.add_argument("--chunk-size", type=int, help="Chunk size for processing (in samples)")
    process_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks (in samples)")
    
    # Process a directory of files
    batch_parser = subparsers.add_parser("batch", help="Process a directory of audio files")
    batch_parser.add_argument("input_dir", help="Input directory containing audio files")
    batch_parser.add_argument("--output-dir", "-o", help="Output directory")
    batch_parser.add_argument("--config", "-c", help="Pipeline configuration file")
    batch_parser.add_argument("--file-pattern", default="*.wav", help="File pattern to match audio files")
    batch_parser.add_argument("--model-type", choices=["svoice", "demucs", "auto"], default="auto", help="Voice separation model type")
    batch_parser.add_argument("--num-speakers", "-n", type=int, help="Number of speakers to separate")
    batch_parser.add_argument("--apply-denoising", action="store_true", help="Apply denoising to separated sources")
    batch_parser.add_argument("--no-normalization", action="store_true", help="Disable normalization of separated sources")
    batch_parser.add_argument("--chunk-size", type=int, help="Chunk size for processing (in samples)")
    batch_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks (in samples)")
    
    # Create a default configuration file
    config_parser = subparsers.add_parser("create-config", help="Create a default configuration file")
    config_parser.add_argument("output", help="Output configuration file path")
    
    # Compare models
    compare_parser = subparsers.add_parser("compare", help="Compare separation models on an audio file")
    compare_parser.add_argument("input", help="Input audio file path")
    compare_parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    compare_parser.add_argument("--num-speakers", "-n", type=int, help="Number of speakers to separate")
    
    return parser.parse_args()


def process_file(args):
    """Process a single audio file."""
    logger.info(f"Processing file: {args.input}")
    
    # Load configuration if provided
    if args.config:
        config = load_pipeline_config(args.config)
    else:
        config = create_default_pipeline_config()
    
    # Override configuration with command-line arguments
    if args.model_type:
        config["separator"]["model_type"] = args.model_type
    if args.num_speakers is not None:
        config["separator"]["num_speakers"] = args.num_speakers
    if args.apply_denoising:
        config["postprocessor"]["apply_denoising"] = True
    if args.no_normalization:
        config["postprocessor"]["apply_normalization"] = False
    if args.chunk_size is not None:
        config["preprocessor"]["chunk_size"] = args.chunk_size
    if args.overlap is not None:
        config["preprocessor"]["overlap"] = args.overlap
    if args.output_dir:
        config["formatter"]["output_dir"] = args.output_dir
    
    # Create pipeline
    pipeline = create_pipeline_from_config(config)
    
    # Process file
    result = pipeline.process_file(args.input)
    
    # Print results
    if "output_paths" in result:
        logger.info("Processing completed successfully")
        logger.info("Output files:")
        for source_name, path in result["output_paths"].items():
            logger.info(f"  {source_name}: {path}")
    else:
        logger.info("Processing completed, but no output files were saved")


def batch_process(args):
    """Process a directory of audio files."""
    logger.info(f"Processing directory: {args.input_dir}")
    
    # Load configuration if provided
    if args.config:
        config = load_pipeline_config(args.config)
    else:
        config = create_default_pipeline_config()
    
    # Override configuration with command-line arguments
    if args.model_type:
        config["separator"]["model_type"] = args.model_type
    if args.num_speakers is not None:
        config["separator"]["num_speakers"] = args.num_speakers
    if args.apply_denoising:
        config["postprocessor"]["apply_denoising"] = True
    if args.no_normalization:
        config["postprocessor"]["apply_normalization"] = False
    if args.chunk_size is not None:
        config["preprocessor"]["chunk_size"] = args.chunk_size
    if args.overlap is not None:
        config["preprocessor"]["overlap"] = args.overlap
    
    # Create pipeline
    pipeline = create_pipeline_from_config(config)
    
    # Process directory
    results = process_directory(
        pipeline,
        args.input_dir,
        args.output_dir,
        args.file_pattern
    )
    
    # Print results
    logger.info(f"Processed {len(results)} files")
    success_count = sum(1 for result in results.values() if result["status"] == "success")
    error_count = sum(1 for result in results.values() if result["status"] == "error")
    logger.info(f"Success: {success_count}, Errors: {error_count}")
    
    if error_count > 0:
        logger.info("Files with errors:")
        for file_name, result in results.items():
            if result["status"] == "error":
                logger.info(f"  {file_name}: {result['error']}")


def create_config(args):
    """Create a default configuration file."""
    logger.info(f"Creating default configuration file: {args.output}")
    
    # Create default configuration
    config = create_default_pipeline_config()
    
    # Save configuration
    save_pipeline_config(config, args.output)
    
    logger.info("Configuration file created successfully")


def compare_models(args):
    """Compare separation models on an audio file."""
    logger.info(f"Comparing models on file: {args.input}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create configurations for each model
    svoice_config = create_default_pipeline_config()
    svoice_config["separator"]["model_type"] = "svoice"
    svoice_config["formatter"]["output_dir"] = os.path.join(args.output_dir, "svoice")
    
    demucs_config = create_default_pipeline_config()
    demucs_config["separator"]["model_type"] = "demucs"
    demucs_config["formatter"]["output_dir"] = os.path.join(args.output_dir, "demucs")
    
    # Set number of speakers if provided
    if args.num_speakers is not None:
        svoice_config["separator"]["num_speakers"] = args.num_speakers
        demucs_config["separator"]["num_speakers"] = args.num_speakers
    
    # Create pipelines
    svoice_pipeline = create_pipeline_from_config(svoice_config)
    demucs_pipeline = create_pipeline_from_config(demucs_config)
    
    # Process with SVoice
    logger.info("Processing with SVoice model")
    svoice_result = svoice_pipeline.process_file(args.input)
    
    # Process with Demucs
    logger.info("Processing with Demucs model")
    demucs_result = demucs_pipeline.process_file(args.input)
    
    # Print results
    logger.info("Comparison completed")
    logger.info("SVoice output files:")
    if "output_paths" in svoice_result:
        for source_name, path in svoice_result["output_paths"].items():
            logger.info(f"  {source_name}: {path}")
    
    logger.info("Demucs output files:")
    if "output_paths" in demucs_result:
        for source_name, path in demucs_result["output_paths"].items():
            logger.info(f"  {source_name}: {path}")
    
    # Save comparison report
    report = {
        "input_file": args.input,
        "svoice": {
            "model_info": svoice_result.get("metadata", {}).get("separation", {}),
            "output_files": svoice_result.get("output_paths", {})
        },
        "demucs": {
            "model_info": demucs_result.get("metadata", {}).get("separation", {}),
            "output_files": demucs_result.get("output_paths", {})
        }
    }
    
    report_path = os.path.join(args.output_dir, "comparison_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Comparison report saved to {report_path}")


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    if args.command == "process":
        process_file(args)
    elif args.command == "batch":
        batch_process(args)
    elif args.command == "create-config":
        create_config(args)
    elif args.command == "compare":
        compare_models(args)
    else:
        logger.error("No command specified")
        sys.exit(1)


if __name__ == "__main__":
    main()