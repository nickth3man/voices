"""
Command-line interface for the Demucs model.

This module provides a command-line interface for using the Demucs model,
allowing users to separate voices in audio files from the command line.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

from .model import DemucsModel
from .utils import load_demucs_model, separate_sources, download_pretrained_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Demucs Voice Separation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Separate command
    separate_parser = subparsers.add_parser("separate", help="Separate voices in an audio file")
    separate_parser.add_argument("input", help="Input audio file path")
    separate_parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    separate_parser.add_argument("--model", "-m", help="Path to model file or directory")
    separate_parser.add_argument("--num-speakers", "-n", type=int, default=2, help="Number of speakers to separate")
    separate_parser.add_argument("--sample-rate", "-sr", type=int, default=16000, help="Sample rate for output files")
    separate_parser.add_argument("--no-normalize", action="store_true", help="Disable normalization of input audio")
    separate_parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Device to run the model on")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a pretrained model")
    download_parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    download_parser.add_argument("--model-name", default="demucs_base", help="Model name to download")
    download_parser.add_argument("--force", action="store_true", help="Force download even if model exists")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Display information about a model")
    info_parser.add_argument("--model", "-m", required=True, help="Path to model file or directory")
    
    return parser.parse_args()


def separate_command(args):
    """Execute the separate command."""
    logger.info(f"Separating voices in {args.input}")
    
    # Determine device
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    
    # Load model
    if args.model:
        model = load_demucs_model(args.model, device=device)
    else:
        # Check if default model exists
        default_model_dir = os.path.join(os.path.dirname(__file__), "pretrained")
        default_model_path = os.path.join(default_model_dir, "demucs_base.pth")
        
        if os.path.exists(default_model_path):
            model = load_demucs_model(default_model_path, device=device)
        else:
            # Download default model
            logger.info("No model specified and no default model found. Downloading default model...")
            os.makedirs(default_model_dir, exist_ok=True)
            model_path = download_pretrained_model(default_model_dir, model_name="demucs_base")
            model = load_demucs_model(model_path, device=device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Separate sources
    result = separate_sources(
        model,
        args.input,
        args.output_dir,
        num_speakers=args.num_speakers,
        sample_rate=args.sample_rate,
        normalize=not args.no_normalize
    )
    
    # Print results
    logger.info(f"Separated {len(result)} sources:")
    for source_name in result.keys():
        logger.info(f"  - {source_name}")
    
    logger.info(f"Output files saved to {args.output_dir}")


def download_command(args):
    """Execute the download command."""
    logger.info(f"Downloading model {args.model_name}")
    
    # Download model
    model_path = download_pretrained_model(
        args.output_dir,
        model_name=args.model_name,
        force_download=args.force
    )
    
    logger.info(f"Model downloaded to {model_path}")


def info_command(args):
    """Execute the info command."""
    logger.info(f"Displaying information for model {args.model}")
    
    # Load model
    import torch
    device = torch.device("cpu")  # Use CPU for info command
    model = load_demucs_model(args.model, device=device)
    
    # Get model info
    info = model.get_model_info()
    
    # Print info
    logger.info("Model Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    if args.command == "separate":
        # Import torch here to avoid importing it when not needed
        import torch
        separate_command(args)
    elif args.command == "download":
        download_command(args)
    elif args.command == "info":
        # Import torch here to avoid importing it when not needed
        import torch
        info_command(args)
    else:
        logger.error("No command specified")
        sys.exit(1)


if __name__ == "__main__":
    main()