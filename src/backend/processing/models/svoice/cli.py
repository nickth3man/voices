"""
Command-line interface for SVoice model.

This module provides a command-line interface for using the SVoice model,
including model loading, inference, and audio processing.
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from processing.models.svoice.model import SVoiceModel
from processing.models.svoice.utils import load_svoice_model, separate_sources, download_pretrained_model


def setup_logger():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('svoice_cli.log')
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='SVoice Model CLI')
    
    # Main commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download a pretrained model')
    download_parser.add_argument('--output-dir', type=str, default='models', help='Directory to save the model')
    download_parser.add_argument('--model-name', type=str, default='svoice_base', help='Name of the model to download')
    download_parser.add_argument('--force', action='store_true', help='Force download even if model exists')
    
    # Separate command
    separate_parser = subparsers.add_parser('separate', help='Separate audio sources')
    separate_parser.add_argument('--model-path', type=str, required=True, help='Path to the model file or directory')
    separate_parser.add_argument('--audio-path', type=str, required=True, help='Path to the audio file')
    separate_parser.add_argument('--output-dir', type=str, default='separated', help='Directory to save separated sources')
    separate_parser.add_argument('--num-speakers', type=int, default=None, help='Number of speakers to separate')
    separate_parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate for loading and saving audio')
    separate_parser.add_argument('--no-normalize', action='store_true', help='Disable audio normalization')
    separate_parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if GPU is available')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get information about a model')
    info_parser.add_argument('--model-path', type=str, required=True, help='Path to the model file or directory')
    info_parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if GPU is available')
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    logger = setup_logger()
    
    # Set device
    if hasattr(args, 'cpu') and args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Execute command
    if args.command == 'download':
        logger.info(f"Downloading pretrained model '{args.model_name}' to {args.output_dir}")
        model_path = download_pretrained_model(
            output_dir=args.output_dir,
            model_name=args.model_name,
            force_download=args.force
        )
        logger.info(f"Model downloaded to {model_path}")
    
    elif args.command == 'separate':
        logger.info(f"Separating sources from {args.audio_path} using model at {args.model_path}")
        
        # Load model
        model = load_svoice_model(args.model_path, device=device)
        
        # Separate sources
        sources = separate_sources(
            model=model,
            audio_path=args.audio_path,
            output_dir=args.output_dir,
            num_speakers=args.num_speakers,
            sample_rate=args.sample_rate,
            normalize=not args.no_normalize
        )
        
        logger.info(f"Separated {len(sources)} sources from {args.audio_path}")
        for source_name in sources:
            logger.info(f"  - {source_name}")
    
    elif args.command == 'info':
        logger.info(f"Getting information about model at {args.model_path}")
        
        # Load model
        model = load_svoice_model(args.model_path, device=device)
        
        # Get model info
        info = model.get_model_info()
        
        # Print info
        logger.info("Model Information:")
        for key, value in info.items():
            logger.info(f"  - {key}: {value}")
    
    else:
        logger.error("No command specified. Use --help for usage information.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())