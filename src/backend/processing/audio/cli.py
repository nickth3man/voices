"""
Command-line interface for audio file I/O and management.

This module provides a command-line interface for working with audio files,
including loading, format detection, metadata extraction, waveform generation,
and saving in different formats.
"""

import os
import sys
import logging
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .io import (
    AudioFileIO,
    AudioFormatDetector,
    AudioMetadataExtractor,
    WaveformGenerator,
    load_audio,
    save_audio,
    get_audio_info,
    extract_audio_metadata,
    generate_waveform
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audio File I/O and Management CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get information about an audio file")
    info_parser.add_argument("input", help="Input audio file path")
    info_parser.add_argument("--output", "-o", help="Output JSON file path (if not specified, prints to console)")
    
    # Metadata command
    metadata_parser = subparsers.add_parser("metadata", help="Extract metadata from an audio file")
    metadata_parser.add_argument("input", help="Input audio file path")
    metadata_parser.add_argument("--output", "-o", help="Output JSON file path (if not specified, prints to console)")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert an audio file to a different format")
    convert_parser.add_argument("input", help="Input audio file path")
    convert_parser.add_argument("output", help="Output audio file path")
    convert_parser.add_argument("--format", "-f", choices=AudioFormatDetector.get_supported_formats(),
                               help="Output format (if not specified, inferred from output file extension)")
    convert_parser.add_argument("--sample-rate", "-sr", type=int, default=16000,
                               help="Output sample rate in Hz")
    convert_parser.add_argument("--mono", action="store_true", help="Convert to mono")
    
    # Waveform command
    waveform_parser = subparsers.add_parser("waveform", help="Generate a waveform visualization")
    waveform_parser.add_argument("input", help="Input audio file path")
    waveform_parser.add_argument("--output", "-o", help="Output image file path (PNG, JPG, etc.)")
    waveform_parser.add_argument("--width", "-w", type=int, default=800, help="Width of the waveform in pixels")
    waveform_parser.add_argument("--height", "-h", type=int, default=200, help="Height of the waveform in pixels")
    waveform_parser.add_argument("--points", "-p", type=int, default=1000, 
                                help="Number of points in the waveform data (for data output)")
    waveform_parser.add_argument("--data-only", action="store_true", 
                                help="Output waveform data as JSON instead of an image")
    
    # Formats command
    formats_parser = subparsers.add_parser("formats", help="List supported audio formats")
    
    return parser.parse_args()


def get_file_info(args):
    """Get information about an audio file."""
    logger.info(f"Getting info for file: {args.input}")
    
    try:
        # Get file info
        info = get_audio_info(args.input)
        
        # Format output
        output = json.dumps(info, indent=2)
        
        # Save or print
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            logger.info(f"File info saved to {args.output}")
        else:
            print(output)
            
    except Exception as e:
        logger.error(f"Error getting file info: {str(e)}")
        sys.exit(1)


def extract_metadata(args):
    """Extract metadata from an audio file."""
    logger.info(f"Extracting metadata from file: {args.input}")
    
    try:
        # Extract metadata
        metadata = extract_audio_metadata(args.input)
        
        # Format output
        output = json.dumps(metadata, indent=2)
        
        # Save or print
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            logger.info(f"Metadata saved to {args.output}")
        else:
            print(output)
            
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        sys.exit(1)


def convert_audio(args):
    """Convert an audio file to a different format."""
    logger.info(f"Converting file: {args.input} to {args.output}")
    
    try:
        # Load audio
        result = load_audio(args.input, sr=args.sample_rate if args.sample_rate else None, 
                           mono=args.mono)
        
        # Save in new format
        output_path = save_audio(
            result["audio"],
            args.output,
            result["sample_rate"],
            format=args.format
        )
        
        logger.info(f"Converted audio saved to {output_path}")
            
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        sys.exit(1)


def generate_waveform_visualization(args):
    """Generate a waveform visualization."""
    logger.info(f"Generating waveform for file: {args.input}")
    
    try:
        # Load audio
        result = load_audio(args.input, mono=True)
        audio = result["audio"]
        
        if args.data_only:
            # Generate waveform data
            waveform_data = generate_waveform(audio, args.points)
            
            # Convert to list for JSON serialization
            waveform_list = waveform_data.tolist()
            
            # Format output
            output = json.dumps({
                "waveform": waveform_list,
                "sample_rate": result["sample_rate"],
                "duration": len(audio) / result["sample_rate"]
            }, indent=2)
            
            # Save or print
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                logger.info(f"Waveform data saved to {args.output}")
            else:
                print(output)
                
        else:
            # Generate waveform image
            waveform_gen = WaveformGenerator()
            image = waveform_gen.generate_waveform_image(audio, args.width, args.height)
            
            # Create figure and plot
            plt.figure(figsize=(args.width/100, args.height/100), dpi=100)
            plt.imshow(image, cmap='Blues', aspect='auto')
            plt.axis('off')
            
            # Save or display
            if args.output:
                plt.savefig(args.output, bbox_inches='tight', pad_inches=0)
                logger.info(f"Waveform image saved to {args.output}")
            else:
                plt.show()
                
    except Exception as e:
        logger.error(f"Error generating waveform: {str(e)}")
        sys.exit(1)


def list_formats(args):
    """List supported audio formats."""
    formats = AudioFormatDetector.get_supported_formats()
    extensions = AudioFormatDetector.get_supported_extensions()
    
    print("Supported audio formats:")
    for format_name in formats:
        print(f"  - {format_name}")
    
    print("\nSupported file extensions:")
    print(f"  - {', '.join(extensions)}")


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    if args.command == "info":
        get_file_info(args)
    elif args.command == "metadata":
        extract_metadata(args)
    elif args.command == "convert":
        convert_audio(args)
    elif args.command == "waveform":
        generate_waveform_visualization(args)
    elif args.command == "formats":
        list_formats(args)
    else:
        logger.error("No command specified")
        sys.exit(1)


if __name__ == "__main__":
    main()