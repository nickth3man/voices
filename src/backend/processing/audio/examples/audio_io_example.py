"""
Example script demonstrating the audio file I/O and management functionality.

This script shows how to use the audio I/O module to:
1. Load an audio file
2. Extract metadata
3. Generate a waveform visualization
4. Convert to a different format
5. Save the processed audio
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to allow importing the audio module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.backend.processing.audio.io import (
    AudioFileIO,
    load_audio,
    save_audio,
    extract_audio_metadata,
    generate_waveform
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audio I/O Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--format", "-f", default="wav", help="Output format")
    parser.add_argument("--sample-rate", "-sr", type=int, default=16000, help="Output sample rate")
    
    return parser.parse_args()


def main():
    """Main function demonstrating audio I/O functionality."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing audio file: {args.input}")
    
    # 1. Load audio file
    print("Loading audio file...")
    result = load_audio(args.input, sr=args.sample_rate)
    
    audio = result["audio"]
    sample_rate = result["sample_rate"]
    metadata = result["metadata"]
    
    print(f"Loaded audio: {len(audio)} samples, {sample_rate}Hz, {metadata['duration']:.2f}s")
    
    # 2. Print metadata
    print("\nMetadata:")
    for key, value in metadata.items():
        if isinstance(value, (int, float, str, bool)):
            print(f"  {key}: {value}")
    
    # 3. Generate and save waveform visualization
    print("\nGenerating waveform visualization...")
    waveform = generate_waveform(audio, n_points=1000)
    
    plt.figure(figsize=(10, 3))
    plt.plot(waveform)
    plt.title("Waveform Visualization")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    waveform_path = output_dir / "waveform.png"
    plt.savefig(waveform_path)
    print(f"Waveform saved to: {waveform_path}")
    
    # 4. Apply some processing (simple gain adjustment)
    print("\nApplying gain adjustment...")
    processed_audio = audio * 0.8  # Reduce volume by 20%
    
    # 5. Save processed audio in the specified format
    print(f"Saving processed audio in {args.format} format...")
    output_path = output_dir / f"processed.{args.format}"
    saved_path = save_audio(processed_audio, output_path, sample_rate, format=args.format)
    print(f"Processed audio saved to: {saved_path}")
    
    # 6. Extract metadata from the processed file
    print("\nExtracting metadata from processed file...")
    processed_metadata = extract_audio_metadata(saved_path)
    
    print("Processed file metadata:")
    for key, value in processed_metadata.items():
        if isinstance(value, (int, float, str, bool)):
            print(f"  {key}: {value}")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()