"""
Example usage of the Voice Separation Model Abstraction Layer.

This script demonstrates how to use the abstraction layer to separate voices
using different models and with automatic model selection.
"""

import os
import sys
import numpy as np
import soundfile as sf
import argparse
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.abstraction import (
    VoiceSeparationManager,
    AudioCharacteristics,
    ModelType,
    create_separation_manager
)


def main():
    """Run the example."""
    parser = argparse.ArgumentParser(description="Voice Separation Example")
    parser.add_argument("--input", "-i", required=True, help="Input audio file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--speakers", "-s", type=int, default=None, help="Number of speakers")
    parser.add_argument("--model", "-m", choices=["svoice", "demucs", "auto"], default="auto", 
                        help="Model type to use")
    parser.add_argument("--registry", "-r", default=None, help="Model registry directory")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load audio file
    print(f"Loading audio file: {args.input}")
    audio, sample_rate = sf.read(args.input)
    
    # Convert to mono if needed
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        print("Converting stereo audio to mono")
        audio = audio.mean(axis=1)
    
    # Create separation manager
    print(f"Creating separation manager with model type: {args.model}")
    manager = create_separation_manager(
        registry_dir=args.registry,
        default_model_type=args.model
    )
    
    # Extract audio characteristics
    print("Extracting audio characteristics")
    characteristics = AudioCharacteristics.from_audio(audio, sample_rate)
    if args.speakers:
        characteristics.num_speakers = args.speakers
        print(f"Setting number of speakers to: {args.speakers}")
    
    # Select model type based on characteristics
    if args.model == "auto":
        model_type = manager.select_model_type(characteristics)
        print(f"Auto-selected model type: {model_type.value}")
    else:
        model_type = ModelType(args.model)
        print(f"Using specified model type: {model_type.value}")
    
    # Separate sources
    print("Separating sources...")
    sources = manager.separate(
        audio,
        num_speakers=characteristics.num_speakers,
        model_type=model_type,
        sample_rate=sample_rate
    )
    
    # Save separated sources
    base_name = Path(args.input).stem
    for i, source in enumerate(sources):
        output_path = os.path.join(args.output, f"{base_name}_source_{i+1}.wav")
        sf.write(output_path, source, sample_rate)
        print(f"Saved source {i+1} to: {output_path}")
    
    print("Separation complete!")
    
    # Print model information
    model = manager.get_model(model_type=model_type)
    model_info = model.get_model_info()
    print("\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()