"""
Standardized test datasets for evaluating voice separation models.

This module provides functionality for creating, managing, and accessing
standardized test datasets used in the evaluation of voice separation models.
"""

import os
import json
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import yaml
import torch
from dataclasses import dataclass, field


@dataclass
class TestDatasetItem:
    """A single item in a test dataset."""
    
    id: str
    mixture_path: str
    source_paths: List[str]
    num_speakers: int
    duration: float
    sample_rate: int
    difficulty: str  # 'easy', 'medium', 'hard'
    environment: str  # 'clean', 'noisy', 'reverberant', etc.
    metadata: Dict = field(default_factory=dict)
    
    def load_audio(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Load the mixture and source audio files.
        
        Returns:
            Tuple containing:
                - mixture: np.ndarray of shape (n_samples,)
                - sources: List of np.ndarray, each of shape (n_samples,)
        """
        mixture, _ = librosa.load(self.mixture_path, sr=self.sample_rate, mono=True)
        sources = []
        
        for source_path in self.source_paths:
            source, _ = librosa.load(source_path, sr=self.sample_rate, mono=True)
            # Ensure same length as mixture
            if len(source) < len(mixture):
                source = np.pad(source, (0, len(mixture) - len(source)))
            elif len(source) > len(mixture):
                source = source[:len(mixture)]
            sources.append(source)
        
        return mixture, sources
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "mixture_path": self.mixture_path,
            "source_paths": self.source_paths,
            "num_speakers": self.num_speakers,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "difficulty": self.difficulty,
            "environment": self.environment,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TestDatasetItem':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            mixture_path=data["mixture_path"],
            source_paths=data["source_paths"],
            num_speakers=data["num_speakers"],
            duration=data["duration"],
            sample_rate=data["sample_rate"],
            difficulty=data["difficulty"],
            environment=data["environment"],
            metadata=data.get("metadata", {})
        )


@dataclass
class TestDataset:
    """A collection of test data for evaluating voice separation models."""
    
    name: str
    description: str
    items: List[TestDatasetItem] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def add_item(self, item: TestDatasetItem) -> None:
        """Add an item to the dataset."""
        self.items.append(item)
    
    def get_item(self, item_id: str) -> Optional[TestDatasetItem]:
        """Get an item by ID."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None
    
    def filter_items(
        self, 
        num_speakers: Optional[int] = None,
        difficulty: Optional[str] = None,
        environment: Optional[str] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None
    ) -> List[TestDatasetItem]:
        """Filter items based on criteria."""
        filtered_items = self.items
        
        if num_speakers is not None:
            filtered_items = [item for item in filtered_items if item.num_speakers == num_speakers]
        
        if difficulty is not None:
            filtered_items = [item for item in filtered_items if item.difficulty == difficulty]
        
        if environment is not None:
            filtered_items = [item for item in filtered_items if item.environment == environment]
        
        if min_duration is not None:
            filtered_items = [item for item in filtered_items if item.duration >= min_duration]
        
        if max_duration is not None:
            filtered_items = [item for item in filtered_items if item.duration <= max_duration]
        
        return filtered_items
    
    def save(self, path: str) -> None:
        """Save the dataset to a file."""
        data = {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "items": [item.to_dict() for item in self.items]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TestDataset':
        """Load a dataset from a file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        dataset = cls(
            name=data["name"],
            description=data["description"],
            metadata=data.get("metadata", {})
        )
        
        for item_data in data["items"]:
            dataset.add_item(TestDatasetItem.from_dict(item_data))
        
        return dataset


def create_test_dataset(
    name: str,
    description: str,
    source_dir: str,
    output_dir: str,
    num_speakers_range: Tuple[int, int] = (2, 5),
    duration_range: Tuple[float, float] = (5.0, 30.0),
    sample_rate: int = 16000,
    num_items_per_category: int = 5,
    difficulties: List[str] = ["easy", "medium", "hard"],
    environments: List[str] = ["clean", "noisy", "reverberant"]
) -> TestDataset:
    """
    Create a standardized test dataset from source audio files.
    
    This function creates a test dataset by mixing source audio files
    in different combinations to create test scenarios with varying
    difficulty levels and acoustic environments.
    
    Args:
        name: Name of the dataset
        description: Description of the dataset
        source_dir: Directory containing source audio files
        output_dir: Directory to save the mixed audio files
        num_speakers_range: Range of number of speakers to include in mixtures
        duration_range: Range of durations for the mixtures
        sample_rate: Sample rate for the audio files
        num_items_per_category: Number of items to create per category
        difficulties: List of difficulty levels to include
        environments: List of acoustic environments to simulate
    
    Returns:
        TestDataset object containing the created test items
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset object
    dataset = TestDataset(name=name, description=description)
    
    # Find all audio files in source directory
    source_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                source_files.append(os.path.join(root, file))
    
    if not source_files:
        raise ValueError(f"No audio files found in {source_dir}")
    
    # Create items for each category
    item_id = 1
    
    for num_speakers in range(num_speakers_range[0], num_speakers_range[1] + 1):
        for difficulty in difficulties:
            for environment in environments:
                for _ in range(num_items_per_category):
                    # Create a unique ID for this item
                    item_id_str = f"test_{num_speakers}spk_{difficulty}_{environment}_{item_id:03d}"
                    
                    # Create subdirectory for this item
                    item_dir = os.path.join(output_dir, item_id_str)
                    os.makedirs(item_dir, exist_ok=True)
                    
                    # Select random source files
                    selected_sources = np.random.choice(source_files, num_speakers, replace=False)
                    
                    # Load and process source files
                    processed_sources = []
                    for i, source_file in enumerate(selected_sources):
                        # Load audio
                        audio, _ = librosa.load(source_file, sr=sample_rate, mono=True)
                        
                        # Trim to desired duration
                        target_duration = np.random.uniform(duration_range[0], duration_range[1])
                        target_samples = int(target_duration * sample_rate)
                        
                        if len(audio) > target_samples:
                            # Randomly select a segment
                            start = np.random.randint(0, len(audio) - target_samples)
                            audio = audio[start:start + target_samples]
                        else:
                            # Pad if too short
                            audio = np.pad(audio, (0, max(0, target_samples - len(audio))))
                        
                        # Apply difficulty adjustments
                        if difficulty == "medium":
                            # Add some background noise
                            noise = np.random.normal(0, 0.01, len(audio))
                            audio = audio + noise
                        elif difficulty == "hard":
                            # Add more noise and some distortion
                            noise = np.random.normal(0, 0.02, len(audio))
                            audio = audio + noise
                            # Simple distortion
                            audio = np.tanh(audio * 2) / 2
                        
                        # Apply environment effects
                        if environment == "noisy":
                            # Add environmental noise
                            noise_level = 0.05 if difficulty == "easy" else 0.1
                            noise = np.random.normal(0, noise_level, len(audio))
                            audio = audio + noise
                        elif environment == "reverberant":
                            # Simple convolution reverb simulation
                            reverb_length = int(0.1 * sample_rate)  # 100ms reverb
                            reverb = np.exp(-np.arange(reverb_length) / (sample_rate * 0.05))
                            audio = np.convolve(audio, reverb, mode='same')
                        
                        # Normalize
                        audio = audio / (np.max(np.abs(audio)) + 1e-8)
                        
                        # Save processed source
                        source_path = os.path.join(item_dir, f"source_{i+1}.wav")
                        sf.write(source_path, audio, sample_rate)
                        processed_sources.append((audio, source_path))
                    
                    # Create mixture
                    mixture = np.zeros(target_samples)
                    for audio, _ in processed_sources:
                        mixture += audio
                    
                    # Normalize mixture
                    mixture = mixture / (np.max(np.abs(mixture)) + 1e-8)
                    
                    # Save mixture
                    mixture_path = os.path.join(item_dir, "mixture.wav")
                    sf.write(mixture_path, mixture, sample_rate)
                    
                    # Create dataset item
                    item = TestDatasetItem(
                        id=item_id_str,
                        mixture_path=mixture_path,
                        source_paths=[path for _, path in processed_sources],
                        num_speakers=num_speakers,
                        duration=target_duration,
                        sample_rate=sample_rate,
                        difficulty=difficulty,
                        environment=environment,
                        metadata={
                            "source_files": [os.path.basename(src) for src in selected_sources]
                        }
                    )
                    
                    dataset.add_item(item)
                    item_id += 1
    
    # Save dataset metadata
    dataset_path = os.path.join(output_dir, f"{name}_metadata.json")
    dataset.save(dataset_path)
    
    return dataset