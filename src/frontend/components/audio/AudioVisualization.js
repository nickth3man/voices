/**
 * Audio Visualization component for the Voices application
 * 
 * This component provides advanced audio visualization using Wavesurfer.js,
 * including multi-track display, waveform coloring by speaker, zoom and
 * navigation controls, and segment markers for speaker transitions.
 */

import React, { useState, useEffect, useRef } from 'react';
import pythonBridge from '../../controllers/PythonBridge';
import WaveSurfer from 'wavesurfer.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.js';
import TimelinePlugin from 'wavesurfer.js/dist/plugins/timeline.js';
import ZoomPlugin from 'wavesurfer.js/dist/plugins/zoom.js';

const AudioVisualization = () => {
  // State for audio files and tracks
  const [audioFile, setAudioFile] = useState(null);
  const [audioPath, setAudioPath] = useState('');
  const [tracks, setTracks] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [speakers, setSpeakers] = useState([]);
  const [currentSpeaker, setCurrentSpeaker] = useState(null);
  
  // State for wavesurfer instances
  const [wavesurfers, setWavesurfers] = useState([]);
  const [mainWavesurfer, setMainWavesurfer] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(50);
  
  // Refs for wavesurfer containers
  const mainWaveformRef = useRef(null);
  const trackWaveformRefs = useRef([]);
  
  // Initialize main wavesurfer instance
  useEffect(() => {
    if (mainWaveformRef.current) {
      const wavesurfer = WaveSurfer.create({
        container: mainWaveformRef.current,
        waveColor: 'rgb(200, 200, 200)',
        progressColor: 'rgb(52, 152, 219)',
        height: 120,
        normalize: true,
        minimap: true,
        backend: 'WebAudio',
        plugins: [
          RegionsPlugin.create(),
          TimelinePlugin.create({
            container: '#timeline',
            formatTimeCallback: formatTime,
            timeInterval: 0.5,
            primaryLabelInterval: 5,
            secondaryLabelInterval: 1
          }),
          ZoomPlugin.create({
            // Zoom options
          })
        ]
      });
      
      setMainWavesurfer(wavesurfer);
      
      // Clean up on unmount
      return () => {
        wavesurfer.destroy();
      };
    }
  }, []);
  
  // Handle zoom level change
  useEffect(() => {
    if (mainWavesurfer) {
      mainWavesurfer.zoom(zoomLevel);
    }
  }, [zoomLevel, mainWavesurfer]);
  
  // Load audio file into main wavesurfer
  useEffect(() => {
    if (mainWavesurfer && audioPath) {
      setIsLoading(true);
      
      // Load audio file
      mainWavesurfer.load(`file://${audioPath}`);
      
      // Handle events
      mainWavesurfer.on('ready', () => {
        setIsLoading(false);
        fetchSpeakerSegments();
      });
      
      mainWavesurfer.on('error', (err) => {
        setIsLoading(false);
        setError(`Error loading audio: ${err}`);
      });
    }
  }, [mainWavesurfer, audioPath]);
  
  // Initialize track wavesurfer instances
  useEffect(() => {
    // Clean up previous instances
    wavesurfers.forEach(ws => ws.destroy());
    
    if (tracks.length > 0) {
      const newWavesurfers = tracks.map((track, index) => {
        if (trackWaveformRefs.current[index]) {
          const wavesurfer = WaveSurfer.create({
            container: trackWaveformRefs.current[index],
            waveColor: getSpeakerColor(track.speaker),
            progressColor: getProgressColor(track.speaker),
            height: 80,
            normalize: true,
            backend: 'WebAudio'
          });
          
          // Load track audio
          wavesurfer.load(`file://${track.path}`);
          
          return wavesurfer;
        }
        return null;
      }).filter(Boolean);
      
      setWavesurfers(newWavesurfers);
      
      // Clean up on unmount
      return () => {
        newWavesurfers.forEach(ws => ws.destroy());
      };
    }
  }, [tracks]);
  
  // Handle file selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setAudioFile(file);
      setAudioPath(file.path);
      setTracks([]);
      setSpeakers([]);
      setCurrentSpeaker(null);
    }
  };
  
  // Fetch speaker segments from backend
  const fetchSpeakerSegments = async () => {
    if (!audioPath) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await pythonBridge.sendRequest('get_speaker_segments', {
        audioPath
      });
      
      if (response.success) {
        const { segments, speakers } = response.data;
        
        // Add regions to main wavesurfer
        if (mainWavesurfer) {
          // Clear existing regions
          mainWavesurfer.regions.clear();
          
          // Add new regions
          segments.forEach(segment => {
            mainWavesurfer.regions.add({
              start: segment.start,
              end: segment.end,
              color: getSpeakerColor(segment.speaker, 0.2),
              drag: false,
              resize: false,
              attributes: {
                speaker: segment.speaker
              }
            });
          });
        }
        
        setSpeakers(speakers);
      } else {
        setError(response.error || 'Failed to fetch speaker segments');
      }
    } catch (err) {
      setError(`Error fetching speaker segments: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Separate tracks by speaker
  const separateTracksBySpeaker = async () => {
    if (!audioPath) {
      setError('Please select an audio file');
      return;
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await pythonBridge.sendRequest('separate_speakers', {
        audioPath
      });
      
      if (response.success) {
        setTracks(response.data.tracks);
      } else {
        setError(response.error || 'Failed to separate speakers');
      }
    } catch (err) {
      setError(`Error separating speakers: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Play/pause main audio
  const togglePlayPause = () => {
    if (mainWavesurfer) {
      mainWavesurfer.playPause();
    }
  };
  
  // Play specific speaker track
  const playSpeakerTrack = (speakerId) => {
    // Pause all wavesurfers
    wavesurfers.forEach(ws => ws.pause());
    
    // Find and play the selected speaker track
    const trackIndex = tracks.findIndex(track => track.speaker === speakerId);
    if (trackIndex >= 0 && wavesurfers[trackIndex]) {
      wavesurfers[trackIndex].play();
      setCurrentSpeaker(speakerId);
    }
  };
  
  // Stop all playback
  const stopAllPlayback = () => {
    if (mainWavesurfer) {
      mainWavesurfer.pause();
    }
    
    wavesurfers.forEach(ws => ws.pause());
    setCurrentSpeaker(null);
  };
  
  // Handle zoom change
  const handleZoomChange = (e) => {
    setZoomLevel(parseInt(e.target.value, 10));
  };
  
  // Format time for timeline (convert seconds to MM:SS format)
  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };
  
  // Get color for speaker waveform
  const getSpeakerColor = (speakerId, alpha = 1) => {
    const colors = [
      `rgba(52, 152, 219, ${alpha})`,  // Blue
      `rgba(46, 204, 113, ${alpha})`,  // Green
      `rgba(155, 89, 182, ${alpha})`,  // Purple
      `rgba(230, 126, 34, ${alpha})`,  // Orange
      `rgba(231, 76, 60, ${alpha})`,   // Red
      `rgba(241, 196, 15, ${alpha})`,  // Yellow
    ];
    
    // Use consistent color for each speaker
    const index = speakers.findIndex(s => s.id === speakerId);
    return index >= 0 ? colors[index % colors.length] : `rgba(200, 200, 200, ${alpha})`;
  };
  
  // Get progress color for speaker waveform
  const getProgressColor = (speakerId) => {
    const colors = [
      'rgba(41, 128, 185, 1)',  // Darker Blue
      'rgba(39, 174, 96, 1)',   // Darker Green
      'rgba(142, 68, 173, 1)',  // Darker Purple
      'rgba(211, 84, 0, 1)',    // Darker Orange
      'rgba(192, 57, 43, 1)',   // Darker Red
      'rgba(243, 156, 18, 1)',  // Darker Yellow
    ];
    
    // Use consistent color for each speaker
    const index = speakers.findIndex(s => s.id === speakerId);
    return index >= 0 ? colors[index % colors.length] : 'rgba(150, 150, 150, 1)';
  };
  
  return (
    <div className="audio-visualization-container">
      <h2>Enhanced Audio Visualization</h2>
      <p>Visualize audio with multi-track display, speaker coloring, and navigation controls.</p>
      
      <div className="audio-selection">
        <h3>Select Audio File</h3>
        <input 
          type="file" 
          accept="audio/*" 
          onChange={handleFileSelect} 
          disabled={isLoading}
        />
        {audioFile && (
          <p>Selected: {audioFile.name}</p>
        )}
      </div>
      
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}
      
      <div className="main-waveform-container">
        <h3>Main Waveform</h3>
        <div className="waveform-controls">
          <button 
            onClick={togglePlayPause}
            disabled={!audioPath || isLoading}
          >
            Play/Pause
          </button>
          <button 
            onClick={stopAllPlayback}
            disabled={!audioPath || isLoading}
          >
            Stop
          </button>
          <button 
            onClick={separateTracksBySpeaker}
            disabled={!audioPath || isLoading}
          >
            Separate Speakers
          </button>
          <div className="zoom-control">
            <label htmlFor="zoom">Zoom: </label>
            <input 
              id="zoom"
              type="range" 
              min="10" 
              max="500" 
              value={zoomLevel} 
              onChange={handleZoomChange}
              disabled={!audioPath || isLoading}
            />
          </div>
        </div>
        
        <div id="timeline"></div>
        <div ref={mainWaveformRef} className="waveform"></div>
        
        {isLoading && (
          <div className="loading-indicator">
            Loading...
          </div>
        )}
      </div>
      
      {speakers.length > 0 && (
        <div className="speaker-legend">
          <h3>Speakers</h3>
          <div className="speaker-list">
            {speakers.map((speaker, index) => (
              <div 
                key={speaker.id} 
                className="speaker-item"
                style={{ 
                  backgroundColor: getSpeakerColor(speaker.id, 0.2),
                  borderLeft: `4px solid ${getSpeakerColor(speaker.id)}`
                }}
              >
                <span className="speaker-name">Speaker {index + 1}</span>
                <button 
                  onClick={() => playSpeakerTrack(speaker.id)}
                  className={currentSpeaker === speaker.id ? 'playing' : ''}
                  disabled={!tracks.some(t => t.speaker === speaker.id)}
                >
                  {currentSpeaker === speaker.id ? 'Playing...' : 'Play'}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {tracks.length > 0 && (
        <div className="speaker-tracks">
          <h3>Speaker Tracks</h3>
          {tracks.map((track, index) => (
            <div key={index} className="track-container">
              <div className="track-header">
                <span className="track-title">
                  Speaker {speakers.findIndex(s => s.id === track.speaker) + 1}
                </span>
                <button 
                  onClick={() => playSpeakerTrack(track.speaker)}
                  className={currentSpeaker === track.speaker ? 'playing' : ''}
                >
                  {currentSpeaker === track.speaker ? 'Playing...' : 'Play'}
                </button>
              </div>
              <div 
                ref={el => trackWaveformRefs.current[index] = el} 
                className="track-waveform"
              ></div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AudioVisualization;