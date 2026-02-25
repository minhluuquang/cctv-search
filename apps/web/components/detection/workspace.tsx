"use client";

import { useState, useRef, useCallback } from "react";
import Hls from "hls.js";
import { toast } from "sonner";
import { formatDateToLocalISO } from "@/lib/utils";
import { DetectedObject } from "./types";
import { VideoPlayer } from "./video-player";
import { ObjectStrip } from "./object-strip";
import { ControlBar, StreamMode } from "./control-bar";

interface FrameObjectsResponse {
  timestamp: string;
  channel: number;
  objects: DetectedObject[];
  total_objects: number;
}

interface StreamStartResponse {
  playlist_url: string;
  channel: number;
  message: string;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const CHANNEL = 1;

export function DetectionWorkspace() {
  // State
  const [streamMode, setStreamMode] = useState<StreamMode>("live");
  const [playbackStartTime, setPlaybackStartTime] = useState(() => {
    const now = new Date();
    now.setMinutes(now.getMinutes() - 5);
    // Format as YYYY-MM-DDTHH:MM:SS in local time
    return formatDateToLocalISO(now).replace(/\.\d{3}$/, '');
  });
  
  const [streamUrl, setStreamUrl] = useState<string | null>(null);
  const [isStreamStarted, setIsStreamStarted] = useState(false);
  const [isStreamLoading, setIsStreamLoading] = useState(false);
  const [isWaitingForStream, setIsWaitingForStream] = useState(false);
  const [isObjectDetectionLoading, setIsObjectDetectionLoading] = useState(false);
  const [isFrozen, setIsFrozen] = useState(false);
  
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const [capturedFrameUrl, setCapturedFrameUrl] = useState<string | null>(null);
  const [selectedObjects, setSelectedObjects] = useState<DetectedObject[]>([]);
  const [hoveredObject, setHoveredObject] = useState<DetectedObject | null>(null);
  const [isStripExpanded, setIsStripExpanded] = useState(false);
  const [searchingObjectId, setSearchingObjectId] = useState<number | null>(null);
  
  // Track search results per object (object_id -> first_seen_timestamp)
  const [searchResults, setSearchResults] = useState<Map<number, Date>>(new Map());
  
  // Mini player state
  const [miniPlayerObjectId, setMiniPlayerObjectId] = useState<number | null>(null);
  const [miniPlayerStreamUrl, setMiniPlayerStreamUrl] = useState<string | null>(null);
  const [miniPlayerPlaybackStart, setMiniPlayerPlaybackStart] = useState<Date>(new Date());
  const [isMiniPlayerLoading, setIsMiniPlayerLoading] = useState(false);

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<Hls | null>(null);

  // Helper: Get frame timestamp
  const getFrameTimestamp = useCallback(() => {
    if (streamMode === "live") {
      return new Date();
    }
    // For playback, we'd need more sophisticated tracking
    return new Date(playbackStartTime);
  }, [streamMode, playbackStartTime]);

  // Start stream
  const startStream = async () => {
    setIsStreamLoading(true);
    try {
      const requestBody: { channel: number; start_time?: string } = {
        channel: CHANNEL,
      };

      if (streamMode === "playback" && playbackStartTime) {
        requestBody.start_time = playbackStartTime;
      }

      const response = await fetch(`${API_BASE_URL}/stream/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error("Failed to start stream");
      }

      const data: StreamStartResponse = await response.json();
      const fullUrl = `${API_BASE_URL}${data.playlist_url}`;
      setStreamUrl(fullUrl);
      setIsStreamStarted(true);
      
      // Poll for stream readiness
      pollStreamReady(fullUrl);
    } catch (error) {
      console.error("Failed to start stream:", error);
      toast.error("Failed to start stream");
      setIsStreamLoading(false);
    }
  };

  // Poll for stream ready
  const pollStreamReady = async (url: string) => {
    setIsWaitingForStream(true);
    const maxAttempts = 30;
    let attempts = 0;

    const checkReady = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/stream/ready/${CHANNEL}`);
        const data = await response.json();

        if (data.ready) {
          setIsWaitingForStream(false);
          setIsStreamLoading(false);
          setupHlsPlayer(url);
          return;
        }

        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(checkReady, 1000);
        } else {
          setIsWaitingForStream(false);
          setIsStreamLoading(false);
          toast.error("Stream failed to start", { description: "Timeout waiting for stream" });
        }
      } catch (_error) {
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(checkReady, 1000);
        } else {
          setIsWaitingForStream(false);
          setIsStreamLoading(false);
          toast.error("Stream failed to start");
        }
      }
    };

    checkReady();
  };

  // Setup HLS player
  const setupHlsPlayer = (url: string) => {
    const video = videoRef.current;
    if (!video) return;

    if (Hls.isSupported()) {
      const hls = new Hls({
        enableWorker: true,
        lowLatencyMode: false,
        backBufferLength: 120,
        maxBufferLength: 60,
        maxMaxBufferLength: 120,
      });

      hls.loadSource(url);
      hls.attachMedia(video);

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        video.play().catch(console.error);
      });

      hls.on(Hls.Events.ERROR, (event, data) => {
        if (data.fatal) {
          console.error("Fatal HLS error", data);
        }
      });

      hlsRef.current = hls;
    } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = url;
      video.addEventListener("loadedmetadata", () => {
        video.play().catch(console.error);
      });
    }
  };

  // Stop stream
  const stopStream = async () => {
    try {
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }

      await fetch(`${API_BASE_URL}/stream/stop`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ channel: CHANNEL }),
      });

      setIsStreamStarted(false);
      setIsWaitingForStream(false);
      setStreamUrl(null);
      setIsFrozen(false);
      setDetectedObjects([]);
      setSelectedObjects([]);
      setIsStripExpanded(false);
    } catch (error) {
      console.error("Failed to stop stream:", error);
    }
  };

  // Freeze and detect
  const handleFreezeAndDetect = async () => {
    const video = videoRef.current;
    if (!video) return;

    // Pause video
    video.pause();
    setIsFrozen(true);
    
    // Pause HLS loading
    if (hlsRef.current) {
      hlsRef.current.stopLoad();
    }

    setIsObjectDetectionLoading(true);
    setDetectedObjects([]);
    setSelectedObjects([]);

    try {
      const frameTime = getFrameTimestamp();
      const frameTimestamp = formatDateToLocalISO(frameTime);

      // Capture frame
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || 1920;
      canvas.height = video.videoHeight || 1080;
      const ctx = canvas.getContext("2d");

      if (!ctx) {
        throw new Error("Failed to get canvas context");
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const frameUrl = canvas.toDataURL("image/jpeg", 0.95);
      setCapturedFrameUrl(frameUrl);

      const frameBlob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.95);
      });

      if (!frameBlob) {
        throw new Error("Failed to convert canvas to blob");
      }

      // Send to API
      const formData = new FormData();
      formData.append("timestamp", frameTimestamp);
      formData.append("channel", CHANNEL.toString());
      formData.append("frame_image", frameBlob, "frame.jpg");

      const response = await fetch(`${API_BASE_URL}/frames/objects`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data: FrameObjectsResponse = await response.json();
      setDetectedObjects(data.objects);
      
      // Expand strip to show detected objects
      setIsStripExpanded(true);
    } catch (error) {
      console.error("Failed to detect objects:", error);
      toast.error("Failed to detect objects", {
        description: error instanceof Error ? error.message : "Unknown error occurred",
      });
      setDetectedObjects([]);
    } finally {
      setIsObjectDetectionLoading(false);
    }
  };

  // Toggle object selection (add/remove from selected list)
  const handleToggleObjectSelect = useCallback((object: DetectedObject) => {
    setSelectedObjects(prev => {
      const isAlreadySelected = prev.some(obj => obj.object_id === object.object_id);
      if (isAlreadySelected) {
        return prev.filter(obj => obj.object_id !== object.object_id);
      } else {
        return [...prev, object];
      }
    });
  }, []);

  // Clear all selections
  const handleClearSelection = useCallback(() => {
    setSelectedObjects([]);
  }, []);

  // Handle box action for a specific object
  const handleBoxAction = useCallback(async (objectId: number, actionId: string) => {
    const object = selectedObjects.find(obj => obj.object_id === objectId);
    if (!object) return;

    switch (actionId) {
      case 'search':
        // Set loading state
        setSearchingObjectId(objectId);
        toast.info(`Searching for ${object.label}...`, {
          description: 'This may take a few minutes',
          duration: 5000,
        });

        try {
          // Prepare FormData with object data (from detection)
          const formData = new FormData();
          const timestamp = getFrameTimestamp();
          // Format as YYYY-MM-DDTHH:MM:SS in local time (not UTC)
          const localTimestamp = formatDateToLocalISO(timestamp).replace(/\.\d{3}$/, '');
          formData.append('timestamp', localTimestamp);
          formData.append('channel', CHANNEL.toString());
          formData.append('bbox_x', object.bbox.x.toString());
          formData.append('bbox_y', object.bbox.y.toString());
          formData.append('bbox_width', object.bbox.width.toString());
          formData.append('bbox_height', object.bbox.height.toString());
          formData.append('object_label', object.label);
          formData.append('object_confidence', object.confidence.toString());
          formData.append('search_duration_seconds', '3600');

          // Call API
          const response = await fetch(`${API_BASE_URL}/search/object`, {
            method: 'POST',
            body: formData,
          });

          if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || 'Search failed');
          }

          const data = await response.json();
          
          if (data.status === 'success' && data.result?.found) {
            const firstSeen = new Date(data.result.first_seen_timestamp);
            // Store search result for this object
            setSearchResults(prev => new Map(prev).set(objectId, firstSeen));
            toast.success(`Found ${object.label}!`, {
              description: `First appeared at ${firstSeen.toLocaleTimeString()}. Track duration: ${data.result.track_duration_seconds?.toFixed(1)}s`,
            });
          } else {
            toast.warning('Object not found', {
              description: data.result?.message || 'Could not find object in search window',
            });
          }
        } catch (error) {
          console.error('Search failed:', error);
          toast.error('Search failed', {
            description: error instanceof Error ? error.message : 'Unknown error',
          });
        } finally {
          setSearchingObjectId(null);
        }
        break;
      case 'playback': {
        const firstSeen = searchResults.get(objectId);
        if (!firstSeen) {
          toast.error('Search result not available');
          return;
        }
        
        // Calculate start time (15 seconds before first_seen)
        const playbackStart = new Date(firstSeen.getTime() - 15 * 1000);
        
        // Set mini player playback start time
        setMiniPlayerPlaybackStart(playbackStart);
        
        // Open mini player for this object
        setMiniPlayerObjectId(objectId);
        setIsMiniPlayerLoading(true);
        
        // Start playback stream
        try {
          // Format playback start time in local timezone
          const localPlaybackStart = formatDateToLocalISO(playbackStart).replace(/\.\d{3}$/, '');
          
          const response = await fetch(`${API_BASE_URL}/stream/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              channel: CHANNEL,
              start_time: localPlaybackStart,
            }),
          });
          
          if (!response.ok) {
            throw new Error('Failed to start playback stream');
          }
          
          const data: StreamStartResponse = await response.json();
          setMiniPlayerStreamUrl(`${API_BASE_URL}${data.playlist_url}`);
        } catch (error) {
          console.error('Failed to start mini player:', error);
          toast.error('Failed to start playback');
          setMiniPlayerObjectId(null);
        } finally {
          setIsMiniPlayerLoading(false);
        }
        break;
      }
      case 'details':
        // Object details shown via UI
        console.log('Details for', object.label, object.object_id);
        break;
      case 'clear':
        setSelectedObjects(prev => prev.filter(obj => obj.object_id !== objectId));
        break;
    }
  }, [selectedObjects, getFrameTimestamp, searchResults]);

  // Close mini player
  const handleCloseMiniPlayer = useCallback(() => {
    setMiniPlayerObjectId(null);
    setMiniPlayerStreamUrl(null);
  }, []);

  // Handle seek - restart stream at new time
  const handleSeek = useCallback(async (time: Date) => {
    if (!isStreamStarted) return;

    // Update playback start time
    const formattedTime = formatDateToLocalISO(time).replace(/\.\d{3}$/, '');
    setPlaybackStartTime(formattedTime);

    // Restart stream at new time
    try {
      // Stop current HLS if exists
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }

      const response = await fetch(`${API_BASE_URL}/stream/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          channel: CHANNEL,
          start_time: formattedTime,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to seek stream");
      }

      const data: StreamStartResponse = await response.json();
      const fullUrl = `${API_BASE_URL}${data.playlist_url}`;
      setStreamUrl(fullUrl);
      
      // Setup HLS with new URL
      pollStreamReady(fullUrl);
    } catch (error) {
      console.error("Failed to seek:", error);
      toast.error("Failed to seek");
    }
  }, [isStreamStarted, pollStreamReady]);

  return (
    <div className="fixed inset-0 bg-black flex flex-col">
      {/* Object Strip (Top) */}
      <ObjectStrip
        objects={detectedObjects}
        frameUrl={capturedFrameUrl}
        isExpanded={isStripExpanded}
        selectedObjects={selectedObjects}
        hoveredObject={hoveredObject}
        onToggle={() => setIsStripExpanded(!isStripExpanded)}
        onObjectHover={setHoveredObject}
        onToggleObjectSelect={handleToggleObjectSelect}
        onClearSelection={handleClearSelection}
      />

      {/* Main Video Area - flex-1 to fill remaining space */}
      <div className="flex-1 relative min-h-0">
        <VideoPlayer
          streamUrl={streamUrl}
          channel={CHANNEL}
          isStreamLoading={isStreamLoading}
          isWaitingForStream={isWaitingForStream}
          isObjectDetectionLoading={isObjectDetectionLoading}
          streamMode={streamMode}
          playbackStartTime={new Date(playbackStartTime)}
          videoRef={videoRef}
          isExpanded={isStripExpanded}
          previewObject={hoveredObject}
          selectedObjects={selectedObjects}
          detectedObjects={detectedObjects}
          searchingObjectId={searchingObjectId}
          searchResults={searchResults}
          miniPlayerObjectId={miniPlayerObjectId}
          miniPlayerStreamUrl={miniPlayerStreamUrl}
          miniPlayerPlaybackStart={miniPlayerPlaybackStart}
          isMiniPlayerLoading={isMiniPlayerLoading}
          onBoxAction={handleBoxAction}
          onCloseMiniPlayer={handleCloseMiniPlayer}
          onSeek={handleSeek}
        />
      </div>

      {/* Control Bar (Bottom) */}
      <div className="flex-shrink-0">
        <ControlBar
          streamMode={streamMode}
          onStreamModeChange={setStreamMode}
          playbackStartTime={playbackStartTime}
          onPlaybackStartTimeChange={setPlaybackStartTime}
          isStreamActive={isStreamStarted}
          isStreamLoading={isStreamLoading}
          onStartStream={startStream}
          onStopStream={stopStream}
          onFreezeAndDetect={handleFreezeAndDetect}
          isFrozen={isFrozen}
          isObjectDetectionLoading={isObjectDetectionLoading}
          objectCount={detectedObjects.length}
          isObjectStripExpanded={isStripExpanded}
          onToggleObjectStrip={() => setIsStripExpanded(!isStripExpanded)}
        />
      </div>
    </div>
  );
}
