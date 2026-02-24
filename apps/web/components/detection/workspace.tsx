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
    return now.toISOString();
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
      } catch (error) {
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
  const handleBoxAction = useCallback((objectId: number, actionId: string) => {
    const object = selectedObjects.find(obj => obj.object_id === objectId);
    if (!object) return;

    switch (actionId) {
      case 'search':
        // TODO: Implement search logic
        console.log('Search for', object.label, object.object_id);
        break;
      case 'playback':
        // TODO: Implement -15s playback logic
        console.log('Playback -15s for', object.label, object.object_id);
        break;
      case 'details':
        // Object details shown via UI
        console.log('Details for', object.label, object.object_id);
        break;
      case 'clear':
        setSelectedObjects(prev => prev.filter(obj => obj.object_id !== objectId));
        break;
    }
  }, [selectedObjects]);

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
          videoRef={videoRef}
          isExpanded={isStripExpanded}
          previewObject={hoveredObject}
          selectedObjects={selectedObjects}
          detectedObjects={detectedObjects}
          onBoxAction={handleBoxAction}
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
