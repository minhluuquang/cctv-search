"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Play, Pause } from "lucide-react";

// Helper function to format seconds to MM:SS
function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "00:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

// Helper function to format timestamp to HH:MM:SS
function formatTimestamp(date: Date): string {
  const hours = date.getHours().toString().padStart(2, "0");
  const minutes = date.getMinutes().toString().padStart(2, "0");
  const seconds = date.getSeconds().toString().padStart(2, "0");
  return `${hours}:${minutes}:${seconds}`;
}

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface DetectedObject {
  object_id: number;
  label: string;
  confidence: number;
  bbox: BoundingBox;
  center: { x: number; y: number };
}

interface VideoPlayerProps {
  streamUrl: string | null;
  channel: number;
  highlightedObjectId: number | null;
  detectedObjects: DetectedObject[];
  isLoading?: boolean;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  isStreamLoading?: boolean;
  isWaitingForStream?: boolean;
  streamMode?: "live" | "playback";
  onPauseStream?: () => void;
  onResumeStream?: () => void;
  playbackStartTime?: string;
  onSeek?: (newStartTime: string) => void;
  onClearDetectedObjects?: () => void;
}

// Constants
const ONE_DAY_SECONDS = 24 * 60 * 60; // 86400 seconds

export function VideoPlayer({
  streamUrl,
  channel,
  highlightedObjectId,
  detectedObjects,
  isLoading = false,
  videoRef: externalVideoRef,
  isStreamLoading = false,
  isWaitingForStream = false,
  streamMode = "live",
  onPauseStream,
  onResumeStream,
  playbackStartTime,
  onSeek,
  onClearDetectedObjects,
}: VideoPlayerProps) {
  const internalVideoRef = useRef<HTMLVideoElement>(null);
  const videoRef = externalVideoRef || internalVideoRef;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isFrozen, setIsFrozen] = useState(false);
  const [isPlaying, setIsPlaying] = useState(true);
  const [sliderValue, setSliderValue] = useState(0); // 0 to ONE_DAY_SECONDS for playback
  const [isDragging, setIsDragging] = useState(false);
  const progressRef = useRef<HTMLDivElement>(null);
  const videoStartTimeRef = useRef<number>(0);
  const accumulatedSeekRef = useRef<number>(0);
  // Fixed detection resolution - NVR streams and detects at 1920x1080
  const DETECTION_WIDTH = 1920;
  const DETECTION_HEIGHT = 1080;

  // Draw bounding boxes on canvas
  const drawBoundingBoxes = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (detectedObjects.length === 0) return;

    // Get container dimensions
    const containerRect = containerRef.current?.getBoundingClientRect();
    
    if (!containerRect) return;
    
    // Calculate where the video is actually displayed within the container
    // When using object-fit: contain, there might be letterboxing
    const videoAspect = DETECTION_WIDTH / DETECTION_HEIGHT;
    const containerAspect = containerRect.width / containerRect.height;
    
    let displayWidth = containerRect.width;
    let displayHeight = containerRect.height;
    let offsetX = 0;
    let offsetY = 0;
    
    if (containerAspect > videoAspect) {
      // Container is wider - black bars on left/right
      displayWidth = containerRect.height * videoAspect;
      offsetX = (containerRect.width - displayWidth) / 2;
    } else {
      // Container is taller - black bars on top/bottom
      displayHeight = containerRect.width / videoAspect;
      offsetY = (containerRect.height - displayHeight) / 2;
    }
    
    // Calculate scale factors based on actual video display area
    const scaleX = displayWidth / DETECTION_WIDTH;
    const scaleY = displayHeight / DETECTION_HEIGHT;

    detectedObjects.forEach((obj) => {
      const isHighlighted = obj.object_id === highlightedObjectId;
      const { x, y, width, height } = obj.bbox;

      // Set styles based on highlight state
      if (isHighlighted) {
        ctx.strokeStyle = "#3b82f6"; // Blue-500
        ctx.lineWidth = 4;
        ctx.fillStyle = "rgba(59, 130, 246, 0.2)";
      } else {
        ctx.strokeStyle = "#22c55e"; // Green-500
        ctx.lineWidth = 2;
        ctx.fillStyle = "rgba(34, 197, 94, 0.1)";
      }

      // Draw bounding box with offset for letterboxing
      const scaledX = x * scaleX + offsetX;
      const scaledY = y * scaleY + offsetY;
      const scaledW = width * scaleX;
      const scaledH = height * scaleY;

      ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
      ctx.fillRect(scaledX, scaledY, scaledW, scaledH);

      // Draw label
      const label = `${obj.label} #${obj.object_id}`;
      const confidence = `${(obj.confidence * 100).toFixed(0)}%`;
      const text = `${label} ${confidence}`;

      ctx.font = isHighlighted ? "bold 16px system-ui" : "14px system-ui";
      const textMetrics = ctx.measureText(text);
      const textHeight = 20;
      const padding = 4;

      // Label background
      ctx.fillStyle = isHighlighted ? "#3b82f6" : "#22c55e";
      ctx.fillRect(
        scaledX,
        scaledY - textHeight - padding * 2,
        textMetrics.width + padding * 2,
        textHeight + padding * 2
      );

      // Label text
      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, scaledX + padding, scaledY - padding - 2);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detectedObjects, highlightedObjectId]);

  // Update canvas when objects or highlight changes
  useEffect(() => {
    drawBoundingBoxes();
  }, [drawBoundingBoxes]);

  // Get slider max value based on mode
  const getSliderMax = () => {
    return streamMode === 'playback' ? ONE_DAY_SECONDS : 0;
  };

  // Calculate current slider position for playback mode
  // Returns seconds elapsed since playbackStartTime
  const calculateSliderPosition = () => {
    if (streamMode === 'live') return 0;
    
    const video = videoRef.current;
    if (!video || !playbackStartTime) return 0;
    
    // Get the actual timestamp the video is currently showing
    // This is playbackStartTime + video elapsed time
    const videoElapsedTime = video.currentTime;
    const totalElapsedSeconds = videoElapsedTime + accumulatedSeekRef.current;
    
    return Math.min(ONE_DAY_SECONDS, Math.max(0, totalElapsedSeconds));
  };

  // Handle video metadata loaded
  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video) return;
    console.log(`[Video] Metadata loaded: ${video.videoWidth}x${video.videoHeight}`);
    videoStartTimeRef.current = video.currentTime;
  };

  // Reset accumulated seek when stream URL changes (new stream started)
  useEffect(() => {
    accumulatedSeekRef.current = 0;
    // Reset slider to 0 when new stream starts
    if (streamMode === 'playback') {
      setSliderValue(0);
    }
    // Reset frozen state when stream stops (streamUrl becomes null)
    if (!streamUrl) {
      setIsFrozen(false);
    }
  }, [streamUrl, streamMode]);

  // Handle time update
  const handleTimeUpdate = () => {
    if (isDragging) return;
    const video = videoRef.current;
    if (!video) return;
    
    if (streamMode === 'playback') {
      setSliderValue(calculateSliderPosition());
    }
  };

  // Handle play/pause toggle
  const togglePlayPause = () => {
    const video = videoRef.current;
    if (!video) return;

    if (video.paused) {
      video.play();
      setIsPlaying(true);
      // Clear frozen state and detected objects when resuming playback
      if (isFrozen) {
        setIsFrozen(false);
        onClearDetectedObjects?.();
      }
      onResumeStream?.();
    } else {
      video.pause();
      setIsPlaying(false);
      onPauseStream?.();
    }
  };

  // Handle seeking via slider
  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(e.target.value);
    setSliderValue(newValue);
  };

  // Handle seek start (mouse down on slider)
  const handleSeekStart = () => {
    setIsDragging(true);
    const video = videoRef.current;
    if (video && !video.paused) {
      video.pause();
    }
  };

  // Handle seek end (mouse up on slider)
  const handleSeekEnd = () => {
    setIsDragging(false);
    if (streamMode !== 'playback' || !playbackStartTime || !onSeek) return;
    
    // Calculate the new playback start time based on slider position
    // The slider represents seconds from the original playback start time
    const originalStartTime = new Date(playbackStartTime);
    const newStartTime = new Date(originalStartTime.getTime() + sliderValue * 1000);
    
    // Call parent to restart stream with new time
    onSeek(newStartTime.toISOString());
  };

  // Handle play/pause events from video element
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);

    return () => {
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && canvasRef.current && videoRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        canvasRef.current.width = rect.width;
        canvasRef.current.height = rect.height;
        drawBoundingBoxes();
      }
    };

    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawBoundingBoxes]);

  return (
    <div className="flex flex-col h-full">
      {/* Video Container */}
      <div
        ref={containerRef}
        className="relative flex-1 bg-black rounded-lg overflow-hidden min-h-[400px]"
      >
        <video
          ref={videoRef}
          src={streamUrl || undefined}
          autoPlay
          muted
          playsInline
          className="w-full h-full object-contain"
          onLoadedMetadata={handleLoadedMetadata}
          onTimeUpdate={handleTimeUpdate}
          crossOrigin="anonymous"
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 pointer-events-none"
          style={{ width: "100%", height: "100%" }}
        />

        {/* Stream Starting Overlay */}
        {isStreamLoading && (
          <div className="absolute inset-0 bg-black/80 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-muted-foreground">
                Starting stream...
              </span>
            </div>
          </div>
        )}

        {/* Waiting for Stream Overlay */}
        {isWaitingForStream && !isStreamLoading && (
          <div className="absolute inset-0 bg-black/80 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <div className="w-12 h-12 border-4 border-yellow-500 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-muted-foreground">
                Waiting for stream to be ready...
              </span>
              <span className="text-xs text-muted-foreground/60">
                This may take a few seconds
              </span>
            </div>
          </div>
        )}

        {/* Object Detection Loading Overlay */}
        {isLoading && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <div className="w-10 h-10 border-4 border-primary border-t-transparent rounded-full animate-spin" />
              <span className="text-sm text-muted-foreground">
                Detecting objects...
              </span>
            </div>
          </div>
        )}

        {/* Channel Badge */}
        <div className="absolute top-4 left-4 px-3 py-1.5 bg-black/70 rounded-md">
          <span className="text-xs font-medium text-white">
            Channel {channel}
          </span>
        </div>

        {/* Stream Mode Badge */}
        <div className="absolute top-4 left-24 px-3 py-1.5 bg-black/70 rounded-md">
          <span className={`text-xs font-medium ${streamMode === "live" ? "text-red-400" : "text-blue-400"}`}>
            {streamMode === "live" ? "● LIVE" : "▶ PLAYBACK"}
          </span>
        </div>

        {/* Frozen Badge */}
        {isFrozen && (
          <div className="absolute top-4 right-4 px-3 py-1.5 bg-amber-500/90 rounded-md">
            <span className="text-xs font-bold text-white">FROZEN</span>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex flex-col gap-3 mt-4 px-2">
        {/* Progress Bar with Play/Pause */}
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="icon"
            onClick={togglePlayPause}
            disabled={!streamUrl}
            className="h-8 w-8 shrink-0"
          >
            {isPlaying ? (
              <Pause className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
          </Button>
          
          <span className="text-xs text-muted-foreground w-16 text-right font-mono">
            {streamMode === 'playback' && playbackStartTime 
              ? formatTimestamp(new Date(new Date(playbackStartTime).getTime() + sliderValue * 1000))
              : formatTime(0)}
          </span>
          <div className="flex-1 relative" ref={progressRef}>
            <input
              type="range"
              min={0}
              max={getSliderMax()}
              step={1}
              value={streamMode === 'playback' ? sliderValue : 0}
              onChange={handleSeek}
              onMouseDown={handleSeekStart}
              onMouseUp={handleSeekEnd}
              onTouchStart={handleSeekStart}
              onTouchEnd={handleSeekEnd}
              disabled={!streamUrl || streamMode === 'live'}
              className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary hover:accent-primary/80 disabled:opacity-50 disabled:cursor-not-allowed"
              style={{
                background: streamMode === 'playback' 
                  ? `linear-gradient(to right, hsl(var(--primary)) 0%, hsl(var(--primary)) ${(sliderValue / ONE_DAY_SECONDS) * 100}%, hsl(var(--secondary)) ${(sliderValue / ONE_DAY_SECONDS) * 100}%, hsl(var(--secondary)) 100%)`
                  : 'hsl(var(--secondary))'
              }}
            />
          </div>
          <span className="text-xs text-muted-foreground w-16 font-mono">
            {streamMode === 'playback' && playbackStartTime
              ? formatTimestamp(new Date(new Date(playbackStartTime).getTime() + ONE_DAY_SECONDS * 1000))
              : 'LIVE'}
          </span>
        </div>

        {/* Detected Objects Count */}
        {detectedObjects.length > 0 && (
          <div className="flex items-center justify-end">
            <span className="text-sm text-muted-foreground">
              {detectedObjects.length} object
              {detectedObjects.length !== 1 ? "s" : ""} detected
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export type { DetectedObject, BoundingBox };
