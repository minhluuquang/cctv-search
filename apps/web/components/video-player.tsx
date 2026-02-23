"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Scan, Play, Pause } from "lucide-react";

// Helper function to format seconds to MM:SS
function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "00:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
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
  onFrameFreeze: (videoElement: HTMLVideoElement | null) => void;
  highlightedObjectId: number | null;
  detectedObjects: DetectedObject[];
  isLoading?: boolean;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
  isStreamLoading?: boolean;
  isWaitingForStream?: boolean;
  streamMode?: "live" | "playback";
  onPauseStream?: () => void;
  onResumeStream?: () => void;
}

export function VideoPlayer({
  streamUrl,
  channel,
  onFrameFreeze,
  highlightedObjectId,
  detectedObjects,
  isLoading = false,
  videoRef: externalVideoRef,
  isStreamLoading = false,
  isWaitingForStream = false,
  streamMode = "live",
  onPauseStream,
  onResumeStream,
}: VideoPlayerProps) {
  const internalVideoRef = useRef<HTMLVideoElement>(null);
  const videoRef = externalVideoRef || internalVideoRef;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isFrozen, setIsFrozen] = useState(false);
  const [isPlaying, setIsPlaying] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [seekableStart, setSeekableStart] = useState(0);
  const [seekableEnd, setSeekableEnd] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const progressRef = useRef<HTMLDivElement>(null);
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

  // Handle video metadata loaded
  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video) return;
    // Video metadata loaded - detection resolution is fixed at 1920x1080
    console.log(`[Video] Metadata loaded: ${video.videoWidth}x${video.videoHeight}`);
    setDuration(video.duration || 0);
    updateSeekableRange();
  };

  // Update seekable range for live streams
  const updateSeekableRange = () => {
    const video = videoRef.current;
    if (!video) return;
    
    if (video.seekable.length > 0) {
      setSeekableStart(video.seekable.start(0));
      setSeekableEnd(video.seekable.end(0));
    }
  };

  // Handle time update
  const handleTimeUpdate = () => {
    if (isDragging) return;
    const video = videoRef.current;
    if (!video) return;
    setCurrentTime(video.currentTime);
    updateSeekableRange();
  };

  // Handle play/pause toggle
  const togglePlayPause = () => {
    const video = videoRef.current;
    if (!video) return;
    
    if (video.paused) {
      video.play();
      setIsPlaying(true);
      onResumeStream?.();
    } else {
      video.pause();
      setIsPlaying(false);
      onPauseStream?.();
    }
  };

  // Handle seeking via slider
  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    if (!video) return;
    
    const seekTime = parseFloat(e.target.value);
    setCurrentTime(seekTime);
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
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = currentTime;
    video.play();
    setIsPlaying(true);
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

  // Freeze frame and detect objects
  const handleFreezeAndDetect = async () => {
    const video = videoRef.current;
    if (!video) return;

    // Pause video
    video.pause();
    setIsFrozen(true);

    // Pause HLS loading to prevent buffering in background
    onPauseStream?.();

    // Pass video element to parent for frame capture
    onFrameFreeze(videoRef.current);
  };

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
        {/* Progress Bar */}
        <div className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground w-12 text-right font-mono">
            {formatTime(currentTime)}
          </span>
          <div className="flex-1 relative" ref={progressRef}>
            <input
              type="range"
              min={seekableStart}
              max={seekableEnd || duration || 0}
              step={0.1}
              value={currentTime}
              onChange={handleSeek}
              onMouseDown={handleSeekStart}
              onMouseUp={handleSeekEnd}
              onTouchStart={handleSeekStart}
              onTouchEnd={handleSeekEnd}
              disabled={!streamUrl}
              className="w-full h-1.5 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary hover:accent-primary/80 disabled:opacity-50 disabled:cursor-not-allowed"
              style={{
                background: `linear-gradient(to right, hsl(var(--primary)) 0%, hsl(var(--primary)) ${((currentTime - seekableStart) / ((seekableEnd || duration || 1) - seekableStart)) * 100}%, hsl(var(--secondary)) ${((currentTime - seekableStart) / ((seekableEnd || duration || 1) - seekableStart)) * 100}%, hsl(var(--secondary)) 100%)`
              }}
            />
          </div>
          <span className="text-xs text-muted-foreground w-12 font-mono">
            {streamMode === 'live' ? 'LIVE' : formatTime(seekableEnd || duration)}
          </span>
        </div>

        {/* Buttons */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={togglePlayPause}
              disabled={!streamUrl || isFrozen}
              className="h-9 w-9"
            >
              {isPlaying ? (
                <Pause className="h-4 w-4" />
              ) : (
                <Play className="h-4 w-4" />
              )}
            </Button>

            <Button
              variant="default"
              size="sm"
              onClick={handleFreezeAndDetect}
              disabled={isLoading || isFrozen}
            >
              <Scan className="h-4 w-4 mr-2" />
              Freeze &amp; Detect
            </Button>
          </div>

          {detectedObjects.length > 0 && (
            <span className="text-sm text-muted-foreground">
              {detectedObjects.length} object
              {detectedObjects.length !== 1 ? "s" : ""} detected
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export type { DetectedObject, BoundingBox };
