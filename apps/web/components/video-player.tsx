"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Pause, Play, Scan } from "lucide-react";

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
}: VideoPlayerProps) {
  const internalVideoRef = useRef<HTMLVideoElement>(null);
  const videoRef = externalVideoRef || internalVideoRef;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(true);
  const [isFrozen, setIsFrozen] = useState(false);
  const [videoSize, setVideoSize] = useState({ width: 1920, height: 1080 });

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

    // Calculate scale factors
    const scaleX = canvas.width / videoSize.width;
    const scaleY = canvas.height / videoSize.height;

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

      // Draw bounding box
      const scaledX = x * scaleX;
      const scaledY = y * scaleY;
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
  }, [detectedObjects, highlightedObjectId, videoSize]);

  // Update canvas when objects or highlight changes
  useEffect(() => {
    drawBoundingBoxes();
  }, [drawBoundingBoxes]);

  // Handle video metadata loaded
  const handleLoadedMetadata = () => {
    const video = videoRef.current;
    if (video) {
      setVideoSize({
        width: video.videoWidth || 1920,
        height: video.videoHeight || 1080,
      });
    }
  };

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

  // Toggle play/pause
  const togglePlayPause = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
      setIsPlaying(false);
    } else {
      video.play();
      setIsPlaying(true);
      setIsFrozen(false);
    }
  };

  // Freeze frame and detect objects
  const handleFreezeAndDetect = async () => {
    const video = videoRef.current;
    if (!video) return;

    // Pause video
    video.pause();
    setIsPlaying(false);
    setIsFrozen(true);

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
      <div className="flex items-center justify-between mt-4 px-2">
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={togglePlayPause}
            disabled={isLoading}
          >
            {isPlaying ? (
              <Pause className="h-4 w-4 mr-2" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            {isPlaying ? "Pause" : "Play"}
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
  );
}

export type { DetectedObject, BoundingBox };
