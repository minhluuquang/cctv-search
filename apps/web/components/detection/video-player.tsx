"use client";

import { useRef, useEffect, useCallback, useState } from "react";
import { DetectedObject } from "./types";
import { Search, Play, Info, X } from "lucide-react";

interface VideoPlayerProps {
  streamUrl: string | null;
  channel: number;
  isStreamLoading: boolean;
  isWaitingForStream: boolean;
  isObjectDetectionLoading: boolean;
  streamMode: "live" | "playback";
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isExpanded: boolean;
  previewObject: DetectedObject | null;
  selectedObjects: DetectedObject[];
  detectedObjects: DetectedObject[];
  onBoxAction: (objectId: number, actionId: string) => void;
}

const DETECTION_WIDTH = 1920;
const DETECTION_HEIGHT = 1080;

export function VideoPlayer({
  streamUrl,
  channel,
  isStreamLoading,
  isWaitingForStream,
  isObjectDetectionLoading,
  streamMode,
  videoRef,
  isExpanded,
  previewObject,
  selectedObjects,
  detectedObjects,
  onBoxAction,
}: VideoPlayerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Store scale factors in state so they trigger re-renders when updated
  const [scaleData, setScaleData] = useState({
    scaleX: 1,
    scaleY: 1,
    offsetX: 0,
    offsetY: 0
  });

  // Calculate scale factors based on current container size
  const calculateScaleFactors = useCallback(() => {
    const container = containerRef.current;
    if (!container) return null;

    const rect = container.getBoundingClientRect();
    const videoAspect = DETECTION_WIDTH / DETECTION_HEIGHT;
    const containerAspect = rect.width / rect.height;

    let displayWidth = rect.width;
    let displayHeight = rect.height;
    let offsetX = 0;
    let offsetY = 0;

    if (containerAspect > videoAspect) {
      // Container is wider - black bars on left/right
      displayWidth = rect.height * videoAspect;
      offsetX = (rect.width - displayWidth) / 2;
    } else {
      // Container is taller - black bars on top/bottom
      displayHeight = rect.width / videoAspect;
      offsetY = (rect.height - displayHeight) / 2;
    }

    const scaleX = displayWidth / DETECTION_WIDTH;
    const scaleY = displayHeight / DETECTION_HEIGHT;

    return { scaleX, scaleY, offsetX, offsetY, rect };
  }, []);

  // Update scale data whenever container changes
  const updateScaleData = useCallback(() => {
    const newScaleData = calculateScaleFactors();
    if (newScaleData) {
      setScaleData({
        scaleX: newScaleData.scaleX,
        scaleY: newScaleData.scaleY,
        offsetX: newScaleData.offsetX,
        offsetY: newScaleData.offsetY
      });
    }
  }, [calculateScaleFactors]);

  // Draw bounding boxes on canvas
  const drawBoundingBoxes = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Set canvas size to match container
    const rect = container.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    // Helper to draw a box
    const drawBox = (obj: DetectedObject, isPreview: boolean, isSelected: boolean) => {
      const { x, y, width, height } = obj.bbox;

      const scaledX = x * scaleData.scaleX + scaleData.offsetX;
      const scaledY = y * scaleData.scaleY + scaleData.offsetY;
      const scaledW = width * scaleData.scaleX;
      const scaledH = height * scaleData.scaleY;

      if (isSelected) {
        // Selected: solid green with glow
        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = 4;
        ctx.fillStyle = "rgba(34, 197, 94, 0.15)";
        
        // Draw glow effect
        ctx.shadowColor = "#22c55e";
        ctx.shadowBlur = 15;
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
        ctx.shadowBlur = 0;
        ctx.fillRect(scaledX, scaledY, scaledW, scaledH);
      } else if (isPreview) {
        // Preview: dashed yellow
        ctx.strokeStyle = "#facc15";
        ctx.lineWidth = 3;
        ctx.fillStyle = "rgba(250, 204, 21, 0.1)";
        ctx.setLineDash([8, 4]);
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
        ctx.fillRect(scaledX, scaledY, scaledW, scaledH);
        ctx.setLineDash([]);
      }
    };

    // Draw preview object first (dashed yellow)
    if (previewObject) {
      drawBox(previewObject, true, false);
    }

    // Draw all selected objects (solid green)
    selectedObjects.forEach(obj => {
      drawBox(obj, false, true);
    });
  }, [previewObject, selectedObjects, scaleData]);

  // Initial setup and resize handling
  useEffect(() => {
    const handleResize = () => {
      updateScaleData();
    };

    // Initial scale calculation
    updateScaleData();

    // Add resize listener
    window.addEventListener("resize", handleResize);
    
    // Use ResizeObserver for more accurate container size changes
    const container = containerRef.current;
    if (container && typeof ResizeObserver !== 'undefined') {
      const resizeObserver = new ResizeObserver(() => {
        updateScaleData();
      });
      resizeObserver.observe(container);
      
      return () => {
        window.removeEventListener("resize", handleResize);
        resizeObserver.disconnect();
      };
    }

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [updateScaleData]);

  // Redraw when objects change or scale data changes
  useEffect(() => {
    drawBoundingBoxes();
  }, [drawBoundingBoxes]);

  // Calculate position for action buttons for a specific object
  const getActionButtonsPosition = (obj: DetectedObject) => {
    if (!containerRef.current) return null;

    const { x, y, width } = obj.bbox;
    const scaledX = x * scaleData.scaleX + scaleData.offsetX;
    const scaledY = y * scaleData.scaleY + scaleData.offsetY;
    const scaledW = width * scaleData.scaleX;

    // Position buttons above the box
    return {
      left: scaledX,
      top: Math.max(8, scaledY - 44), // 44px above box, minimum 8px from top
      width: scaledW
    };
  };

  return (
    <div
      ref={containerRef}
      className={`
        relative w-full h-full bg-black overflow-hidden
        ${isExpanded ? 'opacity-70' : 'opacity-100'}
        transition-opacity duration-300
      `}
    >
      {/* Video Element */}
      <video
        ref={videoRef}
        src={streamUrl || undefined}
        autoPlay
        muted
        playsInline
        className="absolute inset-0 w-full h-full object-contain"
        crossOrigin="anonymous"
      />

      {/* Canvas Overlay */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />

      {/* Action Buttons for Selected Objects */}
      {selectedObjects.map(obj => {
        const position = getActionButtonsPosition(obj);
        if (!position) return null;

        return (
          <div
            key={obj.object_id}
            className="absolute z-30 flex items-center gap-1"
            style={{
              left: position.left,
              top: position.top,
              width: position.width,
            }}
          >
            <div className="flex items-center gap-1 bg-black/80 backdrop-blur-sm rounded-lg p-1">
              <button
                onClick={() => onBoxAction(obj.object_id, 'search')}
                className="p-1.5 rounded-md bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 transition-colors"
                title="Search"
              >
                <Search className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => onBoxAction(obj.object_id, 'playback')}
                className="p-1.5 rounded-md bg-green-500/20 hover:bg-green-500/30 text-green-400 transition-colors"
                title="Playback -15s"
              >
                <Play className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => onBoxAction(obj.object_id, 'details')}
                className="p-1.5 rounded-md bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 transition-colors"
                title="Details"
              >
                <Info className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => onBoxAction(obj.object_id, 'clear')}
                className="p-1.5 rounded-md bg-red-500/20 hover:bg-red-500/30 text-red-400 transition-colors"
                title="Clear"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        );
      })}

      {/* Stream Loading Overlay */}
      {isStreamLoading && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center z-10">
          <div className="flex flex-col items-center gap-3">
            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-white/60">Starting stream...</span>
          </div>
        </div>
      )}

      {/* Waiting for Stream Overlay */}
      {isWaitingForStream && !isStreamLoading && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center z-10">
          <div className="flex flex-col items-center gap-3">
            <div className="w-12 h-12 border-4 border-yellow-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-white/60">Waiting for stream...</span>
          </div>
        </div>
      )}

      {/* Object Detection Loading */}
      {isObjectDetectionLoading && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-10">
          <div className="flex flex-col items-center gap-3">
            <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-white/60">Detecting objects...</span>
          </div>
        </div>
      )}

      {/* Channel Badge */}
      <div className="absolute top-4 left-4 px-3 py-1.5 bg-black/70 rounded-md z-20">
        <span className="text-xs font-medium text-white">Channel {channel}</span>
      </div>

      {/* Stream Mode Badge */}
      <div className="absolute top-4 left-24 px-3 py-1.5 bg-black/70 rounded-md z-20">
        <span className={`text-xs font-medium ${streamMode === "live" ? "text-red-400" : "text-blue-400"}`}>
          {streamMode === "live" ? "● LIVE" : "▶ PLAYBACK"}
        </span>
      </div>

      {/* Object Count (when detected) */}
      {detectedObjects.length > 0 && (
        <div className="absolute bottom-4 left-4 px-3 py-1.5 bg-black/70 rounded-md z-20">
          <span className="text-xs text-white/70">
            {detectedObjects.length} object{detectedObjects.length !== 1 ? 's' : ''} detected
          </span>
        </div>
      )}
    </div>
  );
}
