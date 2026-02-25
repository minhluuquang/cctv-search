"use client";

import { useRef, useEffect, useCallback, useState } from "react";
import Hls from "hls.js";
import { DetectedObject } from "./types";
import { VideoControls } from "./video-controls";
import { Search, Play, Info, X, Loader2 } from "lucide-react";

interface VideoPlayerProps {
  streamUrl: string | null;
  channel: number;
  isStreamLoading: boolean;
  isWaitingForStream: boolean;
  isObjectDetectionLoading: boolean;
  streamMode: "live" | "playback";
  playbackStartTime: Date;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isExpanded: boolean;
  previewObject: DetectedObject | null;
  selectedObjects: DetectedObject[];
  detectedObjects: DetectedObject[];
  searchingObjectId: number | null;
  searchResults: Map<number, Date>;
  miniPlayerObjectId: number | null;
  miniPlayerStreamUrl: string | null;
  miniPlayerPlaybackStart: Date;
  isMiniPlayerLoading: boolean;
  onBoxAction: (objectId: number, actionId: string) => void;
  onCloseMiniPlayer: () => void;
  onSeek: (time: Date) => void;
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
  playbackStartTime,
  videoRef,
  isExpanded,
  previewObject,
  selectedObjects,
  detectedObjects,
  searchingObjectId,
  searchResults,
  miniPlayerObjectId,
  miniPlayerStreamUrl,
  miniPlayerPlaybackStart,
  isMiniPlayerLoading,
  onBoxAction,
  onCloseMiniPlayer,
  onSeek,
}: VideoPlayerProps) {
  const isSearching = searchingObjectId !== null;
  const miniPlayerVideoRef = useRef<HTMLVideoElement>(null);
  const miniPlayerHlsRef = useRef<Hls | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const miniPlayerRef = useRef<HTMLDivElement>(null);

  const [scaleData, setScaleData] = useState({
    scaleX: 1,
    scaleY: 1,
    offsetX: 0,
    offsetY: 0
  });

  const [isMainPlaying, setIsMainPlaying] = useState(true);
  const [isMiniPlaying, setIsMiniPlaying] = useState(true);
  const [mainCurrentTime, setMainCurrentTime] = useState(playbackStartTime);
  const [miniCurrentTime, setMiniCurrentTime] = useState(miniPlayerPlaybackStart);
  const [miniPlayerSize, setMiniPlayerSize] = useState({ width: 320, height: 180 });
  const [, setIsResizing] = useState(false);
  const [buttonPositions, setButtonPositions] = useState<Map<number, { left: number; top: number; width: number }>>(new Map());

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
      displayWidth = rect.height * videoAspect;
      offsetX = (rect.width - displayWidth) / 2;
    } else {
      displayHeight = rect.width / videoAspect;
      offsetY = (rect.height - displayHeight) / 2;
    }

    const scaleX = displayWidth / DETECTION_WIDTH;
    const scaleY = displayHeight / DETECTION_HEIGHT;

    return { scaleX, scaleY, offsetX, offsetY, rect };
  }, []);

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

  const drawBoundingBoxes = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const rect = container.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    const drawBox = (obj: DetectedObject, isPreview: boolean, isSelected: boolean) => {
      const { x, y, width, height } = obj.bbox;

      const scaledX = x * scaleData.scaleX + scaleData.offsetX;
      const scaledY = y * scaleData.scaleY + scaleData.offsetY;
      const scaledW = width * scaleData.scaleX;
      const scaledH = height * scaleData.scaleY;

      if (isSelected) {
        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = 4;
        ctx.fillStyle = "rgba(34, 197, 94, 0.15)";
        ctx.shadowColor = "#22c55e";
        ctx.shadowBlur = 15;
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
        ctx.shadowBlur = 0;
        ctx.fillRect(scaledX, scaledY, scaledW, scaledH);
      } else if (isPreview) {
        ctx.strokeStyle = "#facc15";
        ctx.lineWidth = 3;
        ctx.fillStyle = "rgba(250, 204, 21, 0.1)";
        ctx.setLineDash([8, 4]);
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
        ctx.fillRect(scaledX, scaledY, scaledW, scaledH);
        ctx.setLineDash([]);
      }
    };

    if (previewObject) {
      drawBox(previewObject, true, false);
    }

    selectedObjects.forEach(obj => {
      drawBox(obj, false, true);
    });
  }, [previewObject, selectedObjects, scaleData]);

  useEffect(() => {
    const handleResize = () => {
      updateScaleData();
    };

    updateScaleData();

    window.addEventListener("resize", handleResize);

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

  useEffect(() => {
    drawBoundingBoxes();
  }, [drawBoundingBoxes]);

  useEffect(() => {
    if (!miniPlayerStreamUrl || !miniPlayerVideoRef.current) return;

    const video = miniPlayerVideoRef.current;

    if (Hls.isSupported()) {
      const hls = new Hls({
        enableWorker: true,
        lowLatencyMode: false,
      });
      hls.loadSource(miniPlayerStreamUrl);
      hls.attachMedia(video);
      miniPlayerHlsRef.current = hls;

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        video.play().catch(console.error);
      });

      return () => {
        hls.destroy();
        miniPlayerHlsRef.current = null;
      };
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = miniPlayerStreamUrl;
    }
  }, [miniPlayerStreamUrl]);

  // Listen for mini player video play/pause events
  useEffect(() => {
    const video = miniPlayerVideoRef.current;
    if (!video) return;

    const handlePlay = () => setIsMiniPlaying(true);
    const handlePause = () => setIsMiniPlaying(false);

    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);

    return () => {
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- video ref changes trigger re-run
  }, []);

  // Window-style resize handlers for mini player
  const handleResizeStart = useCallback((direction: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsResizing(true);

    const startX = e.clientX;
    const startY = e.clientY;
    const startWidth = miniPlayerSize.width;
    const startHeight = miniPlayerSize.height;

    const handleMouseMove = (e: MouseEvent) => {
      const deltaX = e.clientX - startX;
      const deltaY = e.clientY - startY;

      let newWidth = startWidth;
      let newHeight = startHeight;

      // Handle horizontal resizing
      if (direction.includes('e')) {
        newWidth = Math.max(240, Math.min(1200, startWidth + deltaX));
      }
      if (direction.includes('w')) {
        newWidth = Math.max(240, Math.min(1200, startWidth - deltaX));
      }

      // Handle vertical resizing
      if (direction.includes('s')) {
        newHeight = Math.max(135, Math.min(675, startHeight + deltaY));
      }
      if (direction.includes('n')) {
        newHeight = Math.max(135, Math.min(675, startHeight - deltaY));
      }

      // If dragging corner, maintain aspect ratio
      if (direction.length === 2) {
        const aspectRatio = 16 / 9;
        if (Math.abs(deltaX) > Math.abs(deltaY)) {
          newHeight = newWidth / aspectRatio;
        } else {
          newWidth = newHeight * aspectRatio;
        }
      }

      setMiniPlayerSize({ width: newWidth, height: newHeight });
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [miniPlayerSize]);

  const handleMainPlayPause = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    if (isMainPlaying) {
      video.pause();
    } else {
      video.play().catch(console.error);
    }
    setIsMainPlaying(!isMainPlaying);
  }, [isMainPlaying, videoRef]);

  const handleMiniPlayPause = useCallback(() => {
    const video = miniPlayerVideoRef.current;
    if (!video) return;

    if (isMiniPlaying) {
      video.pause();
    } else {
      video.play().catch(console.error);
    }
    setIsMiniPlaying(!isMiniPlaying);
  }, [isMiniPlaying]);

  const handleMainSeek = useCallback((time: Date) => {
    setMainCurrentTime(time);
    onSeek(time);
  }, [onSeek]);

  // Handle mini player seek - seek within current video
  const handleMiniSeek = useCallback((time: Date) => {
    setMiniCurrentTime(time);

    const video = miniPlayerVideoRef.current;
    if (!video) return;

    // Calculate offset from playback start time
    const offsetSeconds = (time.getTime() - miniPlayerPlaybackStart.getTime()) / 1000;
    
    // Seek the video to the new position
    if (video.currentTime !== offsetSeconds) {
      video.currentTime = Math.max(0, offsetSeconds);
    }
  }, [miniPlayerPlaybackStart]);

  useEffect(() => {
    setMainCurrentTime(playbackStartTime);
  }, [playbackStartTime]);

  useEffect(() => {
    setMiniCurrentTime(miniPlayerPlaybackStart);
  }, [miniPlayerPlaybackStart]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handlePlay = () => setIsMainPlaying(true);
    const handlePause = () => setIsMainPlaying(false);

    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);

    return () => {
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
    };
  }, [videoRef]);

  const getActionButtonsPosition = useCallback((obj: DetectedObject) => {
    if (!containerRef.current) return null;

    const { x, y, width } = obj.bbox;
    const scaledX = x * scaleData.scaleX + scaleData.offsetX;
    const scaledY = y * scaleData.scaleY + scaleData.offsetY;
    const scaledW = width * scaleData.scaleX;

    return {
      left: scaledX,
      top: Math.max(8, scaledY - 44),
      width: scaledW
    };
  }, [scaleData]);

  useEffect(() => {
    const newPositions = new Map<number, { left: number; top: number; width: number }>();
    selectedObjects.forEach(obj => {
      const position = getActionButtonsPosition(obj);
      if (position) {
        newPositions.set(obj.object_id, position);
      }
    });
    setButtonPositions(newPositions);
  }, [scaleData, selectedObjects, getActionButtonsPosition]);

  return (
    <div
      ref={containerRef}
      className={`
        relative w-full h-full bg-black overflow-hidden
        ${isExpanded ? 'opacity-70' : 'opacity-100'}
        transition-opacity duration-300
      `}
    >
      <video
        ref={videoRef}
        src={streamUrl || undefined}
        autoPlay
        muted
        playsInline
        className="absolute inset-0 w-full h-full object-contain"
        crossOrigin="anonymous"
      />

      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />

      {selectedObjects.map(obj => {
        const position = buttonPositions.get(obj.object_id);
        if (!position) return null;

        const isThisObjectSearching = searchingObjectId === obj.object_id;
        const shouldDisableButtons = isSearching && !isThisObjectSearching;
        const hasSearchResult = searchResults.has(obj.object_id);
        const isMiniPlayerOpen = miniPlayerObjectId === obj.object_id;

        return (
          <div key={obj.object_id}>
            <div
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
                  disabled={isSearching}
                  className={`p-1.5 rounded-md transition-colors relative ${
                    isThisObjectSearching
                      ? 'bg-blue-500 text-white cursor-wait'
                      : 'bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 disabled:opacity-50 disabled:cursor-not-allowed'
                  }`}
                  title={isThisObjectSearching ? 'Searching...' : 'Search origin'}
                >
                  {isThisObjectSearching ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Search className="w-3.5 h-3.5" />
                  )}
                </button>
                {hasSearchResult && (
                  <button
                    onClick={() => onBoxAction(obj.object_id, 'playback')}
                    disabled={shouldDisableButtons || isMiniPlayerOpen}
                    className={`p-1.5 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                      isMiniPlayerOpen
                        ? 'bg-green-500 text-white'
                        : 'bg-green-500/20 hover:bg-green-500/30 text-green-400'
                    }`}
                    title={isMiniPlayerOpen ? 'Playback active' : 'Playback -15s'}
                  >
                    <Play className="w-3.5 h-3.5" />
                  </button>
                )}
                <button
                  onClick={() => onBoxAction(obj.object_id, 'details')}
                  disabled={shouldDisableButtons}
                  className="p-1.5 rounded-md bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Details"
                >
                  <Info className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => onBoxAction(obj.object_id, 'clear')}
                  disabled={shouldDisableButtons}
                  className="p-1.5 rounded-md bg-red-500/20 hover:bg-red-500/30 text-red-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Clear"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {isMiniPlayerOpen && (
              <div
                ref={miniPlayerRef}
                className="absolute z-40 bg-black rounded-lg overflow-hidden shadow-2xl border border-white/20 group"
                style={{
                  left: position.left,
                  top: position.top + 40,
                  width: miniPlayerSize.width,
                  height: 'auto',
                }}
              >
                {/* Resize handles - only visible on hover */}
                {/* Top edge */}
                <div
                  className="absolute -top-1 left-2 right-2 h-2 cursor-n-resize z-50 opacity-0 group-hover:opacity-100 transition-opacity"
                  onMouseDown={handleResizeStart('n')}
                  title="Resize top"
                />
                {/* Bottom edge */}
                <div
                  className="absolute -bottom-1 left-2 right-2 h-2 cursor-s-resize z-50 opacity-0 group-hover:opacity-100 transition-opacity"
                  onMouseDown={handleResizeStart('s')}
                  title="Resize bottom"
                />
                {/* Left edge */}
                <div
                  className="absolute top-2 bottom-2 -left-1 w-2 cursor-w-resize z-50 opacity-0 group-hover:opacity-100 transition-opacity"
                  onMouseDown={handleResizeStart('w')}
                  title="Resize left"
                />
                {/* Right edge */}
                <div
                  className="absolute top-2 bottom-2 -right-1 w-2 cursor-e-resize z-50 opacity-0 group-hover:opacity-100 transition-opacity"
                  onMouseDown={handleResizeStart('e')}
                  title="Resize right"
                />
                {/* Top-left corner */}
                <div
                  className="absolute -top-1 -left-1 w-4 h-4 cursor-nw-resize z-50 opacity-0 group-hover:opacity-100 transition-opacity"
                  onMouseDown={handleResizeStart('nw')}
                  title="Resize corner"
                />
                {/* Top-right corner */}
                <div
                  className="absolute -top-1 -right-1 w-4 h-4 cursor-ne-resize z-50 opacity-0 group-hover:opacity-100 transition-opacity"
                  onMouseDown={handleResizeStart('ne')}
                  title="Resize corner"
                />
                {/* Bottom-left corner */}
                <div
                  className="absolute -bottom-1 -left-1 w-4 h-4 cursor-sw-resize z-50 opacity-0 group-hover:opacity-100 transition-opacity"
                  onMouseDown={handleResizeStart('sw')}
                  title="Resize corner"
                />
                {/* Bottom-right corner */}
                <div
                  className="absolute -bottom-1 -right-1 w-4 h-4 cursor-se-resize z-50 opacity-0 group-hover:opacity-100 transition-opacity"
                  onMouseDown={handleResizeStart('se')}
                  title="Resize corner"
                />

                <div className="flex items-center justify-between px-2 py-1 bg-black/80 border-b border-white/10">
                  <span className="text-xs text-white/70">15s Playback</span>
                  <button
                    onClick={onCloseMiniPlayer}
                    className="p-0.5 hover:bg-white/10 rounded text-white/70 hover:text-white"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>

                <div
                  className="relative bg-black"
                  style={{ height: miniPlayerSize.height }}
                >
                  {isMiniPlayerLoading ? (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <Loader2 className="w-6 h-6 text-green-400 animate-spin" />
                    </div>
                  ) : (
                    <video
                      ref={miniPlayerVideoRef}
                      src={miniPlayerStreamUrl || undefined}
                      autoPlay
                      muted
                      playsInline
                      className="w-full h-full object-contain"
                    />
                  )}
                </div>

                <div className="px-2 pb-2 bg-black/80">
                  <VideoControls
                    videoRef={miniPlayerVideoRef}
                    startTime={miniCurrentTime}
                    isPlaying={isMiniPlaying}
                    onPlayPause={handleMiniPlayPause}
                    onSeek={handleMiniSeek}
                  />
                </div>
              </div>
            )}
          </div>
        );
      })}

      {isStreamLoading && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center z-10">
          <div className="flex flex-col items-center gap-3">
            <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-white/60">Starting stream...</span>
          </div>
        </div>
      )}

      {isWaitingForStream && !isStreamLoading && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center z-10">
          <div className="flex flex-col items-center gap-3">
            <div className="w-12 h-12 border-4 border-yellow-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-white/60">Waiting for stream...</span>
          </div>
        </div>
      )}

      {isObjectDetectionLoading && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-10">
          <div className="flex flex-col items-center gap-3">
            <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-white/60">Detecting objects...</span>
          </div>
        </div>
      )}

      <div className="absolute top-4 left-4 px-3 py-1.5 bg-black/70 rounded-md z-20">
        <span className="text-xs font-medium text-white">Channel {channel}</span>
      </div>

      <div className="absolute top-4 left-24 px-3 py-1.5 bg-black/70 rounded-md z-20">
        <span className={`text-xs font-medium ${streamMode === "live" ? "text-red-400" : "text-blue-400"}`}>
          {streamMode === "live" ? "● LIVE" : "▶ PLAYBACK"}
        </span>
      </div>

      {detectedObjects.length > 0 && (
        <div className="absolute bottom-4 left-4 px-3 py-1.5 bg-black/70 rounded-md z-20">
          <span className="text-xs text-white/70">
            {detectedObjects.length} object{detectedObjects.length !== 1 ? 's' : ''} detected
          </span>
        </div>
      )}

      {streamUrl && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-30 w-full max-w-2xl px-4"
        >
          <VideoControls
            videoRef={videoRef}
            startTime={mainCurrentTime}
            isPlaying={isMainPlaying}
            onPlayPause={handleMainPlayPause}
            onSeek={handleMainSeek}
          />
        </div>
      )}
    </div>
  );
}
