"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Play, Pause } from "lucide-react";

interface VideoControlsProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  startTime: Date;
  isPlaying: boolean;
  onPlayPause: () => void;
  onSeek: (time: Date) => void;
  className?: string;
}

// Format seconds to HH:MM:SS
function formatTime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

// Get seconds from start of day for a date
function getSecondsFromStartOfDay(date: Date): number {
  const startOfDay = new Date(date);
  startOfDay.setHours(0, 0, 0, 0);
  return Math.floor((date.getTime() - startOfDay.getTime()) / 1000);
}

// Create date from seconds from start of day
function createDateFromSeconds(seconds: number, referenceDate: Date): Date {
  const date = new Date(referenceDate);
  date.setHours(0, 0, 0, 0);
  date.setSeconds(seconds);
  return date;
}

export function VideoControls({
  videoRef,
  startTime,
  isPlaying,
  onPlayPause,
  onSeek,
  className = "",
}: VideoControlsProps) {
  const sliderRef = useRef<HTMLInputElement>(null);
  const isDraggingRef = useRef(false);
  
  // Use state for the base seconds so it triggers re-renders when updated
  const [baseSeconds, setBaseSeconds] = useState(() => getSecondsFromStartOfDay(startTime));
  const [dragSeconds, setDragSeconds] = useState(0);
  
  const totalSeconds = Math.min(86399, baseSeconds + dragSeconds);

  // Update base seconds when startTime changes (only when not dragging)
  /* eslint-disable react-hooks/set-state-in-effect -- Intentionally syncing prop to state */
  useEffect(() => {
    if (!isDraggingRef.current) {
      const newBaseSeconds = getSecondsFromStartOfDay(startTime);
      setBaseSeconds(newBaseSeconds);
      setDragSeconds(0);
    }
  }, [startTime]);
  /* eslint-enable react-hooks/set-state-in-effect */

  // Sync with video currentTime
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => {
      if (!isDraggingRef.current && video.currentTime) {
        setDragSeconds(Math.floor(video.currentTime));
      }
    };

    video.addEventListener("timeupdate", handleTimeUpdate);
    return () => video.removeEventListener("timeupdate", handleTimeUpdate);
  }, [videoRef]);

  const handleSliderChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const seconds = parseInt(e.target.value, 10);
    setDragSeconds(seconds - baseSeconds);
  }, [baseSeconds]);

  const handleSliderMouseDown = useCallback(() => {
    isDraggingRef.current = true;
  }, []);

  const handleSeekComplete = useCallback(() => {
    isDraggingRef.current = false;
    const newTime = createDateFromSeconds(totalSeconds, startTime);
    // Update base to current position before seeking
    setBaseSeconds(totalSeconds);
    setDragSeconds(0);
    onSeek(newTime);
  }, [totalSeconds, startTime, onSeek]);

  const handleSliderTouchStart = useCallback(() => {
    isDraggingRef.current = true;
  }, []);

  return (
    <div className={`flex items-center gap-3 bg-black/80 backdrop-blur-sm rounded-lg px-3 py-2 ${className}`}>
      {/* Play/Pause Button */}
      <button
        onClick={onPlayPause}
        className="flex items-center justify-center w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors"
        aria-label={isPlaying ? "Pause" : "Play"}
      >
        {isPlaying ? (
          <Pause className="w-4 h-4" />
        ) : (
          <Play className="w-4 h-4 ml-0.5" />
        )}
      </button>

      {/* Time Display */}
      <div className="text-xs text-white/80 font-mono whitespace-nowrap min-w-[70px]">
        {formatTime(totalSeconds)}
      </div>

      {/* Seek Slider */}
      <div className="flex-1 relative">
        <input
          ref={sliderRef}
          type="range"
          min={0}
          max={86399}
          value={totalSeconds}
          onChange={handleSliderChange}
          onMouseDown={handleSliderMouseDown}
          onMouseUp={handleSeekComplete}
          onTouchStart={handleSliderTouchStart}
          onTouchEnd={handleSeekComplete}
          className="w-full h-2 bg-white/20 rounded-full appearance-none cursor-pointer hover:bg-white/30 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500/50 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-blue-500 [&::-webkit-thumb]:cursor-pointer [&::-webkit-slider-thumb]:hover:scale-110 [&::-webkit-slider-thumb]:transition-transform"
        />
        
        {/* Track Progress */}
        <div
          className="absolute top-1/2 left-0 h-2 bg-blue-500/50 rounded-full pointer-events-none -translate-y-1/2"
          style={{ width: `${(totalSeconds / 86399) * 100}%` }}
        />
      </div>

      {/* End Time Display */}
      <div className="text-xs text-white/50 font-mono whitespace-nowrap">
        23:59:59
      </div>
    </div>
  );
}
