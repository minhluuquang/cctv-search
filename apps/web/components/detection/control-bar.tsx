"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Play, Square, Scan, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { formatDateTimeToISO } from "@/lib/utils";

export type StreamMode = "live" | "playback";

interface ControlBarProps {
  streamMode: StreamMode;
  onStreamModeChange: (mode: StreamMode) => void;
  playbackStartTime: string;
  onPlaybackStartTimeChange: (time: string) => void;
  isStreamActive: boolean;
  isStreamLoading: boolean;
  onStartStream: () => void;
  onStopStream: () => void;
  onFreezeAndDetect: () => void;
  isFrozen: boolean;
  isObjectDetectionLoading: boolean;
  objectCount: number;
  isObjectStripExpanded: boolean;
  onToggleObjectStrip: () => void;
}

export function ControlBar({
  streamMode,
  onStreamModeChange,
  playbackStartTime,
  onPlaybackStartTimeChange,
  isStreamActive,
  isStreamLoading,
  onStartStream,
  onStopStream,
  onFreezeAndDetect,
  isFrozen,
  isObjectDetectionLoading,
  objectCount,
  isObjectStripExpanded,
  onToggleObjectStrip,
}: ControlBarProps) {
  const [date, setDate] = useState(() => {
    const now = new Date();
    now.setMinutes(now.getMinutes() - 5);
    return now.toISOString().split("T")[0];
  });
  
  const [time, setTime] = useState(() => {
    const now = new Date();
    now.setMinutes(now.getMinutes() - 5);
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    return `${hours}:${minutes}`;
  });

  useEffect(() => {
    const isoString = formatDateTimeToISO(date, time);
    onPlaybackStartTimeChange(isoString);
  }, [date, time, onPlaybackStartTimeChange]);

  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="fixed bottom-0 left-0 right-0 z-40 bg-black/90 backdrop-blur-md border-t border-white/10 px-6 py-4"
    >
      <div className="flex items-center gap-4">
        {/* All controls grouped on the left */}
        
        {/* Start/Stop Stream */}
        {!isStreamActive ? (
          <Button
            onClick={onStartStream}
            disabled={isStreamLoading}
            className="bg-blue-500 hover:bg-blue-600 h-10"
          >
            {isStreamLoading ? (
              <>
                <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Starting...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Start
              </>
            )}
          </Button>
        ) : (
          <Button
            onClick={onStopStream}
            variant="destructive"
            className="h-10"
          >
            <Square className="mr-2 h-4 w-4" />
            Stop
          </Button>
        )}

        {/* Freeze & Detect */}
        <Button
          onClick={onFreezeAndDetect}
          disabled={!isStreamActive || isObjectDetectionLoading || isFrozen}
          className="bg-amber-500 hover:bg-amber-600 text-black font-medium disabled:opacity-50 h-10"
        >
          <Scan className="w-4 h-4 mr-2" />
          {isFrozen ? 'Frozen' : 'Freeze'}
        </Button>

        {/* Objects Button */}
        {objectCount > 0 && (
          <Button
            onClick={onToggleObjectStrip}
            variant="outline"
            className={`
              border-amber-500/50 bg-amber-500/10 text-amber-400 hover:bg-amber-500/20 hover:text-amber-300 h-10
              ${isObjectStripExpanded ? 'ring-2 ring-amber-500/50' : ''}
            `}
          >
            <span className="font-semibold">Objects</span>
            <span className="ml-2 px-1.5 py-0.5 bg-amber-500/30 rounded text-xs font-bold min-w-[20px] text-center">
              {objectCount}
            </span>
            {isObjectStripExpanded ? (
              <ChevronUp className="ml-2 w-4 h-4" />
            ) : (
              <ChevronDown className="ml-2 w-4 h-4" />
            )}
          </Button>
        )}

        {/* Divider */}
        <div className="w-px h-8 bg-white/10 mx-2" />

        {/* Mode Toggle */}
        <div className="flex items-center gap-1 bg-white/5 rounded-lg p-1 h-10">
          <button
            onClick={() => onStreamModeChange("live")}
            className={`
              px-4 h-8 rounded-md text-sm font-medium transition-all duration-200 flex items-center
              ${streamMode === "live" 
                ? 'bg-red-500/20 text-red-400' 
                : 'text-white/60 hover:text-white hover:bg-white/5'
              }
            `}
          >
            ● Live
          </button>
          <button
            onClick={() => onStreamModeChange("playback")}
            className={`
              px-4 h-8 rounded-md text-sm font-medium transition-all duration-200 flex items-center
              ${streamMode === "playback" 
                ? 'bg-blue-500/20 text-blue-400' 
                : 'text-white/60 hover:text-white hover:bg-white/5'
              }
            `}
          >
            ▶ Playback
          </button>
        </div>

        {/* Playback Time (only in playback mode) */}
        {streamMode === "playback" && (
          <>
            <Input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="w-40 bg-white/5 border-white/10 text-white text-sm h-10 [color-scheme:dark]"
            />
            <Input
              type="time"
              value={time}
              onChange={(e) => setTime(e.target.value)}
              className="w-36 bg-white/5 border-white/10 text-white text-sm h-10 [color-scheme:dark]"
              step="1"
            />
          </>
        )}
      </div>
    </motion.div>
  );
}
