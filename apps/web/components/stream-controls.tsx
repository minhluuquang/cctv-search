"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Calendar, Clock, Play, Square, Scan } from "lucide-react";
import { formatDateTimeToISO } from "@/lib/utils";

type StreamMode = "live" | "playback";

interface StreamControlsProps {
  mode: StreamMode;
  onModeChange: (mode: StreamMode) => void;
  playbackStartTime: string;
  onPlaybackStartTimeChange: (time: string) => void;
  isStreamActive: boolean;
  onStartStream: () => void;
  onStopStream: () => void;
  isLoading: boolean;
  onFreezeAndDetect?: () => void;
}

export function StreamControls({
  mode,
  onModeChange,
  playbackStartTime,
  onPlaybackStartTimeChange,
  isStreamActive,
  onStartStream,
  onStopStream,
  isLoading,
  onFreezeAndDetect,
}: StreamControlsProps) {
  // Initialize with current date/time
  const now = new Date();
  now.setMinutes(now.getMinutes() - 5); // Default to 5 minutes ago
  
  const [date, setDate] = useState(() => {
    return now.toISOString().split("T")[0];
  });
  
  const [time, setTime] = useState(() => {
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    return `${hours}:${minutes}`;
  });

  // Update parent when date/time changes
  useEffect(() => {
    const isoString = formatDateTimeToISO(date, time);
    onPlaybackStartTimeChange(isoString);
  }, [date, time, onPlaybackStartTimeChange]);

  return (
    <div className="space-y-4">
      <Tabs value={mode} onValueChange={(v) => onModeChange(v as StreamMode)}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="live">Live</TabsTrigger>
          <TabsTrigger value="playback">Playback</TabsTrigger>
        </TabsList>
      </Tabs>

      <div className="flex gap-2">
        {!isStreamActive ? (
          <Button
            onClick={onStartStream}
            disabled={isLoading}
            className="flex-1"
          >
            {isLoading ? (
              <>
                <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                Starting...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Start Stream
              </>
            )}
          </Button>
        ) : (
          <Button
            onClick={onStopStream}
            variant="destructive"
            className="flex-1"
          >
            <Square className="mr-2 h-4 w-4" />
            Stop Stream
          </Button>
        )}
      </div>

      {/* Freeze & Detect Button */}
      <Button
        variant="default"
        size="sm"
        onClick={onFreezeAndDetect}
        disabled={!isStreamActive || isLoading}
        className="w-full"
      >
        <Scan className="h-4 w-4 mr-2" />
        Freeze &amp; Detect
      </Button>

      {mode === "playback" && (
        <div className="space-y-3 p-4 border rounded-lg bg-card/50">
          <div className="space-y-2">
            <Label className="flex items-center gap-2">
              <Calendar className="h-4 w-4" />
              Start Date
            </Label>
            <Input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="w-full"
            />
          </div>

          <div className="space-y-2">
            <Label className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              Start Time
            </Label>
            <Input
              type="time"
              value={time}
              onChange={(e) => setTime(e.target.value)}
              className="w-full"
              step="1"
            />
          </div>

          <div className="text-xs text-muted-foreground space-y-1">
            <p>Stream will start from this time</p>
            <p className="font-mono text-xs opacity-60">
              {formatDateTimeToISO(date, time)}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export type { StreamMode };
