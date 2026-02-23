"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import Hls from "hls.js";
import { VideoPlayer, type DetectedObject } from "@/components/video-player";
import { ObjectList } from "@/components/object-list";
import { StreamControls, type StreamMode } from "@/components/stream-controls";
import { Button } from "@/components/ui/button";
import { Camera, Settings } from "lucide-react";
import { toast } from "sonner";
import { formatDateToLocalISO } from "@/lib/utils";

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

export default function PlayerPage() {
  // Hardcoded for channel 1
  const CHANNEL = 1;

  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const [capturedFrameUrl, setCapturedFrameUrl] = useState<string | null>(null);
  const [highlightedObjectId, setHighlightedObjectId] = useState<number | null>(
    null
  );
  const [isLoading, setIsLoading] = useState(false);
  const [isStreamLoading, setIsStreamLoading] = useState(false);
  const [isWaitingForStream, setIsWaitingForStream] = useState(false);
  const [streamUrl, setStreamUrl] = useState<string | null>(null);
  const [isStreamStarted, setIsStreamStarted] = useState(false);
  const [streamStartTimestamp, setStreamStartTimestamp] = useState<Date | null>(null);
  const [videoStartTime, setVideoStartTime] = useState<number>(0);

  // Stream mode and playback controls
  const [streamMode, setStreamMode] = useState<StreamMode>("live");
  const [playbackStartTime, setPlaybackStartTime] = useState(() => {
    const now = new Date();
    now.setMinutes(now.getMinutes() - 5);
    return now.toISOString();
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<Hls | null>(null);
  const accumulatedSeekTime = useRef<number>(0);
  const timestampMappingRef = useRef<{ start_time: string; segment_duration: number; segments: Record<string, string> } | null>(null);

  // Helper function to get actual timestamp from API mapping or fallback to calculation
  const getFrameTimestamp = useCallback((videoElement: HTMLVideoElement): Date => {
    let actualTime = videoElement.currentTime;

    if (videoElement.buffered.length > 0) {
      const bufferedEnd = videoElement.buffered.end(0);
      actualTime = Math.min(actualTime, bufferedEnd);
    }

    // Try to get timestamp from API mapping first
    if (timestampMappingRef.current) {
      const mapping = timestampMappingRef.current;
      const segmentDuration = mapping.segment_duration;
      const segmentIndex = Math.floor(actualTime / segmentDuration);
      const segmentFile = `segment_${String(segmentIndex).padStart(3, '0')}.ts`;

      if (mapping.segments[segmentFile]) {
        const segmentTime = new Date(mapping.segments[segmentFile]);
        const offsetInSegment = (actualTime % segmentDuration) * 1000;
        const result = new Date(segmentTime.getTime() + offsetInSegment);
        return result;
      }
    }

    // Fallback to calculation
    if (streamMode === "live" || !streamStartTimestamp) {
      return new Date();
    } else {
      const videoElapsedTime = actualTime - videoStartTime;
      const totalElapsedTime = videoElapsedTime + accumulatedSeekTime.current;
      const result = new Date(streamStartTimestamp.getTime() + (totalElapsedTime * 1000));
      return result;
    }
  }, [streamMode, streamStartTimestamp, videoStartTime]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (hlsRef.current) {
        hlsRef.current.destroy();
      }
      stopStream();
    };
  }, []);

  // Start HLS stream
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
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error("Failed to start stream");
      }

      const data: StreamStartResponse = await response.json();
      const fullUrl = `${API_BASE_URL}${data.playlist_url}`;
      setStreamUrl(fullUrl);
      setIsStreamStarted(true);

      // Track the actual stream start timestamp
      if (streamMode === "live") {
        setStreamStartTimestamp(new Date());
      } else {
        setStreamStartTimestamp(new Date(playbackStartTime));
      }

      // Poll for stream readiness before starting playback
      pollStreamReady(fullUrl);

      toast.success(`${streamMode === "live" ? "Live" : "Playback"} stream starting...`, {
        description: `Channel ${CHANNEL} - Waiting for stream...`,
      });
    } catch (error) {
      console.error("Failed to start stream:", error);
      toast.error("Failed to start stream");
    } finally {
      setIsStreamLoading(false);
    }
  };

  // Stop HLS stream
  const stopStream = async () => {
    try {
      if (hlsRef.current) {
        hlsRef.current.destroy();
        hlsRef.current = null;
      }

      await fetch(`${API_BASE_URL}/stream/stop`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ channel: CHANNEL }),
      });

      setIsStreamStarted(false);
      setIsWaitingForStream(false);
      setStreamUrl(null);
      setStreamStartTimestamp(null);
    } catch (error) {
      console.error("Failed to stop stream:", error);
    }
  };

  // Pause HLS stream loading (for Freeze & Detect)
  const pauseStreamLoading = useCallback(() => {
    if (hlsRef.current) {
      hlsRef.current.stopLoad();
      console.log("[HLS] Stream loading paused");
    }
  }, []);

  // Resume HLS stream loading
  const resumeStreamLoading = useCallback(() => {
    if (hlsRef.current) {
      hlsRef.current.startLoad();
      console.log("[HLS] Stream loading resumed");
    }
  }, []);

  // Poll for stream readiness
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
          setupHlsPlayer(url);
          toast.success(`${streamMode === "live" ? "Live" : "Playback"} stream ready!`, {
            description: `Channel ${CHANNEL}`,
          });
          return;
        }

        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(checkReady, 1000);
        } else {
          setIsWaitingForStream(false);
          toast.error("Stream failed to start", {
            description: "Timeout waiting for stream",
          });
        }
      } catch (error) {
        console.error("Error checking stream status:", error);
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(checkReady, 1000);
        } else {
          setIsWaitingForStream(false);
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
        liveSyncDurationCount: 3,
        liveMaxLatencyDurationCount: 5,
        fragLoadPolicy: {
          default: {
            maxTimeToFirstByteMs: 10000,
            maxLoadTimeMs: 120000,
            timeoutRetry: {
              maxNumRetry: 6,
              retryDelayMs: 1000,
              maxRetryDelayMs: 8000,
            },
            errorRetry: {
              maxNumRetry: 6,
              retryDelayMs: 1000,
              maxRetryDelayMs: 8000,
            },
          },
        },
        playlistLoadPolicy: {
          default: {
            maxTimeToFirstByteMs: 10000,
            maxLoadTimeMs: 120000,
            timeoutRetry: {
              maxNumRetry: 6,
              retryDelayMs: 1000,
              maxRetryDelayMs: 8000,
            },
            errorRetry: {
              maxNumRetry: 6,
              retryDelayMs: 1000,
              maxRetryDelayMs: 8000,
            },
          },
        },
      });

      hls.loadSource(url);
      hls.attachMedia(video);

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        video.play().catch(console.error);

        setVideoStartTime(video.currentTime);
        accumulatedSeekTime.current = 0;

        // Fetch timestamp mapping from API for accurate timestamps
        if (streamMode === "playback") {
          fetch(`${API_BASE_URL}/stream/${CHANNEL}/timestamps`)
            .then(res => res.json())
            .then(data => {
              timestampMappingRef.current = data;
              console.log("[Timestamp] Loaded mapping from API:", data.start_time);
            })
            .catch(err => {
              console.warn("[Timestamp] Failed to load mapping:", err);
            });
        }

        // Track seeking events
        let lastSeekTime = video.currentTime;
        video.addEventListener('seeked', () => {
          const seekOffset = video.currentTime - lastSeekTime;
          accumulatedSeekTime.current += seekOffset;
          lastSeekTime = video.currentTime;
          console.log(`[Video] Seeked: offset=${seekOffset.toFixed(2)}s, total=${accumulatedSeekTime.current.toFixed(2)}s`);
        });
      });

      hls.on(Hls.Events.ERROR, (event, data) => {
        if (data.fatal) {
          switch (data.type) {
            case Hls.ErrorTypes.NETWORK_ERROR:
              console.log("Network error, trying to recover...");
              hls.startLoad();
              break;
            case Hls.ErrorTypes.MEDIA_ERROR:
              console.log("Media error, trying to recover...");
              hls.recoverMediaError();
              break;
            default:
              console.error("Fatal HLS error, restarting stream...");
              hls.destroy();
              setTimeout(() => {
                if (isStreamStarted) {
                  startStream();
                }
              }, 2000);
              break;
          }
        }
      });

      // Detect stalled playback and auto-recover
      let lastTime = video.currentTime;
      const checkStall = setInterval(() => {
        if (video.paused || video.ended) return;

        if (video.currentTime === lastTime) {
          console.log("Video stalled, attempting recovery...");
          hls.recoverMediaError();
        }
        lastTime = video.currentTime;
      }, 5000);

      hlsRef.current = hls;

      // Cleanup interval on destroy
      hls.on(Hls.Events.DESTROYING, () => {
        clearInterval(checkStall);
      });
    } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
      // Native HLS support (Safari)
      video.src = url;
      video.addEventListener("loadedmetadata", () => {
        video.play().catch(console.error);
      });
    }
  };

  // Store the exact timestamp when video was frozen
  const frozenTimestampRef = useRef<string | null>(null);

  // Capture frame from video and detect objects
  const handleFrameFreeze = useCallback(
    async (videoElement: HTMLVideoElement | null) => {
      if (!videoElement) return;

      const frameTime = getFrameTimestamp(videoElement);
      const frameTimestamp = formatDateToLocalISO(frameTime);

      frozenTimestampRef.current = frameTimestamp;

      setIsLoading(true);
      setDetectedObjects([]);

      try {
        const canvas = document.createElement("canvas");
        canvas.width = videoElement.videoWidth || 1920;
        canvas.height = videoElement.videoHeight || 1080;
        const ctx = canvas.getContext("2d");

        if (!ctx) {
          throw new Error("Failed to get canvas context");
        }

        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

        const frameUrl = canvas.toDataURL("image/jpeg", 0.95);
        setCapturedFrameUrl(frameUrl);

        const frameBlob = await new Promise<Blob | null>((resolve) => {
          canvas.toBlob(resolve, "image/jpeg", 0.95);
        });

        if (!frameBlob) {
          throw new Error("Failed to convert canvas to blob");
        }

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
          throw new Error(
            errorData.detail || `HTTP error! status: ${response.status}`
          );
        }

        const data: FrameObjectsResponse = await response.json();
        setDetectedObjects(data.objects);

        toast.success(`Detected ${data.total_objects} objects in frame`, {
          description: `Channel ${data.channel}`,
        });
      } catch (error) {
        console.error("Failed to detect objects:", error);
        toast.error("Failed to detect objects", {
          description:
            error instanceof Error ? error.message : "Unknown error occurred",
        });
        setDetectedObjects([]);
      } finally {
        setIsLoading(false);
      }
    },
    [CHANNEL, getFrameTimestamp]
  );

  // Handle object hover
  const handleObjectHover = useCallback((objectId: number | null) => {
    setHighlightedObjectId(objectId);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/50 backdrop-blur-sm">
        <div className="flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2">
            <Camera className="h-5 w-5 text-primary" />
            <div>
              <h1 className="text-lg font-semibold">CCTV Player</h1>
              <p className="text-xs text-muted-foreground">
                {streamMode === "live" ? "Live Stream" : "Playback"} • Channel {CHANNEL}
              </p>
            </div>
          </div>

          <Button variant="outline" size="sm">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-140px)]">
          {/* Video Player Area */}
          <div className="lg:col-span-3">
            <VideoPlayer
              streamUrl={streamUrl}
              channel={CHANNEL}
              onFrameFreeze={handleFrameFreeze}
              highlightedObjectId={highlightedObjectId}
              detectedObjects={detectedObjects}
              isLoading={isLoading}
              videoRef={videoRef}
              isStreamLoading={isStreamLoading}
              isWaitingForStream={isWaitingForStream}
              streamMode={streamMode}
              onPauseStream={pauseStreamLoading}
              onResumeStream={resumeStreamLoading}
            />
          </div>

          {/* Controls and Object List Sidebar */}
          <div className="lg:col-span-1 space-y-4">
            <StreamControls
              mode={streamMode}
              onModeChange={setStreamMode}
              playbackStartTime={playbackStartTime}
              onPlaybackStartTimeChange={setPlaybackStartTime}
              isStreamActive={isStreamStarted}
              onStartStream={startStream}
              onStopStream={stopStream}
              isLoading={isStreamLoading}
            />

            <ObjectList
              objects={detectedObjects}
              onHover={handleObjectHover}
              frameUrl={capturedFrameUrl}
              isLoading={isLoading}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
