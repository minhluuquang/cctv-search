/** Shared TypeScript types for CCTV Search API */

// ==================== API Request Types ====================

export interface FrameExtractRequest {
  timestamp: string; // ISO 8601 datetime
  channel: number;
}

export interface DetectionRequest {
  camera_id: string;
  timestamp?: string; // ISO 8601 datetime
}

export interface FrameObjectsRequest {
  timestamp: string; // ISO 8601 datetime
  channel: number;
}

export interface VideoClipRequest {
  camera_id: string;
  start_timestamp: string; // ISO 8601 datetime
  duration_seconds: number;
  object_id?: number;
  annotate_objects: boolean;
}

export interface ObjectSearchRequest {
  camera_id: string;
  start_timestamp: string; // ISO 8601 datetime
  object_id: number;
  search_duration_seconds: number;
  object_label: string;
  object_bbox: BoundingBox;
  object_confidence: number;
}

// ==================== API Response Types ====================

export interface FrameExtractResponse {
  frame_path: string;
  timestamp: string;
  channel: number;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectedObjectResponse {
  label: string;
  confidence: number;
  bbox: BoundingBox;
  timestamp: number;
}

export interface DetectedObjectWithId {
  object_id: number;
  label: string;
  confidence: number;
  bbox: BoundingBox;
  center: {
    x: number;
    y: number;
  };
}

export interface FrameObjectsResponse {
  timestamp: string;
  channel: number;
  objects: DetectedObjectWithId[];
  image_path: string;
  total_objects: number;
}

export interface VideoClipResponse {
  clip_path: string;
  start_timestamp: string;
  end_timestamp: string;
  duration_seconds: number;
  file_size_bytes: number;
  download_url: string;
  objects_tracked: Record<string, unknown>[] | null;
}

export interface ObjectSearchResult {
  found: boolean;
  first_seen_timestamp: string | null;
  last_seen_timestamp: string | null;
  search_iterations: number;
  confidence: number | null;
  message: string;
  track_duration_seconds: number | null;
  clip_path: string | null;
  image_path: string | null;
  play_command: string | null;
}

export interface ObjectSearchResponse {
  status: "success" | "not_found" | "error";
  result: ObjectSearchResult | null;
}

// ==================== UI Component Types ====================

export interface Camera {
  id: string;
  name: string;
  channel: number;
  status: "online" | "offline";
}

export interface DetectedObject {
  id: number;
  label: string;
  confidence: number;
  bbox: BoundingBox;
  center: { x: number; y: number };
  color?: string;
}

export interface VideoFrame {
  timestamp: string;
  imageUrl: string;
  objects: DetectedObject[];
  cameraId: string;
}

export interface SearchJob {
  id: string;
  cameraId: string;
  startTime: string;
  targetObject: DetectedObject;
  status: "pending" | "running" | "completed" | "failed";
  progress?: number;
  result?: ObjectSearchResult;
}

export interface VideoClip {
  id: string;
  cameraId: string;
  startTime: string;
  endTime: string;
  duration: number;
  size: number;
  downloadUrl: string;
  thumbnailUrl?: string;
}

// ==================== Common Types ====================

export type ObjectClass =
  | "person"
  | "bicycle"
  | "car"
  | "motorcycle"
  | "bus"
  | "truck"
  | "animal"
  | string;

export interface ApiError {
  detail: string;
  statusCode: number;
}
