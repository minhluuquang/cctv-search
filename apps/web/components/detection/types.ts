"use client";

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectedObject {
  object_id: number;
  label: string;
  confidence: number;
  bbox: BoundingBox;
  center: { x: number; y: number };
}

export interface ObjectAction {
  id: string;
  icon: string;
  label: string;
  color: string;
}

export const OBJECT_ACTIONS: ObjectAction[] = [
  { id: 'search', icon: '🔍', label: 'Search', color: '#3b82f6' },
  { id: 'playback', icon: '▶️', label: '-15s', color: '#10b981' },
  { id: 'details', icon: 'ℹ️', label: 'Details', color: '#8b5cf6' },
  { id: 'clear', icon: '✕', label: 'Clear', color: '#ef4444' },
];
