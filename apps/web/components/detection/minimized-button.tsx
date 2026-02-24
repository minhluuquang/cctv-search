"use client";

import { motion } from "framer-motion";
import { Maximize2 } from "lucide-react";
import { DetectedObject } from "./types";

interface MinimizedButtonProps {
  selectedObject: DetectedObject;
  frameUrl: string | null;
  objectCount: number;
  onExpand: () => void;
}

export function MinimizedButton({
  selectedObject,
  frameUrl,
  objectCount,
  onExpand,
}: MinimizedButtonProps) {
  // Calculate thumbnail style
  const getThumbnailStyle = (): React.CSSProperties => {
    if (!frameUrl) {
      return {};
    }

    const centerX = selectedObject.bbox.x + selectedObject.bbox.width / 2;
    const centerY = selectedObject.bbox.y + selectedObject.bbox.height / 2;
    const minDimension = Math.min(selectedObject.bbox.width, selectedObject.bbox.height);
    const scale = Math.max(48 / minDimension, 2);
    
    return {
      backgroundImage: `url(${frameUrl})`,
      backgroundPosition: `${(centerX / 1920) * 100}% ${(centerY / 1080) * 100}%`,
      backgroundSize: `${1920 * scale}px`,
      backgroundRepeat: 'no-repeat',
    };
  };

  return (
    <motion.button
      layoutId={`object-${selectedObject.object_id}`}
      initial={{ opacity: 0, scale: 0.8, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.8, y: 20 }}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={onExpand}
      className="fixed bottom-6 right-6 z-50 flex items-center gap-3 px-3 py-2 bg-black/90 backdrop-blur-md rounded-xl border border-white/10 shadow-2xl hover:bg-black/95 transition-colors"
    >
      {/* Selected object thumbnail */}
      <motion.div
        layoutId={`object-image-${selectedObject.object_id}`}
        className="w-12 h-12 rounded-lg bg-muted flex-shrink-0"
        style={getThumbnailStyle()}
      />
      
      {/* Info */}
      <div className="flex flex-col items-start">
        <span className="text-sm font-medium text-white capitalize">
          {selectedObject.label} #{selectedObject.object_id}
        </span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-white/60">
            {(selectedObject.confidence * 100).toFixed(0)}% confidence
          </span>
          <span className="text-xs text-white/40">•</span>
          <span className="text-xs text-blue-400">
            {objectCount} objects
          </span>
        </div>
      </div>
      
      {/* Expand icon */}
      <Maximize2 className="w-4 h-4 text-white/40 ml-1" />
    </motion.button>
  );
}
