"use client";

import { motion } from "framer-motion";
import { Check } from "lucide-react";
import { DetectedObject } from "./types";

interface ObjectCardProps {
  object: DetectedObject;
  frameUrl: string | null;
  isHovered?: boolean;
  isSelected?: boolean;
  onHover?: (object: DetectedObject | null) => void;
  onToggleSelect?: (object: DetectedObject) => void;
}

export function ObjectCard({
  object,
  frameUrl,
  isHovered = false,
  isSelected = false,
  onHover,
  onToggleSelect,
}: ObjectCardProps) {
  const handleMouseEnter = () => {
    onHover?.(object);
  };

  const handleMouseLeave = () => {
    onHover?.(null);
  };

  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    onToggleSelect?.(object);
  };

  // Calculate crop background position and size for thumbnail
  const getThumbnailStyle = (): React.CSSProperties => {
    if (!frameUrl) {
      return {};
    }

    const FRAME_WIDTH = 1920;
    const FRAME_HEIGHT = 1080;
    const THUMB_SIZE = 80; // thumbnail width/height in px
    const PADDING = 8; // padding around object in pixels

    // Calculate the crop region to fit the entire object
    const bbox = object.bbox;
    
    // Add padding to the bounding box
    const cropX = Math.max(0, bbox.x - PADDING);
    const cropY = Math.max(0, bbox.y - PADDING);
    const cropWidth = Math.min(FRAME_WIDTH - cropX, bbox.width + PADDING * 2);
    const cropHeight = Math.min(FRAME_HEIGHT - cropY, bbox.height + PADDING * 2);
    
    // Calculate scale to fit the crop region into the thumbnail
    const scaleX = THUMB_SIZE / cropWidth;
    const scaleY = THUMB_SIZE / cropHeight;
    const scale = Math.min(scaleX, scaleY); // Use smaller scale to fit entire object
    
    // Calculate the displayed size of the full frame at this scale
    const displayWidth = FRAME_WIDTH * scale;
    const displayHeight = FRAME_HEIGHT * scale;
    
    // Calculate offset to center the crop region in the thumbnail
    const cropDisplayWidth = cropWidth * scale;
    const cropDisplayHeight = cropHeight * scale;
    const offsetX = (THUMB_SIZE - cropDisplayWidth) / 2 - (cropX * scale);
    const offsetY = (THUMB_SIZE - cropDisplayHeight) / 2 - (cropY * scale);
    
    return {
      backgroundImage: `url(${frameUrl})`,
      backgroundSize: `${displayWidth}px ${displayHeight}px`,
      backgroundPosition: `${offsetX}px ${offsetY}px`,
      backgroundRepeat: 'no-repeat',
    };
  };

  return (
    <motion.div
      layoutId={`object-${object.object_id}`}
      className={`
        flex-shrink-0 w-20 cursor-pointer rounded-lg overflow-hidden relative
        transition-all duration-200
        ${isSelected 
          ? 'ring-2 ring-green-500 ring-offset-2 ring-offset-black' 
          : isHovered 
            ? 'ring-2 ring-yellow-400 ring-offset-2 ring-offset-black'
            : ''
        }
      `}
      whileHover={{ scale: 1.05, y: -2 }}
      whileTap={{ scale: 0.95 }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
    >
      {/* Selection Indicator */}
      {isSelected && (
        <div className="absolute top-1 right-1 z-10 w-5 h-5 bg-green-500 rounded-full flex items-center justify-center">
          <Check className="w-3 h-3 text-black" />
        </div>
      )}

      {/* Thumbnail */}
      <div 
        className={`w-20 h-20 bg-muted relative ${isSelected ? 'opacity-100' : ''}`}
        style={getThumbnailStyle()}
      >
        {!frameUrl && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-2xl">
              {object.label.toLowerCase().includes('car') ? '🚗' :
               object.label.toLowerCase().includes('person') ? '👤' :
               object.label.toLowerCase().includes('dog') ? '🐕' :
               object.label.toLowerCase().includes('box') ? '📦' : '📷'}
            </span>
          </div>
        )}
      </div>
      
      {/* Info */}
      <div className={`px-1.5 py-1 ${isSelected ? 'bg-green-500/20' : 'bg-black/80'} backdrop-blur-sm`}>
        <div className={`text-[10px] font-medium truncate capitalize ${isSelected ? 'text-green-400' : 'text-white'}`}>
          {object.label} #{object.object_id}
        </div>
        <div className={`text-[9px] ${object.confidence >= 0.9 ? 'text-green-400' : object.confidence >= 0.7 ? 'text-yellow-400' : 'text-red-400'}`}>
          {(object.confidence * 100).toFixed(0)}%
        </div>
      </div>
    </motion.div>
  );
}
