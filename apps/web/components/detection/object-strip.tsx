"use client";

import { motion, AnimatePresence } from "framer-motion";
import { ChevronUp, X } from "lucide-react";
import { useRef, useEffect } from "react";
import { DetectedObject } from "./types";
import { ObjectCard } from "./object-card";

interface ObjectStripProps {
  objects: DetectedObject[];
  frameUrl: string | null;
  isExpanded: boolean;
  selectedObjects: DetectedObject[];
  hoveredObject: DetectedObject | null;
  onToggle: () => void;
  onObjectHover: (object: DetectedObject | null) => void;
  onToggleObjectSelect: (object: DetectedObject) => void;
  onClearSelection: () => void;
}

export function ObjectStrip({
  objects,
  frameUrl,
  isExpanded,
  selectedObjects,
  hoveredObject,
  onToggle,
  onObjectHover,
  onToggleObjectSelect,
  onClearSelection,
}: ObjectStripProps) {
  const stripRef = useRef<HTMLDivElement>(null);

  // Close strip when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (isExpanded && stripRef.current && !stripRef.current.contains(event.target as Node)) {
        onToggle();
      }
    };

    if (isExpanded) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isExpanded, onToggle]);

  const isObjectSelected = (obj: DetectedObject) => {
    return selectedObjects.some(selected => selected.object_id === obj.object_id);
  };

  return (
    <>
      {/* Backdrop for click-outside-to-close */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/20 z-40"
            onClick={onToggle}
          />
        )}
      </AnimatePresence>

      {/* Expanded Strip */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            ref={stripRef}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ type: "spring", stiffness: 400, damping: 35 }}
            className="flex-shrink-0 z-50 bg-neutral-900/95 backdrop-blur-md border-b border-neutral-700/50 overflow-hidden relative"
          >
            <div className="px-4 py-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-neutral-200">
                    Detected Objects
                  </span>
                  <span className="px-2 py-0.5 bg-neutral-700/50 rounded-full text-xs text-neutral-400">
                    {objects.length}
                  </span>
                  {selectedObjects.length > 0 && (
                    <>
                      <span className="text-neutral-500">•</span>
                      <span className="px-2 py-0.5 bg-green-500/20 text-green-400 rounded-full text-xs">
                        {selectedObjects.length} selected
                      </span>
                    </>
                  )}
                </div>
                
                <div className="flex items-center gap-2">
                  {selectedObjects.length > 0 && (
                    <button
                      onClick={onClearSelection}
                      className="px-2 py-1 rounded text-xs text-neutral-400 hover:text-white hover:bg-neutral-700 transition-colors"
                    >
                      Clear
                    </button>
                  )}
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={onToggle}
                    className="p-1.5 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-400 hover:text-neutral-200 transition-colors"
                  >
                    <ChevronUp className="w-4 h-4" />
                  </motion.button>
                </div>
              </div>
              
              <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-neutral-600 scrollbar-track-transparent">
                <AnimatePresence mode="popLayout">
                  {objects.map((obj) => (
                    <ObjectCard
                      key={obj.object_id}
                      object={obj}
                      frameUrl={frameUrl}
                      isHovered={hoveredObject?.object_id === obj.object_id}
                      isSelected={isObjectSelected(obj)}
                      onHover={onObjectHover}
                      onToggleSelect={onToggleObjectSelect}
                    />
                  ))}
                </AnimatePresence>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
