"use client";

import { motion, AnimatePresence } from "framer-motion";
import { ObjectAction, OBJECT_ACTIONS } from "./types";

interface RadialMenuProps {
  isOpen: boolean;
  position: { x: number; y: number };
  onAction: (actionId: string) => void;
  onClose: () => void;
}

export function RadialMenu({ isOpen, position, onAction, onClose }: RadialMenuProps) {
  const handleActionClick = (actionId: string) => {
    onAction(actionId);
    onClose();
  };

  // Calculate position for each action (4 items at 90° intervals)
  const getActionPosition = (index: number) => {
    const angle = (index * 90 - 45) * (Math.PI / 180); // Start from top-right
    const radius = 55;
    return {
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
    };
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop to capture clicks outside */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40"
            onClick={onClose}
          />
          
          {/* Radial Menu */}
          <motion.div
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            transition={{ type: "spring", stiffness: 400, damping: 25 }}
            className="fixed z-50 pointer-events-none"
            style={{
              left: position.x,
              top: position.y,
              transform: 'translate(-50%, -50%)',
            }}
          >
            {/* Center indicator */}
            <motion.div
              className="absolute w-3 h-3 bg-white rounded-full"
              style={{ transform: 'translate(-50%, -50%)' }}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.1 }}
            />

            {/* Action buttons */}
            {OBJECT_ACTIONS.map((action, index) => {
              const pos = getActionPosition(index);
              
              return (
                <motion.button
                  key={action.id}
                  initial={{ x: 0, y: 0, scale: 0 }}
                  animate={{ 
                    x: pos.x, 
                    y: pos.y, 
                    scale: 1,
                  }}
                  exit={{ x: 0, y: 0, scale: 0 }}
                  transition={{ 
                    type: "spring", 
                    stiffness: 400, 
                    damping: 25,
                    delay: index * 0.05,
                  }}
                  whileHover={{ scale: 1.2 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => handleActionClick(action.id)}
                  className="absolute pointer-events-auto flex flex-col items-center justify-center"
                  style={{ transform: 'translate(-50%, -50%)' }}
                >
                  <div
                    className="w-11 h-11 rounded-full flex items-center justify-center text-lg shadow-lg border-2 border-white/20"
                    style={{ backgroundColor: action.color }}
                  >
                    {action.icon}
                  </div>
                  <motion.span
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 + index * 0.05 }}
                    className="absolute -bottom-5 text-[10px] text-white font-medium whitespace-nowrap bg-black/60 px-1.5 py-0.5 rounded"
                  >
                    {action.label}
                  </motion.span>
                </motion.button>
              );
            })}
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
