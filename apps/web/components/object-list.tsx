"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { User, Car, Bike, Truck, Box, HelpCircle } from "lucide-react";

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface DetectedObject {
  object_id: number;
  label: string;
  confidence: number;
  bbox: BoundingBox;
  center: { x: number; y: number };
}

interface ObjectListProps {
  objects: DetectedObject[];
  onHover: (objectId: number | null) => void;
  frameUrl: string | null;
  isLoading: boolean;
}

// Get icon based on object label
function getObjectIcon(label: string) {
  const normalizedLabel = label.toLowerCase();
  if (normalizedLabel.includes("person") || normalizedLabel.includes("people")) {
    return User;
  }
  if (normalizedLabel.includes("car") || normalizedLabel.includes("vehicle")) {
    return Car;
  }
  if (normalizedLabel.includes("bicycle") || normalizedLabel.includes("bike")) {
    return Bike;
  }
  if (normalizedLabel.includes("truck")) {
    return Truck;
  }
  if (normalizedLabel.includes("box") || normalizedLabel.includes("package")) {
    return Box;
  }
  return HelpCircle;
}

// Get color based on confidence
function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.9) return "bg-green-500/20 text-green-400 border-green-500/30";
  if (confidence >= 0.7) return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
  return "bg-red-500/20 text-red-400 border-red-500/30";
}

export function ObjectList({
  objects,
  onHover,
  frameUrl,
  isLoading,
}: ObjectListProps) {
  const [hoveredId, setHoveredId] = useState<number | null>(null);

  const handleMouseEnter = (objectId: number) => {
    setHoveredId(objectId);
    onHover(objectId);
  };

  const handleMouseLeave = () => {
    setHoveredId(null);
    onHover(null);
  };

  if (isLoading) {
    return (
      <Card className="flex-1 min-h-0 border-border/50 flex flex-col">
        <CardHeader className="pb-3 flex-shrink-0">
          <CardTitle className="text-sm font-medium">Detected Objects</CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col items-center justify-center">
          <div className="flex flex-col items-center justify-center gap-3">
            <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin" />
            <span className="text-sm text-muted-foreground">Analyzing frame...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (objects.length === 0) {
    return (
      <Card className="flex-1 min-h-0 border-border/50 flex flex-col">
        <CardHeader className="pb-3 flex-shrink-0">
          <CardTitle className="text-sm font-medium">Detected Objects</CardTitle>
        </CardHeader>
        <CardContent className="flex-1 flex flex-col items-center justify-center">
          <div className="flex flex-col items-center justify-center text-center">
            <p className="text-sm text-muted-foreground">
              No objects detected
            </p>
            <p className="text-xs text-muted-foreground/60 mt-1">
              Click &quot;Freeze &amp; Detect&quot; to analyze the current frame
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="flex-1 min-h-0 border-border/50 flex flex-col">
      <CardHeader className="pb-3 flex-shrink-0">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">Detected Objects</CardTitle>
          <Badge variant="secondary" className="text-xs">
            {objects.length}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="p-0 flex-1 min-h-0 overflow-hidden">
        <div className="h-full overflow-y-auto">
          <div className="space-y-2 p-4 pt-0">
            {objects.map((obj) => {
              const Icon = getObjectIcon(obj.label);
              const isHovered = hoveredId === obj.object_id;

              return (
                <div
                  key={obj.object_id}
                  className={`
                    group flex items-center gap-3 p-3 rounded-lg cursor-pointer
                    transition-all duration-200 ease-in-out
                    ${
                      isHovered
                        ? "bg-primary/20 border border-primary/50"
                        : "bg-card hover:bg-accent/50 border border-transparent"
                    }
                  `}
                  onMouseEnter={() => handleMouseEnter(obj.object_id)}
                  onMouseLeave={handleMouseLeave}
                >
                  {/* Object Thumbnail - Shows cropped region of this object */}
                  <div className="relative w-16 h-16 rounded-md bg-muted overflow-hidden flex-shrink-0">
                    {frameUrl ? (
                      // Show zoomed region of the frame for this specific object
                      // Scale factor zooms in on the object region
                      <div
                        className="w-full h-full bg-cover bg-no-repeat"
                        style={{
                          backgroundImage: `url(${frameUrl})`,
                          // Calculate position to center on the object
                          backgroundPosition: `${((obj.bbox.x + obj.bbox.width / 2) / 1920) * 100}% ${((obj.bbox.y + obj.bbox.height / 2) / 1080) * 100}%`,
                          // Zoom in to make the object fill the thumbnail
                          backgroundSize: `${1920 / Math.max(obj.bbox.width, obj.bbox.height) * 16}px`,
                        }}
                      />
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center bg-muted/80 group-hover:bg-muted/60 transition-colors">
                        <Icon className="h-6 w-6 text-muted-foreground" />
                      </div>
                    )}
                  </div>

                  {/* Object Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm capitalize truncate">
                        {obj.label}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        #{obj.object_id}
                      </span>
                    </div>

                    <Badge
                      variant="outline"
                      className={`mt-1 text-xs ${getConfidenceColor(obj.confidence)}`}
                    >
                      {(obj.confidence * 100).toFixed(0)}%
                    </Badge>

                    <div className="mt-1 text-xs text-muted-foreground">
                      {obj.bbox.width.toFixed(0)}×{obj.bbox.height.toFixed(0)} px
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export type { DetectedObject };
