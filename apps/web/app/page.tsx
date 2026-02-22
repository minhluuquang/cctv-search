import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Camera, Search, Video, Settings } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Camera className="h-6 w-6" />
            <h1 className="text-xl font-bold">CCTV Search</h1>
          </div>
          <Button variant="outline" size="sm">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-2">Dashboard</h2>
          <p className="text-muted-foreground">
            AI-powered video analysis and object tracking
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Frame Viewer Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Camera className="h-5 w-5" />
                Frame Viewer
              </CardTitle>
              <CardDescription>
                Extract and view frames from cameras
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link href="/frames">
                <Button className="w-full">View Frames</Button>
              </Link>
            </CardContent>
          </Card>

          {/* Object Search Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="h-5 w-5" />
                Object Search
              </CardTitle>
              <CardDescription>
                Search backward to find object appearances
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link href="/search">
                <Button className="w-full">Start Search</Button>
              </Link>
            </CardContent>
          </Card>

          {/* Video Clips Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Video className="h-5 w-5" />
                Video Clips
              </CardTitle>
              <CardDescription>
                Generate and download video clips
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link href="/clips">
                <Button className="w-full">View Clips</Button>
              </Link>
            </CardContent>
          </Card>
        </div>

        {/* API Status */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>System Status</CardTitle>
            <CardDescription>API and service health</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-full bg-green-500 animate-pulse" />
              <span className="text-sm text-muted-foreground">API Connected</span>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
