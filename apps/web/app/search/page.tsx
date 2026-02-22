import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ArrowLeft, Search } from "lucide-react";
import Link from "next/link";

export default function SearchPage() {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <Link
            href="/"
            className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Dashboard
          </Link>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-2">
            <Search className="h-6 w-6" />
            <h1 className="text-3xl font-bold">Object Search</h1>
          </div>
          <p className="text-muted-foreground">
            Search backward in time to find when objects first appeared
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>New Search</CardTitle>
            <CardDescription>
              Configure search parameters to find objects
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="start-time">Start Time</Label>
                  <Input
                    id="start-time"
                    type="datetime-local"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="camera">Camera</Label>
                  <Input
                    id="camera"
                    type="number"
                    defaultValue="1"
                    min={1}
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="object-class">Object Class</Label>
                <Input
                  id="object-class"
                  placeholder="e.g., person, car, bicycle"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="duration">Search Duration (seconds)</Label>
                <Input
                  id="duration"
                  type="number"
                  defaultValue={3600}
                  min={60}
                  max={10800}
                />
              </div>

              <Button type="submit">Start Search</Button>
            </form>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
