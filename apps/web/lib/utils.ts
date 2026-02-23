import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Format date string and time string to ISO format without timezone conversion.
 * Preserves local time as entered by user.
 * 
 * @param dateStr - Date string in YYYY-MM-DD format
 * @param timeStr - Time string in HH:MM or HH:MM:SS format
 * @returns ISO string in format YYYY-MM-DDTHH:MM:SS
 */
export function formatDateTimeToISO(dateStr: string, timeStr: string): string {
  // Combine date and time, treat as local time
  const dateTimeStr = `${dateStr}T${timeStr}`;
  const date = new Date(dateTimeStr);
  
  // Create ISO string but preserve the local time
  // by manually constructing it without timezone offset
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const seconds = String(date.getSeconds()).padStart(2, '0');
  
  return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}`;
}

/**
 * Format a Date object to ISO string without timezone conversion.
 * Preserves local time as displayed.
 * 
 * @param date - Date object
 * @returns ISO string in format YYYY-MM-DDTHH:MM:SS.mmm
 */
export function formatDateToLocalISO(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const seconds = String(date.getSeconds()).padStart(2, '0');
  const milliseconds = String(date.getMilliseconds()).padStart(3, '0');
  
  return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}.${milliseconds}`;
}
