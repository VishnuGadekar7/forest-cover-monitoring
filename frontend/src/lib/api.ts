/**
 * API client — typed wrapper around the backend.
 * Base URL from env (defaults to local dev backend).
 */

import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

export interface ChangeDetectionResult {
  forest_area_t1: number;
  forest_area_t2: number;
  forest_loss: number;
  forest_gain: number;
  percentage_change: number;
  change_map_url: string;
  mask_t1_url: string;
  mask_t2_url: string;
}

/**
 * POST /api/v1/detect-change
 * Sends the two image files and returns the full detection result.
 */
export async function detectChange(
  imageT1: File,
  imageT2: File,
  onProgress?: (pct: number) => void
): Promise<ChangeDetectionResult> {
  const form = new FormData();
  form.append("image_t1", imageT1);
  form.append("image_t2", imageT2);

  const response = await axios.post<ChangeDetectionResult>(
    `${API_BASE}/detect-change`,
    form,
    {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (e) => {
        if (onProgress && e.total) {
          onProgress(Math.round((e.loaded * 100) / e.total));
        }
      },
    }
  );
  return response.data;
}

/** Build a full URL for a static asset returned by the backend. */
export function assetUrl(relativePath: string): string {
  const backendBase = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
  return `${backendBase}${relativePath}`;
}
