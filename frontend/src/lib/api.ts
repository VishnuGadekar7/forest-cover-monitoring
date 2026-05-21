/**
 * API client — typed wrapper around the backend.
 * Base URL from env (defaults to local dev backend).
 */

import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

export interface ChangeDetectionResult {
  id: string;
  forest_area_t1: number;
  forest_area_t2: number;
  forest_loss: number;
  forest_gain: number;
  percentage_change: number;
  change_map_url: string;
  mask_t1_url: string;
  mask_t2_url: string;
  image_t1_url: string;
  image_t2_url: string;
}

/**
 * POST /api/v1/detect-change
 * Sends the two image files and returns the full detection result.
 */
export async function detectChange(
  imageT1: File,
  imageT2: File,
  modelName: string = "attention_unet",
  onProgress?: (pct: number) => void
): Promise<ChangeDetectionResult> {
  const form = new FormData();
  form.append("image_t1", imageT1);
  form.append("image_t2", imageT2);
  form.append("model_name", modelName);

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

/**
 * POST /api/v1/detect-forest-snow
 * Sends the two image files and returns the full detection result.
 */
export async function detectForestSnow(
  imageT1: File,
  imageT2: File,
  modelName: string = "attention_unet",
  onProgress?: (pct: number) => void
): Promise<ChangeDetectionResult> {
  const form = new FormData();
  form.append("image_t1", imageT1);
  form.append("image_t2", imageT2);
  form.append("model_name", modelName);

  const response = await axios.post<ChangeDetectionResult>(
    `${API_BASE}/detect-forest-snow`,
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

export interface STACQueryRequest {
  bbox: [number, number, number, number];
  date_t1: string;
  date_t2: string;
  max_cloud_cover?: number;
  model_name?: string;
}

/**
 * POST /api/v1/detect-change-automated
 * Searches for and processes satellite tiles automatically based on a BBox.
 */
export async function detectChangeAutomated(
  query: STACQueryRequest
): Promise<ChangeDetectionResult> {
  const response = await axios.post<ChangeDetectionResult>(
    `${API_BASE}/detect-change-automated`,
    query
  );
  return response.data;
}

/** Build a full URL for a static asset returned by the backend. */
export function assetUrl(relativePath: string): string {
  const backendBase = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
  return `${backendBase}${relativePath}`;
}

/**
 * GET /api/v1/export-tif
 * Requests a 16-bit GeoTIFF export for a specific task ID.
 * Returns a Blob that can be downloaded by the browser.
 */
export async function exportChangeMapTif(
  taskId: string
): Promise<Blob> {
  const response = await axios.get<Blob>(
    `${API_BASE}/export-tif`,
    {
      params: { task_id: taskId },
      responseType: "blob", // Critical for preventing binary data corruption
    }
  );
  return response.data;
}