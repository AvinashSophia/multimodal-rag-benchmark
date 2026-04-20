import axios from "axios";
import type { QueryRequest, QueryResponse, HealthResponse, ConfigOptions } from "../types";

const api = axios.create({ baseURL: "/" });

export async function submitQuery(request: QueryRequest): Promise<QueryResponse> {
  const { data } = await api.post<QueryResponse>("/query", request);
  return data;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>("/health");
  return data;
}

export async function fetchConfigOptions(): Promise<ConfigOptions> {
  const { data } = await api.get<ConfigOptions>("/config/options");
  return data;
}

export async function submitFeedback(request: import("../types").FeedbackRequest): Promise<void> {
  await api.post("/feedback", request);
}

export async function uploadQueryImage(file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<{ path: string }>("/upload-query-image", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data.path;
}

export async function fetchHeatmap(query: string, pageId: string): Promise<string> {
  const { data } = await api.get<{ heatmap: string }>("/heatmap", {
    params: { query, page_id: pageId },
  });
  return data.heatmap;
}

export async function fetchStorageOverview(): Promise<import("../types/index").StorageOverview> {
  const { data } = await api.get("/storage");
  return data;
}

export function imageUrl(pageId: string, _dataset?: string): string {
  // Served by /image/{page_id} — works for both local (redirect) and AWS (S3 proxy)
  return `/image/${pageId}`;
}
