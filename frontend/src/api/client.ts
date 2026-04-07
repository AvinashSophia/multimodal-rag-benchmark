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

export function imageUrl(pageId: string): string {
  // Page screenshots stored at data/altumint/parsed/figures/{page_id}_page.png
  // Served by FastAPI static mount at /images/ → data/
  return `/images/altumint/parsed/figures/${pageId}_page.png`;
}
