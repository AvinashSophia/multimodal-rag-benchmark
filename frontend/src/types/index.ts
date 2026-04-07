export interface RetrievedTextChunk {
  page_id: string;
  text: string;
  score: number;
}

export interface RetrievedImage {
  page_id: string;
  score: number;
}

export interface QueryResponse {
  query: string;
  answer: string;
  sources: string[];
  retrieved_text: RetrievedTextChunk[];
  retrieved_images: RetrievedImage[];
  metrics: Record<string, number> | null;
  latency_ms: number;
}

export interface QueryRequest {
  query: string;
  ground_truth?: string;
  query_image_path?: string;
  model?: string;
  text_method?: string;
  image_method?: string;
}

export interface HealthResponse {
  status: string;
  initialized: boolean;
  dataset: string;
  text_retriever: string;
  image_retriever: string;
  model: string;
}

export interface ConfigOptions {
  datasets: string[];
  text_methods: string[];
  image_methods: string[];
  models: string[];
  active_dataset: string;
  active_text_method: string;
  active_image_method: string;
  active_model: string;
}
