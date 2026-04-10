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
  latency_breakdown: Record<string, number> | null;
}

export interface QueryRequest {
  query: string;
  ground_truth?: string;
  query_image_path?: string;
  model?: string;
  text_method?: string;
  image_method?: string;
  dataset?: string;
}

export interface HealthResponse {
  status: string;
  initialized: boolean;
  dataset: string;
  text_retriever: string;
  image_retriever: string;
  model: string;
}

export interface User {
  name: string;
  email: string;
}

export interface FeedbackRequest {
  query: string;
  answer: string;
  rating: "positive" | "negative";
  feedback_text?: string;
  sources?: string[];
  config?: Record<string, string>;
  user_name?: string;
  user_email?: string;
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
