import { Client } from "@langchain/langgraph-sdk";

const url = process.env.NEXT_PUBLIC_LANGGRAPH_URL ?? "http://localhost:8123";

export const lgClient = new Client({ apiUrl: url });

export interface RunInput extends Record<string, unknown> {
  username: string;
  user_narrative: string;
  decision_options: string[];
}

export async function createThread() {
  const thread = await lgClient.threads.create();
  return thread.thread_id;
}

export async function* streamRun(threadId: string, input: RunInput) {
  const stream = lgClient.runs.stream(threadId, "chorus", {
    input,
    streamMode: "updates",
  });
  for await (const chunk of stream) {
    yield chunk;
  }
}

export async function* resumeRun(
  threadId: string,
  value: Record<string, number>
) {
  const stream = lgClient.runs.stream(threadId, "chorus", {
    command: { resume: value },
    streamMode: "updates",
  });
  for await (const chunk of stream) {
    yield chunk;
  }
}
