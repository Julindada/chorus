"use client";

import { useState } from "react";
import { createThread, streamRun, resumeRun, type RunInput } from "./lib/langgraph";

const SCHWARTZ_DIMS: { key: string; label: string }[] = [
  { key: "self_direction", label: "自主导向" },
  { key: "stimulation",    label: "刺激" },
  { key: "hedonism",       label: "享乐" },
  { key: "achievement",    label: "成就" },
  { key: "power",          label: "权力" },
  { key: "security",       label: "安全" },
  { key: "conformity",     label: "顺从" },
  { key: "tradition",      label: "传统" },
  { key: "benevolence",    label: "善意" },
  { key: "universalism",   label: "普世" },
];

interface StreamChunk {
  node: string;
  data: unknown;
}

export default function Home() {
  const [username, setUsername]           = useState("");
  const [narrative, setNarrative]         = useState("");
  const [options, setOptions]             = useState(["", ""]);
  const [running, setRunning]             = useState(false);
  const [chunks, setChunks]               = useState<StreamChunk[]>([]);
  const [threadId, setThreadId]           = useState<string | null>(null);
  const [interrupted, setInterrupted]     = useState(false);
  const [schwartz, setSchwartz]           = useState<Record<string, number>>(
    Object.fromEntries(SCHWARTZ_DIMS.map((d) => [d.key, 0.5]))
  );

  function addOption() {
    setOptions((prev) => [...prev, ""]);
  }

  function removeOption(i: number) {
    setOptions((prev) => prev.filter((_, idx) => idx !== i));
  }

  function updateOption(i: number, val: string) {
    setOptions((prev) => prev.map((o, idx) => (idx === i ? val : o)));
  }

  function _detectInterrupt(data: unknown): boolean {
    if (typeof data !== "object" || data === null) return false;
    return "__interrupt__" in data;
  }

  async function _consumeStream(
    gen: AsyncGenerator<{ event: string; data: unknown }>
  ): Promise<boolean> {
    for await (const chunk of gen) {
      if (_detectInterrupt(chunk.data)) {
        setInterrupted(true);
        return true;
      }
      if (chunk.data && typeof chunk.data === "object") {
        for (const [nodeName, nodeData] of Object.entries(
          chunk.data as Record<string, unknown>
        )) {
          if (nodeName === "__metadata__") continue;
          setChunks((prev) => [...prev, { node: nodeName, data: nodeData }]);
        }
      }
    }
    return false;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const filledOptions = options.filter((o) => o.trim());
    if (!username.trim() || !narrative.trim() || filledOptions.length < 2 || running) return;

    setChunks([]);
    setInterrupted(false);
    setRunning(true);

    try {
      const tid = await createThread();
      setThreadId(tid);
      const input: RunInput = {
        username: username.trim(),
        user_narrative: narrative.trim(),
        decision_options: filledOptions,
      };
      await _consumeStream(streamRun(tid, input));
    } finally {
      setRunning(false);
    }
  }

  async function handleResumeValueVector(e: React.FormEvent) {
    e.preventDefault();
    if (!threadId || running) return;

    setInterrupted(false);
    setRunning(true);

    try {
      await _consumeStream(resumeRun(threadId, schwartz));
    } finally {
      setRunning(false);
    }
  }

  const filledOptions = options.filter((o) => o.trim());
  const canSubmit = username.trim() && narrative.trim() && filledOptions.length >= 2 && !running;

  const finalReport = chunks
    .filter(
      (c) =>
        typeof c.data === "object" &&
        c.data !== null &&
        "final_report" in (c.data as object)
    )
    .at(-1)?.data as { final_report?: string } | undefined;

  return (
    <main className="min-h-screen bg-zinc-50 dark:bg-zinc-950 p-8">
      <div className="mx-auto max-w-2xl space-y-8">

        {/* Header */}
        <div>
          <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-50">Chorus</h1>
          <p className="mt-1 text-sm text-zinc-500">多 Agent 决策支持系统</p>
        </div>

        {/* Main form */}
        {!interrupted && (
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="mb-1 block text-xs font-medium text-zinc-500">用户名</label>
              <input
                className="w-full rounded-lg border border-zinc-200 bg-white px-4 py-2 text-sm text-zinc-900 placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-400 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50"
                placeholder="输入用户名"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                disabled={running}
              />
            </div>

            <div>
              <label className="mb-1 block text-xs font-medium text-zinc-500">决策背景描述</label>
              <textarea
                className="w-full rounded-lg border border-zinc-200 bg-white px-4 py-3 text-sm text-zinc-900 placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-400 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50"
                rows={4}
                placeholder="描述你的决策背景，例如：我在现公司工作 3 年，最近收到一份薪资高 30% 但需要换城市的邀请..."
                value={narrative}
                onChange={(e) => setNarrative(e.target.value)}
                disabled={running}
              />
            </div>

            <div>
              <label className="mb-1 block text-xs font-medium text-zinc-500">
                候选选项（至少 2 个）
              </label>
              <div className="space-y-2">
                {options.map((opt, i) => (
                  <div key={i} className="flex gap-2">
                    <input
                      className="flex-1 rounded-lg border border-zinc-200 bg-white px-4 py-2 text-sm text-zinc-900 placeholder-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-400 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50"
                      placeholder={`选项 ${i + 1}`}
                      value={opt}
                      onChange={(e) => updateOption(i, e.target.value)}
                      disabled={running}
                    />
                    {options.length > 2 && (
                      <button
                        type="button"
                        onClick={() => removeOption(i)}
                        disabled={running}
                        className="rounded-lg border border-zinc-200 px-3 text-sm text-zinc-400 hover:text-zinc-700 disabled:opacity-50"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                ))}
              </div>
              <button
                type="button"
                onClick={addOption}
                disabled={running}
                className="mt-2 text-xs text-zinc-400 hover:text-zinc-700 disabled:opacity-50"
              >
                + 添加选项
              </button>
            </div>

            <button
              type="submit"
              disabled={!canSubmit}
              className="rounded-lg bg-zinc-900 px-6 py-2 text-sm font-medium text-white transition-colors hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-50 dark:text-zinc-900 dark:hover:bg-zinc-200"
            >
              {running ? "分析中..." : "开始分析"}
            </button>
          </form>
        )}

        {/* Value vector interrupt form */}
        {interrupted && (
          <form onSubmit={handleResumeValueVector} className="space-y-5 rounded-lg border border-amber-200 bg-amber-50 p-5 dark:border-amber-700 dark:bg-amber-950">
            <div>
              <p className="text-sm font-medium text-amber-800 dark:text-amber-200">
                首次使用需要建立你的价值观画像
              </p>
              <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">
                请为以下 10 个维度打分（0.0 = 完全不重要，1.0 = 极其重要）
              </p>
            </div>
            <div className="space-y-3">
              {SCHWARTZ_DIMS.map((dim) => (
                <div key={dim.key} className="flex items-center gap-4">
                  <span className="w-16 text-xs text-zinc-600 dark:text-zinc-400">{dim.label}</span>
                  <input
                    type="range"
                    min={0} max={1} step={0.1}
                    value={schwartz[dim.key]}
                    onChange={(e) =>
                      setSchwartz((prev) => ({ ...prev, [dim.key]: parseFloat(e.target.value) }))
                    }
                    className="flex-1"
                  />
                  <span className="w-8 text-right text-xs text-zinc-500">
                    {schwartz[dim.key].toFixed(1)}
                  </span>
                </div>
              ))}
            </div>
            <button
              type="submit"
              disabled={running}
              className="rounded-lg bg-amber-700 px-6 py-2 text-sm font-medium text-white transition-colors hover:bg-amber-600 disabled:opacity-50"
            >
              {running ? "提交中..." : "提交并继续"}
            </button>
          </form>
        )}

        {/* Final report */}
        {finalReport?.final_report && (
          <div className="rounded-lg border border-green-200 bg-green-50 p-5 dark:border-green-700 dark:bg-green-950">
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-green-600 dark:text-green-400">
              最终建议
            </p>
            <p className="whitespace-pre-wrap text-sm text-zinc-800 dark:text-zinc-200">
              {finalReport.final_report}
            </p>
          </div>
        )}

        {/* Stream updates */}
        {chunks.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium uppercase tracking-wide text-zinc-400">Agent 进度</p>
            {chunks.map((c, i) => (
              <div
                key={i}
                className="rounded-lg border border-zinc-200 bg-white p-3 dark:border-zinc-700 dark:bg-zinc-900"
              >
                <p className="mb-1 text-xs font-medium text-zinc-400">{c.node}</p>
                <pre className="whitespace-pre-wrap text-xs text-zinc-600 dark:text-zinc-400">
                  {JSON.stringify(c.data, null, 2)}
                </pre>
              </div>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}
