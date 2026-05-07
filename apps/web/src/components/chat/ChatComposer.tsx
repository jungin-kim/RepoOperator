"use client";

import { type KeyboardEvent } from "react";

interface ChatComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onCancelQueuedMessage?: (id: string) => void;
  onSteerQueuedMessage?: (id: string) => void;
  onStopRun?: () => void;
  disabled: boolean;
  pending: boolean;
  writeMode?: "basic" | "auto_review" | "full_access";
  queuedMessages?: Array<{ id: string; text: string; status: string; error?: string | null }>;
}

export function ChatComposer({
  value,
  onChange,
  onSubmit,
  onCancelQueuedMessage,
  onSteerQueuedMessage,
  onStopRun,
  disabled,
  pending,
  writeMode = "basic",
  queuedMessages = [],
}: ChatComposerProps) {
  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      if (!disabled && value.trim()) {
        onSubmit();
      }
    }
  }

  // During pending the user can still type and queue messages
  const canSubmit = !disabled && value.trim().length > 0;

  const placeholder = disabled
    ? "Open a repository above before asking a question…"
    : pending
      ? "Type to queue a follow-up message… (⌘+Enter to queue)"
      : writeMode === "auto_review"
        ? "Ask a question or request a change… (⌘+Enter to send)"
        : "Ask a question about the repository… (⌘+Enter to send)";

  const hint = disabled
    ? "Open a repository to start asking questions."
    : pending
      ? queuedMessages.length > 0
        ? `${queuedMessages.length} message${queuedMessages.length === 1 ? "" : "s"} queued — will run after current task finishes.`
        : "Agent is running — type to queue a follow-up."
      : writeMode === "auto_review"
        ? "Auto review — elevated commands and risky actions use approval cards."
        : writeMode === "full_access"
          ? "Full access — broader local actions are enabled and logged."
          : "Basic permissions — repository sandbox work is allowed with guardrails.";

  const buttonLabel = pending
    ? value.trim()
      ? "Queue"
      : writeMode === "auto_review"
        ? "Working…"
        : "Working…"
    : "Ask RepoOperator";

  return (
    <div className="chat-composer-area">
      {queuedMessages.length > 0 && (
        <div className="composer-queue-list" aria-label="Queued messages">
          <div className="composer-queue-title">Queued next</div>
          {queuedMessages.map((item, index) => (
            <div className="composer-queue-item" key={item.id}>
              <span className="composer-queue-order">{index + 1}</span>
              <span className="composer-queue-text">{item.text}</span>
              <span className={`composer-queue-status composer-queue-status-${item.status}`}>{item.status}</span>
              <button type="button" onClick={() => onSteerQueuedMessage?.(item.id)} aria-label="Steer current run with queued message">
                Steer
              </button>
              {item.status === "queued" ? (
                <button type="button" onClick={() => onCancelQueuedMessage?.(item.id)} aria-label="Cancel queued message">
                  Cancel
                </button>
              ) : null}
              {item.error ? <span className="composer-queue-error">{item.error}</span> : null}
            </div>
          ))}
        </div>
      )}
      <div className="composer-form">
        <textarea
          className="composer-textarea"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={3}
        />
        <div className="composer-actions">
          <div className="composer-hint-stack">
            <span className="composer-hint">{hint}</span>
            {pending ? (
              <div className="composer-run-controls" aria-label="Active run controls">
                <button type="button" className="composer-stop-btn" data-testid="stop-run-button" onClick={onStopRun}>
                  Stop
                </button>
              </div>
            ) : null}
          </div>
          <button
            className={`composer-send-btn${pending && value.trim() ? " composer-send-btn-queue" : ""}`}
            type="button"
            onClick={onSubmit}
            disabled={!canSubmit}
          >
            {buttonLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
