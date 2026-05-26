# RepoOperator Web Manual QA

Use this checklist after runtime or chat-stream changes.

1. Open a repository chat.
2. Send `이 레포가 뭐 하는 프로젝트인지 알아내줘.`
3. Confirm progress appears while the final answer is still buffered.
4. Navigate away before completion.
5. Navigate back.
6. Confirm the same thread is visible.
7. Confirm progress or the final answer rehydrates from backend events.
8. Confirm there is no duplicate assistant message.
9. Confirm repeated progress updates merge into stable cards.
10. Send `README.md랑 가장 중요한 entrypoint 파일 하나만 읽고, 실행 흐름을 설명해줘.`
11. Confirm the answer appears and the thread remains after navigation.
12. Send `지난 작업 이력 보여줘.`
13. Confirm read-only Git history/status output appears.
14. Send `지난 작업 이력 보여주고, 지금 변경사항 커밋해줘.`
15. Confirm commit execution is approval-gated and is not run automatically.
16. Send `저장 로직을 안전하게 고쳐줘.`
17. Confirm the work trace says it will search for persistence code, explains why the selected files are candidates, identifies risky serialization evidence if found, and reports proposal-only/no files modified.
18. Send `세이브 파일 깨졌을 때 복구 가능하게 해줘.`
19. Confirm the work trace shows search/read/patch-proposal steps from structured request facts, without needing phrase-list routing.
20. Send `README.md랑 main.py만 읽고 실행 흐름 설명해줘.`
21. Confirm the work trace says it will read the named files first, observes the entrypoint evidence, and prepares an answer from only those files.

## Apply, Validation, And Git Approval

1. Request a small code change and confirm a proposal is created before disk writes.
2. Approve apply and confirm a validation result/card appears after the change is written.
3. Request a commit and confirm a commit approval card appears before any git write.
4. Deny the commit and verify no git commit was created.
5. Request push or PR/MR creation and confirm it is approval-gated.
6. If approval is requested or denied without execution, confirm the UI accurately says no commit, push, or PR/MR was created.

## Agent Work Trace

- Work trace cards should show concise safe summaries: why the next action was chosen, what evidence was observed, remaining uncertainty, and any safety decision.
- Repeated updates with the same activity id should merge into one card.
- Details may be collapsed, but `safe_reasoning_summary` must remain visible.
- Low-level events such as `Loaded context`, `Framed request`, `Updated plan`, and `Recorded observation` should not appear as primary cards; use `Show technical log` only when debugging.
- Raw model reasoning, hidden reasoning, and private reasoning fields must not render in the chat or copy output.

## Thread rehydration (from 2026 patch)

- Active-thread localStorage key is now **repo+branch scoped**: `repooperator-active-thread:{provider}:{path}:{branch}`.
  Opening a different repo or branch selects the saved thread for that exact identity, or starts a fresh thread intentionally.
  Missing branch uses the stable `default` fallback.
  Legacy global key `repooperator-active-thread-id` is read as a one-time fallback and then removed.
- The rehydrate loop guard (using a ref) prevents repeated rehydration when `activeRunByThread` changes.
  Verify by: opening a thread, starting a run, navigating away mid-run, navigating back — confirm single rehydration, no loop.
- Progress cleanup after a run completes is **synchronous** (via `finalizeRunInUi`), replacing the old `setTimeout(100)` hack.
  Verify by: completing a run, immediately navigating away and back — confirm no stale progress cards.
