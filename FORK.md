# FORK.md — この fork が upstream と違うところ

> **次の upstream 取り込みを始める前に、まずこのファイルを読む。**
> branch: `codex/knowledge-core-integration` / upstream: `NousResearch/hermes-agent`
> 手順そのものは castle 側の `docs/hermes-upstream-merge.md` にある。ここに置くのは
> 「castle が名前で依存していて、rename されると無言で壊れるもの」の一覧と、
> 次回先に取り込むべき upstream commit。
>
> 直近の取り込み: upstream `v2026.8.31` (v0.21.0, `29112bef09`) を `540eb829de` へ merge して
> `10ff3cda89`（2026-09-01）。この文書は 2026-09-02 に作成し、fork HEAD `10ff3cda89` と
> upstream tag `29112bef09` の実測差分で裏を取っている。

---

## 1. castle が名前で依存しているもの

castle（`~/ghq/github.com/branch10480/castle`）は Hermes をライブラリのように使う。
**自動マージが競合マーカーを出さずに rename を通しても、castle 側の呼び出しは
`except Exception` に飲まれて無言で素通りする**（前回の merge で全 `pre_tool_call` hook が
実際に全滅した）。merge 後は必ず下記を個別に確認する。

### 1.1 fork 固有の plugin hook（6 件）

`hermes_cli/plugins.py` の `VALID_HOOKS` にあるが **upstream v2026.8.31 には 1 件も存在しない**
（`git grep` で 0 hit を確認済み）:

| hook | 使う castle plugin |
|---|---|
| `attest_scheduled_turn` | `knowledge-jobs` / `local-coder-enforcer` |
| `attest_manual_scheduled_turn` | `local-coder-enforcer` |
| `on_user_correction` | `local-coder-enforcer` |
| `on_context_pressure` | `task-checkpoint` |
| `pre_context_compression` | `task-checkpoint` |
| `post_context_compression` | `task-checkpoint` |

castle の plugin は upstream 由来の hook も使う。こちらは rename されると同じように壊れる:
`pre_tool_call` / `post_tool_call` / `pre_llm_call` / `post_llm_call` /
`transform_llm_output` / `post_api_request` / `on_session_start` /
`on_session_finalize` / `on_session_reset`。

### 1.2 fork 固有のモジュールとシンボル

**ファイルごと fork 専用**（upstream v2026.8.31 に無い）:

- `agent/direct_user_authority.py` — `claim_cloud_egress` / `claim_publication` /
  `issue_bound_capability` / `current_revision`
- `agent/turn_control.py` — `current_tool_execution_context` / `request_current_turn_defer`

**upstream のファイルに fork が足した口**（シンボル名で upstream 0 hit）:

- `hermes_cli.plugins.PluginContext.register_tool` の `halt_on_error` キーワード引数
- `tools.registry.ToolRegistry.handler_accepts_keyword`

**upstream 由来だが castle が import しているもの**（消えたら castle が落ちる）:
`agent.runtime_cwd.resolve_agent_cwd` / `agent.redact.redact_sensitive_text` /
`hermes_cli.plugins.VALID_HOOKS` / `hermes_cli.plugins.PluginContext.register_hook`。

これらは castle の `scripts/setup-hermes-local-llm-safety.sh` の
`verify_hermes_core_contract()` が、空環境の python で import + `assert` して確認する。
壊れると `nrs` が
`Hermes core is missing the required delegation safety interfaces` で止まる。

### 1.3 castle が値を固定している設定キー

いずれも **upstream 由来のキー**だが、castle が `config.yaml` に特定の値を焼いて
毎回検証している。upstream が rename / 意味変更すると、castle の安全設定が黙って外れる:

- `tools.tool_search.enabled` = `off`
- `plugins.enabled` / `plugins.disabled` — castle の allowlist と完全一致であること
- `agent.disabled_toolsets` — 曖昧な core `memory` tool を隠す
- `platform_toolsets.<platform>` — `no_mcp` / `qwen-memory` / `task-checkpoint` /
  `knowledge-jobs` を必須。`cron` / `discord` では `code_execution` 禁止
- `mcp_servers` — 空であること
- `approvals.mode` = `smart`、`auxiliary.approval.{provider,model,api_mode,timeout}`
- `delegation.{provider,model,reasoning_effort}`、`agent.reasoning_effort`
- `cron.mirror_delivery`、`thread_sessions_per_user`
- `security.tirith_path`

見張っているのは castle の `scripts/check-hermes-tool-surface.py`
（`--mode preflight | check | apply | verify-runtime`）と
`scripts/setup-hermes-discord.sh`。

### 1.4 revision pin と「working tree は常に clean」制約

castle は 40 桁の revision を 2 箇所に焼いている:

- `scripts/setup-hermes-local-llm-safety.sh` の `expected_hermes_revision`
- `scripts/test-hermes-local-coder-enforcer.py` の `VERIFIED_HERMES_REVISION`

現在の pin は上記 2 ファイルを参照する。この文書には可変の revision を重複させない。

⚠️ **この checkout の working tree が完全に clean でないと `nrs` が落ちる。**
`verify_hermes_core_contract()` は
`git status --porcelain=v1 --untracked-files=normal` が空であることを要求し、
`~/.hermes/hermes-agent` はこの checkout への symlink なので、
**untracked ファイルを 1 つ置くだけで darwin-rebuild が止まる**。

`FORK.md` は追跡済み。新しい変更は fork branch へ commit・push してから、castle 側の
`hermesup --pin-current`（`scripts/update-hermes-core.sh --pin-current`）で再 pin する。
同期は checkout が clean で、HEAD が対象 branch の remote OID と一致し、従来 pin から
fast-forward できる場合だけ成功する。未公開 commit を先に pin しない。

テスト用 `.venv` と実行用 `venv` は共存できる。castle の更新検証は
`scripts/run_tests.sh --python <実行用venvのPython>` で対象環境を明示し、指定先が
使えなければ他の環境へ切り替えず停止する。通常のテスト実行は従来の選択順を保つ。

---

### Smart Approval の復旧用メタデータ

`tools.approval.SmartApprovalVerdict` は既存の文字列 verdict と互換にし、proxy の `hermes_smart_approval.classification` だけを保持する。castle の binary wrapper が `escalate` を `deny` にしても、`AMBIGUOUS` / `UNAVAILABLE` は実行拒否を保ちつつ危険操作の連続拒否回数から外れる。未知の分類や通常の文字列 `deny` は従来の拒否処理を通る。provider の自由文は復旧指示に使わない。

コマンドと `execute_code` の両経路でこの詳細を受け取り、安全な別操作を提示する。拒否されたコードを保存し直すことは許可の根拠にしない。castle の `scripts/test-hermes-smart-approval-integration.py` が実 proxy → core → plugin → breaker を通して、サービス不調と危険操作を分ける契約を検証する。

## 2. 次の upstream merge で先に取り込む commit

2026-09-02 時点で `HEAD..upstream/main` は 297 commit。全部を一度に入れる前に、
以下 4 件は **castle の運用に直接効く**ので優先して取り込む。
4 件とも `git log -1` で実在を確認済みで、`upstream/main` から到達可能・fork HEAD には未到達。

| commit | 日付 | upstream の件名 | 先に取る理由 |
|---|---|---|---|
| `375ce8eee5` | 2026-09-01 | `ci: block tracked paths that collide case-insensitively` | 前回の merge で実際に踏んだ。APFS は大小文字を区別しないので、contributor email の大小文字違い重複のような tracked path 衝突があると working tree が永久に dirty になり、§1.4 の clean 要求で `nrs` が止まる |
| `ecdbcef7af` | 2026-08-31 | `fix(compression): roll the live transcript back when an in-place compaction commit fails (#99477)` | in-place compaction が失敗したとき live transcript を巻き戻す。ローカル LLM 運用では履歴の in-place 書き換えが KV prefix 失効 = 数分の再 prefill に直結する（§3）ので、失敗して壊れた履歴が残らないことの価値が大きい |
| `045865377c` | 2026-09-01 | `fix(update): restore user model settings config.yaml rewrites drop during update` | update 時の `config.yaml` 書き換えでユーザー設定が落ちる。castle は §1.3 のキー群を `config.yaml` に焼いているので、これを踏むと安全設定が無言で外れる |
| `043c258ac2` | 2026-09-01 | `fix(web): stale removed-backend config warns at startup and errors by name` | 削除済み backend が config に残っているとき、起動時に名前付きで警告・エラーにする。**「インストールされていない依存」は拾わないので、`ddgs` 未インストール問題（castle P37）はこれではカバーされない** |

---

## 3. H-029 の記録欄（履歴の in-place 書き換えによる KV prefix 失効）

castle の `docs/hermes-harness-observations.md` の H-029。
ターン中に prompt の途中（実測では 93,703 token 目以降）が書き換わって
`token-mismatch` になり、live KV が捨てられて 5〜6 分の再 prefill が発生する。
単独稼働時にも再現するため session 競合とは独立で、ローカル LLM 運用での最大の時間損失源。

**現時点の結論（2026-09-02）**

- `in_place_committed` telemetry は **原因ではない**。この指標が立つことと実際の
  prefix 破壊は対応していないので、これを追っても犯人には辿り着かない。
- 実際に効いている書き換えは **31 件**で、いずれも **ターン開始時の
  `idx=3` / `role=tool` の再構成**。ターン中に rolling で走る prune ではなく、
  ターンの入口で tool message を組み直していることが prefix を割っている。

**次に見るところ**

- `~/.hermes/profiles/discord/logs/agent.log` の `prompt_prefix_stability` 行。
  判定軸は `append_only` で、**`since=-` かつ `append_only=false`** の組み合わせだけが
  未解明の履歴変異にあたる（`prefix_reuse_ratio` 単体では判定できない）。
- 上の `idx=3` / `role=tool` 再構成を fork 側で潰すか、upstream の compaction 経路
  （§2 の `ecdbcef7af`）に寄せて集約するか。
