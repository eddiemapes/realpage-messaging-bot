# messaging-bot

A context-aware messaging bot. For each input record (persona, consent, preferences, context, constraints), it decides **whether** to send a message, **which channel** to use, **when** to send, **what** to say, and the **next action**. It learns all of this only from labelled example records.

## How it works

1. **Load:** `records.py` reads every JSONL file in the pool.
2. **Retrieve:** `retrieval.py` embeds each record's situation with `all-MiniLM-L6-v2` and finds the most similar *other* labelled records by cosine similarity, in memory with numpy.
3. **Decide:** `agent.py` sends those worked examples plus the target record to the decider model through `chat.completions.parse`. It returns a Pydantic `BotDecision`.
4. **Evaluate:** `evaluate.py` runs leave-one-out over the pool. Each record is scored with deterministic rule checks (`checks.py`) and an LLM judge (`judge.py`). Every case is a Langfuse trace with its scores attached.

## Setup

Create a `.env` file in the repository root (one level up) containing `OPENAI_API_KEY`, `DECIDER_MODEL`, `JUDGE_MODEL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL`. Then run:

```
uv sync
```

## Commands

```
uv run messaging-bot validate                         # check the data files parse and are usable
uv run messaging-bot similar <task_id> --show-text    # see which examples retrieval picks
uv run messaging-bot decide <task_id>                 # run the bot on one record, compare to expected
uv run messaging-bot eval                             # leave-one-out over all records
uv run messaging-bot eval --only-file data/holdout.jsonl   # score only the hold-out records
```

## Adding the hold-out set

Paste the records into `data/holdout.jsonl`. You can paste one JSON object per line, pretty-printed objects, objects back-to-back, or a JSON array; the loader accepts all of them. Then:

```
uv run messaging-bot validate
uv run messaging-bot eval --only-file data/holdout.jsonl
```

Hold-out records join the retrieval pool too. Every record is scored against its own `expected` block, using only the *other* records as examples. A record never sees its own answer. Records without `expected` can still be run with `decide`, but they are not scored.

Each eval run writes `outputs/<run_id>.jsonl`. All of its traces share the Langfuse session id `<run_id>`.
