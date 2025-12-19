PRAGMA foreign_keys = ON;

-- A) Questionnaire registry (supports multiple questionnaires + versions)
CREATE TABLE IF NOT EXISTS questionnaires (
  questionnaire_id TEXT PRIMARY KEY,
  name             TEXT NOT NULL,
  version          TEXT NOT NULL,
  created_at       TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(name, version)
);

-- B) Schema registry: question definitions
CREATE TABLE IF NOT EXISTS questionnaire_schema (
  question_id       TEXT PRIMARY KEY,
  questionnaire_id  TEXT NOT NULL,
  column_name       TEXT NOT NULL,          -- stable key used by agents (semantic handle)
  question_text     TEXT NOT NULL,
  type              TEXT NOT NULL,          -- numeric/categorical/likert/text/date/json
  allowed_values    TEXT,                   -- JSON string (nullable)
  missing_rules     TEXT,                   -- JSON string (nullable)
  privacy_level     TEXT NOT NULL DEFAULT 'normal', -- normal/sensitive/restricted
  order_index       INTEGER NOT NULL DEFAULT 0,
  is_active         INTEGER NOT NULL DEFAULT 1,
  created_at        TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at        TEXT NOT NULL DEFAULT (datetime('now')),

  FOREIGN KEY(questionnaire_id) REFERENCES questionnaires(questionnaire_id) ON DELETE CASCADE,
  UNIQUE(questionnaire_id, column_name)
);

CREATE INDEX IF NOT EXISTS idx_schema_questionnaire
ON questionnaire_schema(questionnaire_id);

CREATE INDEX IF NOT EXISTS idx_schema_column_name
ON questionnaire_schema(column_name);

-- C) Response header (one row per submission)
CREATE TABLE IF NOT EXISTS responses (
  response_id      TEXT PRIMARY KEY,
  questionnaire_id TEXT NOT NULL,
  submitted_at     TEXT NOT NULL DEFAULT (datetime('now')),
  respondent_id    TEXT,                    -- hashed/pseudonymized (nullable)

  FOREIGN KEY(questionnaire_id) REFERENCES questionnaires(questionnaire_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_responses_questionnaire
ON responses(questionnaire_id);

CREATE INDEX IF NOT EXISTS idx_responses_submitted_at
ON responses(submitted_at);

-- D) EAV response values (one row per answered question)
CREATE TABLE IF NOT EXISTS response_values (
  value_id     TEXT PRIMARY KEY,
  response_id  TEXT NOT NULL,
  question_id  TEXT NOT NULL,

  value_type   TEXT NOT NULL,   -- numeric/text/date/json
  value_text   TEXT,
  value_num    REAL,
  value_date   TEXT,            -- ISO date or datetime
  value_json   TEXT,            -- JSON string

  created_at   TEXT NOT NULL DEFAULT (datetime('now')),

  FOREIGN KEY(response_id) REFERENCES responses(response_id) ON DELETE CASCADE,
  FOREIGN KEY(question_id) REFERENCES questionnaire_schema(question_id) ON DELETE CASCADE,
  UNIQUE(response_id, question_id)
);

CREATE INDEX IF NOT EXISTS idx_values_response
ON response_values(response_id);

CREATE INDEX IF NOT EXISTS idx_values_question
ON response_values(question_id);

-- E) Analysis runs (one row per user question/run)
CREATE TABLE IF NOT EXISTS analysis_runs (
  run_id            TEXT PRIMARY KEY,
  questionnaire_id  TEXT,
  created_at        TEXT NOT NULL DEFAULT (datetime('now')),
  user_question     TEXT NOT NULL,

  mapped_columns    TEXT,    -- JSON
  analysis_plan     TEXT,    -- JSON/TEXT
  code              TEXT,    -- Python code
  execution_status  TEXT,    -- pending/success/failed
  result_artifacts  TEXT,    -- JSON (stats, charts, paths)
  final_report      TEXT,    -- Markdown/Text

  FOREIGN KEY(questionnaire_id) REFERENCES questionnaires(questionnaire_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_created_at
ON analysis_runs(created_at);

-- F) Notes/state persistence (LangGraph state)
CREATE TABLE IF NOT EXISTS notes_state (
  run_id      TEXT PRIMARY KEY,
  state_json  TEXT NOT NULL,
  updated_at  TEXT NOT NULL DEFAULT (datetime('now')),

  FOREIGN KEY(run_id) REFERENCES analysis_runs(run_id) ON DELETE CASCADE
);
