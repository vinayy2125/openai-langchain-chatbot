-- Create prompts table
CREATE TABLE prompts (
    id SERIAL PRIMARY KEY,
    prompt_text TEXT NOT NULL,
    parent_id INT REFERENCES prompts(id) ON DELETE CASCADE,
    response_text TEXT,
    display_order INT DEFAULT 0
);
