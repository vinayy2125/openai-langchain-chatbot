-- Extend sessions table to track current_prompt_id
ALTER TABLE sessions
ADD COLUMN current_prompt_id INT REFERENCES prompts(id);
