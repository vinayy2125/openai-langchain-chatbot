-- Extend users table with optional fields
ALTER TABLE users
ADD COLUMN first_name VARCHAR(255),
ADD COLUMN last_name VARCHAR(255),
ADD COLUMN email_opt_in BOOLEAN DEFAULT FALSE;
