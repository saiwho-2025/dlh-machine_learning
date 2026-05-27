-- Create an index on only the first letter of name
CREATE INDEX idx_na_first
ON names (name(1));
