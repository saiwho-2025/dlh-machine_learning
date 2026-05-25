-- Create trigger to reset valid_email when email changes
CREATE TRIGGER reset_valid_email
BEFORE UPDATE ON users
FOR EACH ROW
SET NEW.valid_email = IF(OLD.email <> NEW.email, 0, OLD.valid_email);
