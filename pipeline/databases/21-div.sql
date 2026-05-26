-- creates a function SafeDiv that divides two numbers and returns 0 if the second number is zero

DELIMITTER $$


CREATE FUNCTION SafeDiv(a INT, b INT) 
RETURNS FLOAT
BEGIN
    IF b = 0 THEN
        RETURN 0;
    ELSE
        RETURN a / b;
    END IF;
END$$
DELIMITER ;
