-- creates a function SafeDiv that divides two numbers and returns 0 if the second number is zero
DROP FUNCTION IF EXISTS SafeDiv;
DELIMITER $$
CREATE FUNCTION SafeDiv(a INT, b INT) RETURNS FLOAT
BEGIN
    RETURN IF(b = 0, 0, a / b);
END$$
DELIMITER ;
