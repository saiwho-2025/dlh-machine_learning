-- Create a stored procedure ComputeAverageScoreForUser that computes and stores a user's average score
DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser;

DELIMITER $$

CREATE PROCEDURE ComputeAverageScoreForUser(IN user_id INT)
BEGIN
    UPDATE users
    SET average_score = (
        SELECT IFNULL(AVG(score), 0)
        FROM corrections
        WHERE corrections.user_id = user_id
    )
    WHERE users.id = user_id;
END$$

DELIMITER ;