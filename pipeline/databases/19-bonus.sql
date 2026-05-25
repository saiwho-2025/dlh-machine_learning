-- create a stored procedure AddBonus that adds a new correction for a student
CREATE PROCEDURE AddBonus(
    IN p_user_id INT,
    IN p_project_name VARCHAR(255),
    IN p_score INT
)
BEGIN
    -- Insert project if it does not exist
    INSERT INTO projects (name)
    SELECT p_project_name
    WHERE NOT EXISTS (
        SELECT 1 FROM projects WHERE name = p_project_name
    );

    -- Insert correction for the student
    INSERT INTO corrections (user_id, project_id, score)
    VALUES (
        p_user_id,
        (SELECT id FROM projects WHERE name = p_project_name),
        p_score
    );
END;