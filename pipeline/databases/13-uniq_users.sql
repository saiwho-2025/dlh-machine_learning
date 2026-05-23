-- create a table users with attributes id, name, email, created_at
CREATE TABLE if NOT EXISTS users (
    id INT NOT NULL AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255),
    PRIMARY KEY (id)
);
  