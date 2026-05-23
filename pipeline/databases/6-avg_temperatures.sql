-- Display average temperature by city in descending order
SELECT city, AVG(temperature) AS avg_temperature
FROM temperatures
GROUP BY city
ORDER BY avg_temperature DESC;
