-- List Glam rock bands ranked by lifespan
SELECT band_name,
       IF(split IS NULL, 2020, split) - formed AS `lifespan until 2020 (in years)`
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY `lifespan until 2020 (in years)` DESC;
