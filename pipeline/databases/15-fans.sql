--ranks country origins of bands, by number of (non-unique)fans
-- Rank country origins by total number of fans
SELECT origin, SUM(fans) AS nb_fans
FROM metal_bands
GROUP BY origin
ORDER BY nb_fans DESC;
