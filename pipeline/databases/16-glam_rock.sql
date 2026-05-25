--list bands and rank them
SELECT band_name, formed and split 
FROM metal_bands
WHERE genre = 'glam rock'
lifespan = split - formed
ORDER BY lifespan DESC;
