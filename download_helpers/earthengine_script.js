// Copy-paste this into Google Earth Engine Code Editor
// https://code.earthengine.google.com/


// Salinas Valley, California
var center_salinas = ee.Geometry.Point([-121.0, 36.6]);
var landsat_salinas = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterBounds(center_salinas)
    .filterDate('2023-06-15', '2023-12-31')
    .filter(ee.Filter.lt('CLOUD_COVER', 10))
    .sort('CLOUD_COVER').first();

var thermal_salinas = landsat_salinas.select('ST_B10')
    .multiply(0.00341802).add(149.0).subtract(273.15);

Export.image.toDrive({
    image: thermal_salinas,
    description: 'salinas_thermal',
    folder: 'HSI_Thermal',
    scale: 30,
    region: center_salinas.buffer(15000)
});


// San Diego Airport
var center_san_diego = ee.Geometry.Point([-117.19, 32.73]);
var landsat_san_diego = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterBounds(center_san_diego)
    .filterDate('2023-06-15', '2023-12-31')
    .filter(ee.Filter.lt('CLOUD_COVER', 10))
    .sort('CLOUD_COVER').first();

var thermal_san_diego = landsat_san_diego.select('ST_B10')
    .multiply(0.00341802).add(149.0).subtract(273.15);

Export.image.toDrive({
    image: thermal_san_diego,
    description: 'san_diego_thermal',
    folder: 'HSI_Thermal',
    scale: 30,
    region: center_san_diego.buffer(15000)
});

