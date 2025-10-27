// =================================================================
// --- CONFIGURATION ---
// PASTE THE CONFIGURATION FOR YOUR DESIRED LOCATION HERE
// =================================================================
var locationName = 'san_diego';
var centerPoint = ee.Geometry.Point([-117.16, 32.71]); // San Diego, CA
var bufferRadius = 20000; // 20km radius

// =================================================================
// --- SCRIPT LOGIC (No need to change below) ---
// =================================================================

// --- Define Date Range and Cloud Cover ---
var START_DATE = '2021-01-01';
var END_DATE = '2023-12-31';
var CLOUD_COVER_MAX = 15;

// --- Function to process Landsat images ---
function processLandsat(img) {
  var thermal = img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST').toFloat();
  return thermal.copyProperties(img, ['system:time_start']);
}

// --- Get Landsat 8 & 9 Collections ---
var landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(centerPoint).filterDate(START_DATE, END_DATE).filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_MAX)).map(processLandsat);
var landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(centerPoint).filterDate(START_DATE, END_DATE).filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_MAX)).map(processLandsat);
var thermalCollection = ee.ImageCollection(landsat8.merge(landsat9));
print('Found ' + thermalCollection.size().getInfo() + ' clear images for ' + locationName);

// --- Get FIRMS Fire Mask as Ground Truth ---
var firms = ee.ImageCollection('FIRMS').filterDate(START_DATE, END_DATE).filterBounds(centerPoint);
var fireMask = firms.map(function(img) { return img.select('T21').gt(0); }).max().rename('FIRE_MASK').toFloat();

// --- Map Preview ---
Map.centerObject(centerPoint, 9);
Map.addLayer(thermalCollection.mean(), {min: 10, max: 40, palette: ['blue','cyan','yellow','red']}, 'Mean LST');
Map.addLayer(fireMask, {min: 0, max: 1, palette: ['white','orange']}, 'Fire Mask (GT)');

// --- Export Logic ---
var list = thermalCollection.toList(thermalCollection.size());
var count = thermalCollection.size().getInfo();
var exportCount = Math.min(count, 20); // Limit to 20 images per location
var region = centerPoint.buffer(bufferRadius);

for (var i = 0; i < exportCount; i++) {
  var image = ee.Image(list.get(i));
  var date = ee.Date(image.get('system:time_start')).format('YYYYMMdd').getInfo();
  var merged = image.addBands(fireMask).toFloat();

  Export.image.toDrive({
    image: merged,
    description: locationName + '_thermal_gt_' + date,
    folder: 'HSI_Thermal_Exports',
    region: region,
    scale: 30,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
}

print('✅ Export script is ready for '  + locationName);
print('Go to the "Tasks" tab and click RUN for each export.');


// For san_diego, abu_airport, abu_beach, abu_urban
// These datasets are all from the San Diego, CA area. Use this configuration for all of them:
// 
// 
//  1 var locationName = 'san_diego';
//  2 var centerPoint = ee.Geometry.Point([-117.16, 32.71]); // San Diego, CA
//  3 var bufferRadius = 20000; // 20km radius
// 
// 
// For Salinas
// This dataset is from the Salinas Valley, CA. Use this configuration:
// 
// 
//  1 var locationName = 'salinas';
//  2 var centerPoint = ee.Geometry.Point([-121.65, 36.67]); // Salinas Valley, CA
//  3 var bufferRadius = 20000; // 20km radius
// 
// 
// For hydice_urban
// This dataset is from Copperas Cove, TX. Use this configuration:
// 
// 
// var locationName = 'hydice_urban';
// var centerPoint = ee.Geometry.Point([-97.92, 31.12]); // Copperas Cove, TX
// var bufferRadius = 10000; // 10km radius
// 
// 