/* ---------------------------------------------------------
   Patch a Leaflet.draw 1.0.4 bug.

   `L.GeometryUtil.readableArea` references an undeclared `type`, throwing
   "ReferenceError: type is not defined" the moment a polygon/rectangle has
   enough vertices to compute an area (with `showArea: true`). The throw happens
   inside the mousemove handler, which leaves the draw handler wedged — the
   symptom is a polygon that "maxes out" after three points.

   Redefine the function with the variable properly declared. Import this module
   once, AFTER `leaflet-draw`.
   Ref: https://github.com/Leaflet/Leaflet.draw/issues/1026
--------------------------------------------------------- */

import L from 'leaflet';

if (L.GeometryUtil) {
  L.GeometryUtil.readableArea = function (area, isMetric, precision) {
    const prec = L.Util.extend(
      {},
      { km: 2, ha: 2, m: 2, mi: 2, ac: 2, yd: 2, ft: 2 },
      precision,
    );
    let areaStr;

    if (isMetric) {
      let units = ['ha', 'm'];
      const type = typeof isMetric;
      if (type === 'string') {
        units = [isMetric];
      } else if (type !== 'boolean') {
        units = isMetric;
      }

      if (area >= 1000000 && units.indexOf('km') !== -1) {
        areaStr =
          L.GeometryUtil.formattedNumber(area * 0.000001, prec.km) + ' km²';
      } else if (area >= 10000 && units.indexOf('ha') !== -1) {
        areaStr = L.GeometryUtil.formattedNumber(area * 0.0001, prec.ha) + ' ha';
      } else {
        areaStr = L.GeometryUtil.formattedNumber(area, prec.m) + ' m²';
      }
    } else {
      area /= 0.836127; // m² -> yd²

      if (area >= 3097600) {
        areaStr =
          L.GeometryUtil.formattedNumber(area / 3097600, prec.mi) + ' mi²';
      } else if (area >= 4840) {
        areaStr = L.GeometryUtil.formattedNumber(area / 4840, prec.ac) + ' ac';
      } else {
        areaStr = L.GeometryUtil.formattedNumber(area, prec.yd) + ' yd²';
      }
    }

    return areaStr;
  };
}
