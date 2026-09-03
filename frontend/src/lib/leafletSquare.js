/* Square draw tool for Leaflet.draw.

   Leaflet.draw 1.0.4 has no square mode, so this subclasses the rectangle
   handler and constrains the dragged corner. Import once, AFTER `leaflet-draw`
   (and the fix in leafletDrawFix.js); then enable it in the draw control
   options like any built-in tool: `draw: { square: { showArea: true } }`.
   The toolbar button has no sprite icon — MapView.module.css draws its glyph. */

import L from 'leaflet';

L.Draw.Square = L.Draw.Rectangle.extend({
  statics: { TYPE: 'square' },

  initialize(map, options) {
    L.Draw.Rectangle.prototype.initialize.call(this, map, options);
    // After the parent sets 'rectangle' — the type names the toolbar button
    // class and the layerType on the created event.
    this.type = L.Draw.Square.TYPE;
  },

  // Constrain the dragged corner so the shape stays square on screen. Web
  // Mercator is conformal, so a screen square is a ground square to within
  // rounding at the map-area scales this app works at.
  _drawShape(latlng) {
    const map = this._map;
    const start = map.latLngToLayerPoint(this._startLatLng);
    const cur = map.latLngToLayerPoint(latlng);
    const dx = cur.x - start.x;
    const dy = cur.y - start.y;
    const side = Math.max(Math.abs(dx), Math.abs(dy));
    const corner = map.layerPointToLatLng(
      L.point(
        start.x + (dx < 0 ? -side : side),
        start.y + (dy < 0 ? -side : side),
      ),
    );
    L.Draw.Rectangle.prototype._drawShape.call(this, corner);
  },
});

// Splice the square tool into the draw toolbar, right after the rectangle.
const getModeHandlers = L.DrawToolbar.prototype.getModeHandlers;
L.DrawToolbar.prototype.getModeHandlers = function (map) {
  const handlers = getModeHandlers.call(this, map);
  const iRect = handlers.findIndex((h) => h.handler.type === 'rectangle');
  handlers.splice(iRect + 1, 0, {
    enabled: this.options.square,
    handler: new L.Draw.Square(map, this.options.square),
    title: 'Draw a square',
  });
  return handlers;
};
