

(One honest note: in _update, the diagnostic theory-locus redraw line is awkward — simplest is to delete the if self.show_diag: block inside _update and instead draw the locus once in _draw_static as a plain line; it's static per geometry anyway. Do that if the line errors.)

