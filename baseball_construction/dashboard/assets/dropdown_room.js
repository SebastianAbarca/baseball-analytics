/*
 * Keep dropdown menus opening downward.
 *
 * dcc.Dropdown in Dash 4 positions its menu with a floating-ui style flip: it
 * opens downward when there is room and upward when there is not. There is no
 * placement prop to pin it. The player pickers sit low in their cards, so on a
 * short window the control ends up ~13px from the bottom of the viewport and
 * the menu correctly, and unhelpfully, flips up over the card above it.
 *
 * Fighting the placement would mean overriding inline styles the library
 * recomputes on scroll. Giving it somewhere to go is simpler and leaves the
 * library's own logic intact: when a dropdown takes focus, scroll it far
 * enough up that a menu fits underneath, and it opens downward on its own.
 */
(function () {
  var MENU_ROOM = 320;   // px a menu may need; more than dcc's default maxHeight
  var MARGIN    = 16;

  function ensureRoom(el) {
    var box = el.getBoundingClientRect();
    var below = window.innerHeight - box.bottom;
    if (below >= MENU_ROOM) return;              // already fine, leave it alone
    // Only scroll by what is missing, so the control does not jump further
    // than necessary and the page keeps its place.
    var by = Math.min(MENU_ROOM - below + MARGIN,
                      document.documentElement.scrollHeight
                        - window.innerHeight - window.scrollY);
    // Instant, not smooth: the menu measures its available space as it opens,
    // and an animating scroll is still mid-flight at that moment, so it would
    // decide to flip up anyway and then slide with the page.
    if (by > 0) window.scrollBy({ top: by, behavior: "instant" });
  }

  document.addEventListener("focusin", function (ev) {
    var dd = ev.target && ev.target.closest && ev.target.closest(".dash-dropdown");
    if (dd) ensureRoom(dd);
  }, true);

  // A click on the trigger does not always move focus before the menu opens.
  document.addEventListener("mousedown", function (ev) {
    var dd = ev.target && ev.target.closest && ev.target.closest(".dash-dropdown");
    if (dd) setTimeout(function () { ensureRoom(dd); }, 0);
  }, true);
})();
