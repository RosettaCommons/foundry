// "Was this page helpful?" feedback, logged to Simple Analytics.
// The inline widget at the bottom of each page logs "<prefix>_page_feedback".
// The popup logs "<prefix>_page_feedback_popup" so its data is kept separate.
// After a thumbs down, visitors can pick an optional reason and leave an optional
// comment, which are sent to a Google Form in the background (Simple Analytics
// doesn't allow free text that might contain personal data).
//
// This widget is shared by several docs sites on rosettacommons.github.io.
// Each site sets its own FEEDBACK_EVENT_PREFIX; everything else can be the same.

// Start of the Simple Analytics event names for this site.
var FEEDBACK_EVENT_PREFIX = "foundry";

// Google Form that receives reasons and comments. Leave FEEDBACK_FORM_URL empty
// to turn them off. To fill these in, see the "Get pre-filled link" option in the
// form's menu: each question's field name looks like "entry.123456789".
var FEEDBACK_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSe5esGvNU1nM-t2wwNTsXTK3FE2p2MfAS-RMQUwWtX3AcRmng/formResponse";
var FEEDBACK_FORM_FIELDS = {
  reason: "entry.72427035", // multiple-choice question whose options match the reasons in page.html exactly
  comment: "entry.1352834238", // paragraph question for the comment
  site: "entry.2123336275", // short-answer question for the site (repo) name
  page: "entry.927082595", // short-answer question for the page path
  source: "entry.948484219", // short-answer question for "inline" or "popup"
};

// The popup only runs for one week every three months: the 15th-21st of
// November, February, May and August, starting 15 November 2026.
var POPUP_FIRST_DAY = new Date(2026, 10, 15);
var POPUP_MONTHS = [1, 4, 7, 10]; // Feb, May, Aug, Nov (0-indexed)
var POPUP_DAYS = 7;
var POPUP_DELAY_MS = 8000; // time on the page before the popup opens

// Returns an id for the current popup window (e.g. "2026-11"), or null if
// today is outside every window.
function currentPopupWindow(today) {
  if (today < POPUP_FIRST_DAY) return null;
  var day = today.getDate();
  if (POPUP_MONTHS.indexOf(today.getMonth()) === -1) return null;
  if (day < POPUP_FIRST_DAY.getDate() || day >= POPUP_FIRST_DAY.getDate() + POPUP_DAYS) return null;
  return today.getFullYear() + "-" + (today.getMonth() + 1);
}

// The site's repo name, from the first part of the path on GitHub Pages
// (rosettacommons.github.io/RFDpoly/... gives "RFDpoly"). Elsewhere, such as
// a local build, falls back to the hostname or "local".
function currentSite() {
  var firstPart = window.location.pathname.split("/")[1];
  if (/\.github\.io$/.test(window.location.hostname) && firstPart && firstPart.indexOf(".") === -1) {
    return firstPart;
  }
  return window.location.hostname || "local";
}

// localStorage can be unavailable (blocked storage, some private windows).
function storageGet(key) {
  try { return window.localStorage.getItem(key); } catch (e) { return null; }
}
function storageSet(key, value) {
  try { window.localStorage.setItem(key, value); } catch (e) {}
}

function sendComment(widget, reason, comment) {
  var data = new URLSearchParams();
  var values = {
    reason: reason,
    comment: comment,
    site: currentSite(),
    page: window.location.pathname,
    source: widget.dataset.source,
  };
  Object.keys(FEEDBACK_FORM_FIELDS).forEach(function (key) {
    if (FEEDBACK_FORM_FIELDS[key] && values[key]) data.append(FEEDBACK_FORM_FIELDS[key], values[key]);
  });
  // Google Forms doesn't allow reading the response, so this is fire-and-forget.
  // keepalive lets the request finish if the visitor leaves the page.
  fetch(FEEDBACK_FORM_URL, { method: "POST", mode: "no-cors", body: data, keepalive: true })
    .catch(function () {});
}

// onAnswer runs when a thumb is clicked; onDone runs once the widget shows
// its thank-you message (after the optional reason and comment for a thumbs down).
function setUpFeedback(widget, eventName, onAnswer, onDone) {
  var commentForm = widget.querySelector(".page-feedback-comment");
  var commentsEnabled = FEEDBACK_FORM_URL && (FEEDBACK_FORM_FIELDS.reason || FEEDBACK_FORM_FIELDS.comment);

  function showThanks() {
    commentForm.hidden = true;
    widget.querySelector(".page-feedback-thanks").hidden = false;
    if (onDone) onDone();
  }

  widget.querySelectorAll(".page-feedback-button").forEach(function (button) {
    button.addEventListener("click", function () {
      if (typeof window.sa_event === "function") {
        window.sa_event(eventName, {
          helpful: button.dataset.helpful,
          site: currentSite(),
          page: window.location.pathname,
        });
      }
      widget.querySelector(".page-feedback-question").hidden = true;
      widget.querySelector(".page-feedback-buttons").hidden = true;
      if (onAnswer) onAnswer();
      if (button.dataset.helpful === "no" && commentsEnabled) {
        commentForm.hidden = false;
        commentForm.querySelector("input, textarea").focus();
      } else {
        showThanks();
      }
    });
  });

  commentForm.addEventListener("submit", function (event) {
    event.preventDefault();
    var checked = commentForm.querySelector("input[type=radio]:checked");
    var reason = checked ? checked.value : "";
    var comment = commentForm.querySelector("textarea").value.trim();
    if (reason || comment) sendComment(widget, reason, comment);
    showThanks();
  });
  widget.querySelector(".page-feedback-skip").addEventListener("click", showThanks);
}

document.addEventListener("DOMContentLoaded", function () {
  var popup = document.querySelector(".page-feedback-popup");
  var popupWindow = currentPopupWindow(new Date());
  // The sites share one domain, so the key includes the site to keep each
  // site's popup separate.
  var popupKey = "page_feedback_popup_seen_" + currentSite() + "_" + popupWindow;
  var popupAllowed = popup && typeof popup.showModal === "function" && popupWindow && !storageGet(popupKey);

  // Don't pop up asking for feedback someone already gave inline, on this
  // page or any later page during the same popup window.
  document.querySelectorAll(".page-feedback").forEach(function (widget) {
    setUpFeedback(widget, FEEDBACK_EVENT_PREFIX + "_page_feedback", function () {
      popupAllowed = false;
      if (popupWindow) storageSet(popupKey, "1");
    }, null);
  });

  if (!popupAllowed) return;

  // Each visitor sees the popup at most once per site per popup window.
  function closePopup() {
    popup.close();
  }
  setUpFeedback(popup, FEEDBACK_EVENT_PREFIX + "_page_feedback_popup", null, function () {
    setTimeout(closePopup, 2000);
  });
  popup.querySelector(".page-feedback-popup-close").addEventListener("click", closePopup);
  // Clicking the dimmed backdrop outside the popup also closes it. Require the
  // press to start on the backdrop too, so a text selection that is dragged
  // out of the textarea doesn't close the popup and lose the comment.
  var pressStartedOnBackdrop = false;
  popup.addEventListener("mousedown", function (event) {
    pressStartedOnBackdrop = event.target === popup;
  });
  popup.addEventListener("click", function (event) {
    if (event.target === popup && pressStartedOnBackdrop) closePopup();
    pressStartedOnBackdrop = false;
  });

  setTimeout(function () {
    if (!popupAllowed) return;
    storageSet(popupKey, "1");
    popup.showModal();
  }, POPUP_DELAY_MS);
});
