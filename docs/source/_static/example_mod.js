// This contains code with copyright by the scikit-learn project, subject to
// the license in /thirdparty/LICENSES/LICENSE.scikit_learn

$(document).ready(function () {
   /* Add a [>>>] button on the top-right corner of code samples to hide
    * the >>> and ... prompts and the output and thus make the code
    * copyable. */
   var div = $('.highlight-python .highlight,' +
      '.highlight-python3 .highlight,' +
      '.highlight-pycon .highlight,' +
      '.highlight-default .highlight')
   var pre = div.find('pre');

   // get the styles from the current theme
   pre.parent().parent().css('position', 'relative');
   var hide_text = 'Hide prompts and outputs';
   var show_text = 'Show prompts and outputs';

   // create and add the button to all the code blocks that contain >>>
   div.each(function (index) {
      var jthis = $(this);
      if (jthis.find('.gp').length > 0) {
         var button = $('<span class="copybutton">&gt;&gt;&gt;</span>');
         button.attr('title', hide_text);
         button.data('hidden', 'false');
         jthis.prepend(button);
      }
      // tracebacks (.gt) contain bare text elements that need to be
      // wrapped in a span to work with .nextUntil() (see later)
      jthis.find('pre:has(.gt)').contents().filter(function () {
         return ((this.nodeType == 3) && (this.data.trim().length > 0));
      }).wrap('<span>');
   });

   // define the behavior of the button when it's clicked
   $('.copybutton').click(function (e) {
      e.preventDefault();
      var button = $(this);
      if (button.data('hidden') === 'false') {
         // hide the code output
         button.parent().find('.go, .gp, .gt').hide();
         button.next('pre')
            .find('.gt')
            .nextUntil('.gp, .go')
            .css('visibility', 'hidden');
         button.css('text-decoration', 'line-through');
         button.attr('title', show_text);
         button.data('hidden', 'true');
      } else {
         // show the code output
         button.parent().find('.go, .gp, .gt').show();
         button.next('pre')
            .find('.gt')
            .nextUntil('.gp, .go')
            .css('visibility', 'visible');
         button.css('text-decoration', 'none');
         button.attr('title', hide_text);
         button.data('hidden', 'false');
      }
   });
});