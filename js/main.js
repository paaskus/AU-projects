$(document).ready(function() {
    $('.datepicker').pickadate({
        firstDay: 1,
        max: new Date()
    });

    $('.filter-option').click(function() {
        $(this).toggleClass('selected');
    });

    $('.tri-state-option').click(function() {
        $('.tri-state-option').removeClass('selected');
        $(this).addClass('selected');
    });

    $('#result-page').hide();
    $('#result-button').click(function() {
        $('#result-page').show();
        $('html, body').animate({ scrollTop: $('#result-page').offset().top }, 1500);
    });
});
