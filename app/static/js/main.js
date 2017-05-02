$(document).ready(function() {
    $('.datepicker').pickadate({
        firstDay: 1,
        max: new Date()
    });

    $('.timepicker').pickatime({
        format: 'HH:i'
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
        var configuration = {};
        var filters = getFilters();
        configuration.filters = filters;
        console.log(JSON.stringify(configuration));
        ajax(configuration);
        $('#result-page').show();
        $('html, body').animate({ scrollTop: $('#result-page').offset().top }, 1500);
    });
});

function getFilters() {
    var filters = [];
    $('.filter-section').each(function() {
        var attributeName = $(this).attr('data-attribute-name');
        var $selectedFilters = $(this).find('.selected');
        var selectedFilterNames = [];
        $selectedFilters.each(function() {
            selectedFilterNames.push($(this).attr('data-value-name'));
        });

        if (selectedFilterNames.length > 0) {
            filters.push({filterName: attributeName, selected: selectedFilterNames});
        }
    });
    return filters;
}

function ajax(filters) {
    $.post({
        url: 'http://localhost:5000/' + 'get-result',
        data: JSON.stringify(filters),
        datatype: 'json',
        contentType: 'application/json',
        success: function(data, textStatus, jqXHR) {
            drawCharts(data);
        },
        error: function (jqXHR, textStatus, errorThrown) {
            console.log(textStatus+" "+errorThrown)
        }
    });
}
