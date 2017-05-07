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
        // remove any previous results
        $('.result-box').empty();
        var configuration = {};
        var filters = getFilters();
        configuration.filters = filters;
        console.log(JSON.stringify(configuration));
        requestResult(configuration);
        $('#result-page').show();
        $('html, body').animate({ scrollTop: $('#result-page').offset().top }, 1500);
        insertLoadingIcons();
    });

    $('#reset-filters').click(function() {
        $('.selected').removeClass('selected');
        $('.default').addClass('selected');
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

function requestResult(filters) {
    $.post({
        url: 'http://localhost:5000/' + 'get-result',
        data: JSON.stringify(filters),
        datatype: 'json',
        contentType: 'application/json',
        success: function(data, textStatus, jqXHR) {
            loadingIcons.forEach(function(icon) {
                icon.loading = false;
            });
            $('.result-box').empty();
            drawCharts(data);
        },
        error: function (jqXHR, textStatus, errorThrown) {
            console.log(textStatus+" "+errorThrown)
        }
    });
}

function LoadingIcon() {
    this.loading = true;
    this.representation = $('<svg xmlns="http://www.w3.org/2000/svg" class="loading-icon" viewBox="2 2 26 26"><circle cx="22" cy="8" r="1.125"/><circle cx="25" cy="15" r="1.25"/><circle cx="22" cy="22" r="1.375"/><circle cx="15" cy="25" r="1.5"/><circle cx="8" cy="22" r="1.625"/><circle cx="5" cy="15" r="1.75"/><circle cx="8" cy="8" r="1.875"/><circle cx="15" cy="5" r="2"/></svg>');
    this.showLoadingAnimation = function() {
        var circles = $(this.representation).find("circle");
        var firstCircle = circles.first();
        var lastCircle = circles.last();

        // make the last circle become the first circle
        firstCircle.detach().insertAfter(lastCircle);

        // update radii
        circles.each(function(index, circle) {
            var radius = 1.125 + 0.18 * index;
            $(circle).attr("r", radius);
        });
        var timer = null;
        if (this.loading)
            timer = setTimeout(this.showLoadingAnimation, 125);
        else {
            (this.representation).remove();
            clearTimeout(timer);
        }
    }.bind(this);
}

var loadingIcons = null;

function insertLoadingIcons() {
    var li = [];
    $('.result-box').each(function() {
        var loadingIcon = new LoadingIcon();
        $(this).append(loadingIcon.representation);
        loadingIcon.showLoadingAnimation();
        li.push(loadingIcon);
    });
    loadingIcons = li;
    return loadingIcons;
}
