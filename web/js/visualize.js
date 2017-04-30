function drawCharts() {
    var platformData = {
        'total': {'Computer': 291821,
                  'Unknown': 5,
                  'Unspecified': 1899,
                  'Mobile': 390598,
                  'Tablet': 222191,
                  'Big screen': 29},
        'sso': {'Unspecified': 228,
                'Mobile': 46439,
                'Tablet': 39986,
                'Computer': 54292},
        'primary': {'Unspecified': 26,
                    'Mobile': 4171,
                    'Tablet': 4782,
                    'Computer': 5099}
    };

    renderBarChart('#platform', platformData.total, 'Total');
    renderBarChart('#platform', platformData.sso, 'SSO only');
    renderBarChart('#platform', platformData.primary, 'Primary platform');

    var deviceData = {'total': {'Kindle': 78, 'Linux Desktop': 4275, 'Java Mobile': 63, 'Android Tablet': 23360, 'Android Desktop': 1, 'iPhone': 227243, 'Android Mobile': 143018, 'Windows Mobile': 12405, 'Symbian Mobile': 138, 'Windows Tablet': 1623, 'Unspecified': 1, 'Windows Desktop': 231049, 'TV': 16, 'Macintosh Desktop': 55366, 'Android Laptop': 1, 'iPad': 196260, 'iPod': 91, 'Console': 11, 'Chromium OS Desktop': 1131, 'FreeBSD Desktop': 3, 'Android Device': 10139, 'BlackBerry': 82, 'Unknown': 189}, 'sso': {'Kindle': 30, 'Macintosh Desktop': 10250, 'Android Tablet': 4969, 'iPad': 34596, 'iPod': 7, 'Unknown': 15, 'Chromium OS Desktop': 279, 'Linux Desktop': 551, 'Android Device': 449, 'Android Mobile': 20588, 'Windows Mobile': 1825, 'Symbian Mobile': 12, 'Windows Tablet': 312, 'iPhone': 23831, 'Windows Desktop': 43212, 'BlackBerry': 19}, 'primary': {'Kindle': 2, 'Macintosh Desktop': 1081, 'Android Tablet': 380, 'iPad': 3492, 'iPod': 2, 'Unknown': 2, 'Chromium OS Desktop': 25, 'Linux Desktop': 36, 'Android Device': 20, 'Android Mobile': 1039, 'Windows Mobile': 185, 'Symbian Mobile': 2, 'Windows Tablet': 39, 'iPhone': 3247, 'Windows Desktop': 4525, 'BlackBerry': 1}};
    
    renderBarChart('#device', deviceData.total, 'Total');
    renderBarChart('#device', deviceData.sso, 'SSO only');
    renderBarChart('#device', deviceData.primary, 'Primary device');

    var operatingSystemData = {'total': {'BlackBerry OS': 7, 'Xbox OS': 2, 'FreeBSD': 3, 'Windows RT': 184, 'Java': 66, 'Windows': 232764, 'Chromium OS': 1131, 'Android': 176823, 'Tizen': 9, 'Unspecified': 1, 'iOS': 423594, 'Unknown': 12, 'Linux': 4275, 'PlayStation': 9, 'Windows Phone': 12159, 'Symbian': 138, 'Mac OS': 55366}, 'sso': {'Chromium OS': 279, 'Android': 26070, 'Linux': 551, 'iOS': 58434, 'Windows RT': 17, 'Windows Phone': 1778, 'Symbian': 12, 'Windows': 43554, 'Mac OS': 10250}, 'primary': {'Chromium OS': 25, 'Android': 1444, 'Linux': 36, 'iOS': 6741, 'Windows RT': 1, 'Windows Phone': 182, 'Symbian': 2, 'Windows': 4566, 'Mac OS': 1081}};

    renderBarChart('#operating-system', operatingSystemData.total, 'Total');
    renderBarChart('#operating-system', operatingSystemData.sso, 'SSO only');
    renderBarChart('#operating-system', operatingSystemData.primary, 'Primary operating system');

    var webBrowserData = {'total': {'Facebook for iPhone': 31056, 'Thunderbird': 6, 'Iceweasel': 2, 'UC Browser': 9, 'Firefox': 40869, 'Firefox for iOS': 791, 'IE': 64221, 'Vivaldi': 118, 'Vienna': 1, 'Edge': 42148, 'PaleMoon': 20, 'Chrome': 289442, 'Android Browser': 897, 'Opera': 65, 'Facebook for Android': 6090, 'Chrome for iOS': 16323, 'Safari': 413874, 'Maxthon': 76, 'BlackBerry Browser': 7, 'Chromium': 277, 'Amazon Silk': 60, 'Unknown': 19, 'NetFront': 9, 'SeaMonkey': 30, 'Opera Mini': 133}, 'sso': {'Facebook for iPhone': 372, 'IE': 10190, 'Facebook for Android': 151, 'Chrome for iOS': 3182, 'Firefox for iOS': 111, 'Firefox': 6137, 'Safari': 61282, 'Maxthon': 6, 'Vivaldi': 3, 'Edge': 7995, 'Chromium': 27, 'Amazon Silk': 20, 'SeaMonkey': 2, 'Chrome': 51231, 'Android Browser': 235, 'Opera Mini': 1}, 'primary': {'Facebook for iPhone': 121, 'IE': 1169, 'Facebook for Android': 54, 'Chrome for iOS': 301, 'Firefox for iOS': 14, 'Firefox': 677, 'Safari': 7061, 'Maxthon': 1, 'Vivaldi': 2, 'Edge': 839, 'Chromium': 5, 'Amazon Silk': 1, 'SeaMonkey': 1, 'Chrome': 3813, 'Android Browser': 18, 'Opera Mini': 1}};

    renderBarChart('#web-browser', webBrowserData.total, 'Total');
    renderBarChart('#web-browser', webBrowserData.sso, 'SSO only');
    renderBarChart('#web-browser', webBrowserData.primary, 'Primary web browser');

    var categoryData = {'total': {'foto': 88, 'guide': 571, 'debat': 53329, 'indland': 62775, 'sport': 50082, 'other': 474016, 'erhverv': 4867, 'advertorial': 206, 'livsstil': 48259, 'topic': 548, 'international': 73818, 'timeout': 1314, 'anmeldelser': 9417, 'feature': 1, 'download': 203, 'viden': 9142, 'aarhus': 55898, 'kultur': 16127, 'indblik': 16798, 'abnjp': 1965, 'mitjp': 1511, 'briefing': 245, 'navne': 1338, 'faktameter': 1, 'jptv': 224, 'frontpage': 152, 'kommentarer': 3117, 'article': 2013, 'politik': 18359, 'section': 159}, 'sso': {'foto': 10, 'guide': 192, 'debat': 5931, 'indland': 11779, 'sport': 8130, 'other': 69300, 'erhverv': 1780, 'advertorial': 19, 'livsstil': 6686, 'topic': 84, 'download': 201, 'timeout': 148, 'anmeldelser': 2619, 'international': 11263, 'viden': 1706, 'aarhus': 5927, 'kultur': 2459, 'indblik': 4929, 'abnjp': 1965, 'mitjp': 1511, 'briefing': 68, 'navne': 401, 'jptv': 6, 'frontpage': 31, 'kommentarer': 1141, 'article': 607, 'politik': 2005, 'section': 47}, 'primary': {'guide': 7, 'debat': 93, 'indland': 271, 'sport': 2832, 'other': 7631, 'erhverv': 22, 'livsstil': 224, 'topic': 42, 'download': 16, 'timeout': 75, 'anmeldelser': 29, 'international': 230, 'viden': 1130, 'aarhus': 79, 'kultur': 81, 'abnjp': 195, 'indblik': 161, 'mitjp': 140, 'briefing': 10, 'navne': 8, 'kommentarer': 35, 'article': 5, 'politik': 755, 'section': 7}};

    renderBarChart('#category', categoryData.total, 'Total');
    renderBarChart('#category', categoryData.sso, 'SSO only');
    renderBarChart('#category', categoryData.primary, 'Primary category');
};

// D3
function renderBarChart(containerID, barchartdata, chartname) {
    var $svg = $('<svg class="chart" width="380" height="450" viewBox="0 0 380 450" preerveAspectRatio="xMinYMin meet"></svg>');
    $(containerID).append($svg);
    var svg = d3.select($svg[0]),
        margin = {top: 40, right: 20, bottom: 150, left: 50},
        width = +svg.attr("width") - margin.left - margin.right,
        height = +svg.attr("height") - margin.top - margin.bottom;

    var x = d3.scaleBand().rangeRound([0, width]).padding(0.1),
        y = d3.scaleLinear().rangeRound([height, 0]);

    var g = svg.append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
    
    x.domain(Object.keys(barchartdata));
    y.domain([0, d3.max(Object.values(barchartdata))]);

    g.append("g")
        .attr("class", "axis axis--x")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x))
        .selectAll("text")
        .attr("y", 0)
        .attr("x", 9)
        .attr("dy", ".35em")
        .attr("transform", "rotate(90)")
        .style("text-anchor", "start");

    g.append("g")
        .attr("class", "axis axis--y")
        .call(d3.axisLeft(y).ticks(10, "").tickFormat(d3.format(",d")))
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dy", "0.71em")
        .attr("text-anchor", "end")
        .text("Frequency");

    g.selectAll(".bar")
        .data(Object.values(barchartdata))
        .enter().append("rect")
        .attr("class", "bar")
        .attr("x", function(d, i) { return x(Object.keys(barchartdata)[i]); })
        .attr("y", function(d, i) { return y(Object.values(barchartdata)[i]); })
        .attr("width", x.bandwidth())
        .attr("height", function(d) { return height - y(d); });

    g.append("text")
        .attr("x", (width / 2))             
        .attr("y", 0 - (margin.top / 2))
        .attr("text-anchor", "middle")  
        .style("font-size", "16px") 
        .text(chartname);
};
