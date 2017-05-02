function drawCharts() {
    var data = {"Category": {"sso": {"politik": 2005, "kommentarer": 1141, "indland": 11779, "section": 47, "foto": 10, "debat": 5931, "navne": 401, "indblik": 4929, "advertorial": 19, "kultur": 2459, "erhverv": 1780, "guide": 192, "viden": 1706, "sport": 8130, "anmeldelser": 2619, "article": 607, "international": 11263, "frontpage": 31, "jptv": 6, "mitjp": 1511, "aarhus": 5927, "timeout": 148, "briefing": 68, "other": 69300, "topic": 84, "livsstil": 6686, "abnjp": 1965, "download": 201}, "total": {"politik": 18359, "kommentarer": 3117, "faktameter": 1, "indland": 62775, "section": 159, "foto": 88, "debat": 53329, "viden": 9142, "indblik": 16798, "advertorial": 206, "kultur": 16127, "erhverv": 4867, "guide": 571, "topic": 548, "sport": 50082, "anmeldelser": 9417, "feature": 1, "article": 2013, "international": 73818, "frontpage": 152, "jptv": 224, "mitjp": 1511, "aarhus": 55898, "timeout": 1314, "briefing": 245, "other": 474016, "navne": 1338, "livsstil": 48259, "abnjp": 1965, "download": 203}, "primary": {"viden": 1130, "politik": 755, "kommentarer": 35, "indland": 271, "section": 7, "debat": 93, "navne": 8, "indblik": 161, "abnjp": 195, "erhverv": 22, "guide": 7, "topic": 42, "sport": 2832, "anmeldelser": 29, "article": 5, "international": 230, "mitjp": 140, "aarhus": 79, "timeout": 75, "briefing": 10, "other": 7631, "livsstil": 224, "kultur": 81, "download": 16}}, "Platform": {"sso": {"Computer": 54292, "Mobile": 46439, "Unspecified": 228, "Tablet": 39986}, "total": {"Computer": 291821, "Big screen": 29, "Mobile": 390598, "Unspecified": 1899, "Unknown": 5, "Tablet": 222191}, "primary": {"Computer": 5099, "Mobile": 4171, "Unspecified": 26, "Tablet": 4782}}, "Operating system": {"sso": {"iOS": 58434, "Android": 26070, "Windows": 43554, "Symbian": 12, "Windows Phone": 1778, "Windows RT": 17, "Linux": 551, "Mac OS": 10250, "Chromium OS": 279}, "total": {"iOS": 423594, "Symbian": 138, "FreeBSD": 3, "BlackBerry OS": 7, "Linux": 4275, "Xbox OS": 2, "Unknown": 12, "Java": 66, "Tizen": 9, "Android": 176823, "Windows": 232764, "Windows Phone": 12159, "PlayStation": 9, "Chromium OS": 1131, "Mac OS": 55366, "Unspecified": 1, "Windows RT": 184}, "primary": {"iOS": 6741, "Android": 1444, "Windows": 4566, "Symbian": 2, "Windows Phone": 182, "Windows RT": 1, "Linux": 36, "Mac OS": 1081, "Chromium OS": 25}}, "Device": {"sso": {"Android Tablet": 4969, "iPad": 34596, "Android Mobile": 20588, "Android Device": 449, "Chromium OS Desktop": 279, "Windows Desktop": 43212, "Unknown": 15, "Windows Tablet": 312, "Macintosh Desktop": 10250, "Kindle": 30, "iPod": 7, "BlackBerry": 19, "Windows Mobile": 1825, "Symbian Mobile": 12, "Linux Desktop": 551, "iPhone": 23831}, "total": {"Macintosh Desktop": 55366, "Android Laptop": 1, "Android Mobile": 143018, "Chromium OS Desktop": 1131, "Windows Desktop": 231049, "Windows Tablet": 1623, "Symbian Mobile": 138, "Kindle": 78, "BlackBerry": 82, "Linux Desktop": 4275, "Java Mobile": 63, "TV": 16, "Android Tablet": 23360, "iPad": 196260, "Android Desktop": 1, "Console": 11, "Unknown": 189, "Android Device": 10139, "iPod": 91, "Windows Mobile": 12405, "FreeBSD Desktop": 3, "Unspecified": 1, "iPhone": 227243}, "primary": {"Android Tablet": 380, "iPad": 3492, "Macintosh Desktop": 1081, "Android Device": 20, "Chromium OS Desktop": 25, "Windows Desktop": 4525, "Unknown": 2, "Windows Tablet": 39, "Android Mobile": 1039, "Symbian Mobile": 2, "Kindle": 2, "iPod": 2, "BlackBerry": 1, "Windows Mobile": 185, "Linux Desktop": 36, "iPhone": 3247}}, "Web browser": {"sso": {"Chrome for iOS": 3182, "Amazon Silk": 20, "IE": 10190, "Edge": 7995, "Chromium": 27, "SeaMonkey": 2, "Chrome": 51231, "Firefox": 6137, "Facebook for iPhone": 372, "Opera Mini": 1, "Facebook for Android": 151, "Android Browser": 235, "Vivaldi": 3, "Safari": 61282, "Firefox for iOS": 111, "Maxthon": 6}, "total": {"Chrome for iOS": 16323, "IE": 64221, "Edge": 42148, "Amazon Silk": 60, "Facebook for iPhone": 31056, "UC Browser": 9, "PaleMoon": 20, "Vienna": 1, "Opera Mini": 133, "Android Browser": 897, "Vivaldi": 118, "BlackBerry Browser": 7, "Firefox for iOS": 791, "Maxthon": 76, "Thunderbird": 6, "Chrome": 289442, "Chromium": 277, "SeaMonkey": 30, "NetFront": 9, "Firefox": 40869, "Unknown": 19, "Facebook for Android": 6090, "Iceweasel": 2, "Safari": 413874, "Opera": 65}, "primary": {"Chrome for iOS": 301, "Amazon Silk": 1, "IE": 1169, "Edge": 839, "Chromium": 5, "SeaMonkey": 1, "Chrome": 3813, "Firefox": 677, "Facebook for iPhone": 121, "Opera Mini": 1, "Facebook for Android": 54, "Android Browser": 18, "Vivaldi": 2, "Safari": 7061, "Firefox for iOS": 14, "Maxthon": 1}}};
    
    var platformData = data['Platform'];

    renderBarChart('#platform', platformData.total, 'Total');
    renderBarChart('#platform', platformData.sso, 'SSO only');
    renderBarChart('#platform', platformData.primary, 'Primary platform');

    var deviceData = data['Device'];
    
    renderBarChart('#device', deviceData.total, 'Total');
    renderBarChart('#device', deviceData.sso, 'SSO only');
    renderBarChart('#device', deviceData.primary, 'Primary device');

    var operatingSystemData = data['Operating system'];

    renderBarChart('#operating-system', operatingSystemData.total, 'Total');
    renderBarChart('#operating-system', operatingSystemData.sso, 'SSO only');
    renderBarChart('#operating-system', operatingSystemData.primary, 'Primary operating system');

    var webBrowserData = data['Web browser'];

    renderBarChart('#web-browser', webBrowserData.total, 'Total');
    renderBarChart('#web-browser', webBrowserData.sso, 'SSO only');
    renderBarChart('#web-browser', webBrowserData.primary, 'Primary web browser');

    var categoryData = data['Category'];

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
